import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint

from weasel.constants import (
    CHANNELS,
    LATENTS,
    NEURONS,
    DOWN_FACTOR,
    MODE,
)

class ArrayCondDataset(Dataset):
    def __init__(self, X: np.ndarray, M: np.ndarray):
        assert len(X) == len(M)
        self.X = X
        self.M = M

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # ensure float32 + WRITABLE (copy only if needed; cheap vs a crash)
        x_np = np.asarray(self.X[idx], dtype=np.float32)
        if not x_np.flags.writeable:
            x_np = x_np.copy()
        m_np = np.asarray(self.M[idx], dtype=np.float32)
        if not m_np.flags.writeable:
            m_np = m_np.copy()
        x = torch.from_numpy(x_np)
        m = torch.from_numpy(m_np)
        return x, m


class ShardedDataset(Dataset):
    """
    Dataset that loads data from sharded .npy files.
    Each shard contains a batch of samples with corresponding metadata.
    """
    def __init__(self, split_dir: str):
        import json
        from pathlib import Path
        
        self.split_dir = Path(split_dir)
        
        # Load manifest
        manifest_path = self.split_dir / 'manifest.json'
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)
        
        self.total_samples = self.manifest['total_samples']
        self.samples_per_shard = self.manifest['samples_per_shard']
        self.n_shards = self.manifest['n_shards']
        
        # Build index: global_idx -> (shard_idx, local_idx)
        self.shard_info = []
        cumulative = 0
        for shard in self.manifest['shards']:
            n_samples = shard['n_samples']
            self.shard_info.append({
                'start': cumulative,
                'end': cumulative + n_samples,
                'shard_idx': shard['shard_idx'],
                'n_samples': n_samples
            })
            cumulative += n_samples
        
        # Cache for currently loaded shard
        self._cached_shard_idx = None
        self._cached_data = None
        self._cached_meta = None
    
    def __len__(self):
        return self.total_samples
    
    def _load_shard(self, shard_idx: int):
        """Load a shard into cache."""
        if self._cached_shard_idx == shard_idx:
            return
        
        data_path = self.split_dir / f'shard_{shard_idx:04d}_data.npy'
        meta_path = self.split_dir / f'shard_{shard_idx:04d}_meta.npy'
        
        self._cached_data = np.load(data_path, mmap_mode='r')
        self._cached_meta = np.load(meta_path, mmap_mode='r')
        self._cached_shard_idx = shard_idx
    
    def _get_shard_idx(self, global_idx: int):
        """Find which shard contains the global index."""
        for info in self.shard_info:
            if info['start'] <= global_idx < info['end']:
                return info['shard_idx'], global_idx - info['start']
        raise IndexError(f"Index {global_idx} out of range")
    
    def __getitem__(self, idx):
        shard_idx, local_idx = self._get_shard_idx(idx)
        self._load_shard(shard_idx)
        
        x_np = np.asarray(self._cached_data[local_idx], dtype=np.float32)
        if not x_np.flags.writeable:
            x_np = x_np.copy()
        
        m_np = np.asarray(self._cached_meta[local_idx], dtype=np.float32)
        if not m_np.flags.writeable:
            m_np = m_np.copy()
        
        x = torch.from_numpy(x_np)
        m = torch.from_numpy(m_np)
        return x, m
    
    def get_sample_shape(self):
        """Return (C, H, W) shape of a single sample."""
        self._load_shard(0)
        return self._cached_data.shape[1:]
    
    def get_meta_dim(self):
        """Return dimension of metadata vector."""
        self._load_shard(0)
        return self._cached_meta.shape[1]
        
class ModGNConv(nn.Module):
    """GroupNorm -> (optional FiLM) -> SiLU -> Conv"""
    def __init__(self, in_ch, out_ch, groups=32, k=3, s=1, p=1):
        super().__init__()
        self.gn   = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, padding_mode='reflect')
    def forward(self, x, gamma=None, beta=None):
        h = self.gn(x)
        # Avoid huge temporaries: h = h + h*gamma ; then + beta
        if gamma is not None:
            # h = h + 1.0 * (h * gamma)  (no "1.0+gamma" tensor)
            h = torch.addcmul(h, h, gamma)
        if beta is not None:
           h.add_(beta)  # inplace add
        
        h = F.silu(h, inplace=True)
        return self.conv(h)

class ModResBlock(nn.Module):
    """Residual block with the same FiLM (gamma/beta) applied on both convs"""
    def __init__(self, ch, mid=None, groups=32):
        super().__init__()
        m = mid or ch
        self.c1 = ModGNConv(ch, m, groups)
        self.c2 = ModGNConv(m, ch, groups)
    def forward(self, x, gamma=None, beta=None):
        h = self.c1(x, gamma, beta)
        h = self.c2(h, gamma, beta)
        return x + h

class FiLMSequential(nn.Sequential):
    def forward(self, x, gamma=None, beta=None):
        for mod in self:
            if isinstance(mod, ModResBlock):
                x = mod(x, gamma, beta)
            else:
                x = mod(x)  # in case you ever add a plain layer
        return x

class Down(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 2, 1, padding_mode='reflect')
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = nn.Conv2d(ch_in, ch_out, 3, 1, 1, padding_mode='reflect')
    def forward(self, x): 
        return self.conv(self.up(x))

class MapPyramid(nn.Module):
    def __init__(self, in_maps:int, ch:int, use_h8: bool):
        super().__init__()
        self.use_h8 = use_h8
        act = nn.Tanh()
        self.proj_H  = nn.Sequential(nn.Conv2d(in_maps, ch,   3, 1, 1, padding_mode='reflect'), act)
        self.proj_H2 = nn.Sequential(nn.Conv2d(in_maps, ch,   3, 1, 1, padding_mode='reflect'), act)
        self.proj_H4 = nn.Sequential(nn.Conv2d(in_maps, ch*2, 3, 1, 1, padding_mode='reflect'), act)
        if self.use_h8:
            self.proj_H8 = nn.Sequential(nn.Conv2d(in_maps, ch*4, 3, 1, 1, padding_mode='reflect'), act)

    def forward(self, m):
        m_H  = self.proj_H(m)
        m_H2 = self.proj_H2(F.avg_pool2d(m, 2, 2))
        m_H4 = self.proj_H4(F.avg_pool2d(m, 4, 4))
        if self.use_h8:
            m_H8 = self.proj_H8(F.avg_pool2d(m, 8, 8))
            return m_H, m_H2, m_H4, m_H8
        return m_H, m_H2, m_H4, None

class SpatialVAE(nn.Module):
    def __init__(
        self,
        S: int           = 2,
        meta_dim: int    = 6,
        C:int            = CHANNELS,
        z_channels:int   = LATENTS,
        ch:int           = NEURONS,
        down_factor:int  = DOWN_FACTOR,
        mode:str         = MODE,
        logvar_bounds    = (-30.0, 20.0),
        use_checkpoint: bool = True,
        use_film: bool   = True,
    ):
        super().__init__()
        assert down_factor in (4, 8)
        self.C = C
        self.zc = z_channels
        self.mode = mode.lower()
        self.logvar_min, self.logvar_max = logvar_bounds
        self.sample_tau = 0.0
        self.use_checkpoint = use_checkpoint
        self.use_film = use_film

        self.S = S
        self.M = meta_dim

        
        # ----- unconditional encoder (spatial) -----
        self.enc_stem = nn.Conv2d(C, ch, 3, 1, 1, padding_mode='reflect')
        self.e_b1 = FiLMSequential(ModResBlock(ch),   ModResBlock(ch))     # H
        self.d1   = Down(ch, ch*2)                                          # H -> H/2
        self.e_b2 = FiLMSequential(ModResBlock(ch*2), ModResBlock(ch*2))    # H/2
        self.d2   = Down(ch*2, ch*4)                                        # H/2 -> H/4
        self.e_b3 = FiLMSequential(ModResBlock(ch*4), ModResBlock(ch*4))    # H/4
        
        # third down for factor=8
        if down_factor == 8:
            self.d3   = Down(ch*4, ch*4)                                    # H/4 -> H/8
            self.e_b4 = FiLMSequential(ModResBlock(ch*4), ModResBlock(ch*4))# H/8
        else:
            self.d3   = None
            self.e_b4 = nn.Identity()
        
        enc_out_ch = ch*4  # deepest channel width we use for z in both cases

        if self.mode == "kl":
            self.to_mu     = nn.Conv2d(enc_out_ch, z_channels, 3, 1, 1, padding_mode='reflect')
            self.to_logvar = nn.Conv2d(enc_out_ch, z_channels, 3, 1, 1, padding_mode='reflect')
        else:
            self.to_z = nn.Conv2d(enc_out_ch, z_channels, 3, 1, 1)


        # ----- Encoder conditioning heads -----
        use_h8 = (down_factor == 8)
        self.maps_pyr = MapPyramid(S, ch=ch, use_h8=use_h8)

        # Map → spatial (gamma, beta) for each encoder scale
        self.map_film_1 = nn.Sequential(
            nn.Conv2d(ch,   ch,   3, 1, 1, groups=ch, padding_mode='reflect'),
            nn.SiLU(),
            nn.Conv2d(ch,   2*ch, 1)
        )
        self.map_film_2 = nn.Sequential(
            nn.Conv2d(ch,   ch*2, 3, 1, 1, groups=ch, padding_mode='reflect'),
            nn.SiLU(),
            nn.Conv2d(ch*2, 2*ch*2, 1)
        )
        self.map_film_4 = nn.Sequential(
            nn.Conv2d(ch*2, ch*4, 3, 1, 1, groups=ch*2, padding_mode='reflect'),
            nn.SiLU(),
            nn.Conv2d(ch*4, 2*ch*4, 1)
        )
        if down_factor == 8:
            self.map_film_8 = nn.Sequential(
                nn.Conv2d(ch*4, ch*4, 3, 1, 1, groups=ch*4, padding_mode='reflect'),
                nn.SiLU(),
                nn.Conv2d(ch*4, 2*ch*4, 1)
            )
        
        # Encoder FiLM from meta
        self.enc_meta_1 = nn.Sequential(nn.Linear(meta_dim, 2*ch),    nn.SiLU())
        self.enc_meta_2 = nn.Sequential(nn.Linear(meta_dim, 2*ch*2),  nn.SiLU())
        self.enc_meta_4 = nn.Sequential(nn.Linear(meta_dim, 2*ch*4),  nn.SiLU())
        if down_factor == 8:
            self.enc_meta_8 = nn.Sequential(nn.Linear(meta_dim, 2*ch*4), nn.SiLU())

        # ----- decoder -----
        self.dec_in = nn.Conv2d(z_channels, enc_out_ch, 3, 1, 1, padding_mode='reflect')
        if down_factor == 8:
            self.d_b8 = FiLMSequential(ModResBlock(enc_out_ch), ModResBlock(enc_out_ch))
            self.up4  = Up(enc_out_ch, ch*2)
            self.d_b4 = FiLMSequential(ModResBlock(ch*2), ModResBlock(ch*2))
            self.up2  = Up(ch*2, ch)
            self.d_b2 = FiLMSequential(ModResBlock(ch), ModResBlock(ch))
            self.up1  = Up(ch, ch)
            self.d_b1 = FiLMSequential(ModResBlock(ch), ModResBlock(ch))
        else:  # factor 4
            self.d_b4 = FiLMSequential(ModResBlock(enc_out_ch), ModResBlock(enc_out_ch))
            self.up2  = Up(enc_out_ch, ch)
            self.d_b2 = FiLMSequential(ModResBlock(ch), ModResBlock(ch))
            self.up1  = Up(ch, ch)
            self.d_b1 = FiLMSequential(ModResBlock(ch), ModResBlock(ch))

        self.dec_out = nn.Conv2d(ch, C, 3, 1, 1, padding_mode='reflect', bias=True)

        # latent scaling buffer for diffusion compatibility
        self.register_buffer("latent_scale", torch.tensor(1.0, dtype=torch.float32))

    def _combine_gb(self, *gb_list, g_lim=0.5, b_lim=0.2):
        # gb can be (B,2C) vectors (already broadcasted) or (B,2C,H,W) maps
        g_sum, b_sum = 0.0, 0.0
        for gb in gb_list:
            g, b = gb.chunk(2, dim=1)
            g_sum = g_sum + torch.tanh(g)
            b_sum = b_sum + torch.tanh(b)
        
        # clamp final scale
        g_sum = g_sum * g_lim
        b_sum = b_sum * b_lim
        
        return g_sum, b_sum

    def _split_gb(self, gb, C, H=None, W=None, *, squash: bool = True, g_lim=1.0, b_lim=1.0):
        gamma, beta = gb.chunk(2, dim=1)
        if squash:
            gamma = torch.tanh(gamma) * g_lim
            beta  = torch.tanh(beta)  * b_lim
        if H is not None and gamma.dim() == 2:
            gamma = gamma.view(gamma.size(0), C, 1, 1).expand(-1, -1, H, W)
            beta  = beta .view(beta .size(0), C, 1, 1).expand(-1, -1, H, W)
        return gamma, beta

    def _enc_block1(self, h, g1, b1):
        """Encoder block 1 - wrapped for checkpointing."""
        return self.e_b1(h, g1, b1)
    
    def _enc_block2(self, h, g2, b2):
        """Encoder block 2 - wrapped for checkpointing."""
        h = self.d1(h)
        return self.e_b2(h, g2, b2)
    
    def _enc_block3(self, h, g4, b4):
        """Encoder block 3 - wrapped for checkpointing."""
        h = self.d2(h)
        return self.e_b3(h, g4, b4)
    
    def _enc_block4(self, h, g8, b8):
        """Encoder block 4 (H/8) - wrapped for checkpointing."""
        h = self.d3(h)
        return self.e_b4(h, g8, b8)

    def encode_stats(self, x, maps_b, meta):
        # Compute FiLM conditioning only if enabled
        if self.use_film:
            m_H, m_H2, m_H4, m_H8 = self.maps_pyr(maps_b)
        else:
            m_H, m_H2, m_H4, m_H8 = None, None, None, None
    
        # H
        h = self.enc_stem(x)
        if self.use_film:
            H1, W1 = h.shape[-2:]
            g1e, b1e = self._split_gb(self.enc_meta_1(meta), h.size(1), H1, W1, squash=True)
            g1m, b1m = self.map_film_1(m_H).chunk(2, dim=1)
            g1, b1   = self._combine_gb(torch.cat([g1e, b1e], 1), torch.cat([g1m, b1m], 1))
        else:
            g1, b1 = None, None
        
        if self.use_checkpoint and self.training:
            h = checkpoint(self._enc_block1, h, g1, b1, use_reentrant=False)
        else:
            h = self._enc_block1(h, g1, b1)
    
        # H/2 - precompute conditioning before checkpoint
        if self.use_film:
            H2, W2 = (h.shape[-2] // 2, h.shape[-1] // 2)
            g2e, b2e = self._split_gb(self.enc_meta_2(meta), h.size(1) * 2, H2, W2, squash=True)
            g2m, b2m = self.map_film_2(m_H2).chunk(2, dim=1)
            g2, b2   = self._combine_gb(torch.cat([g2e, b2e], 1), torch.cat([g2m, b2m], 1))
        else:
            g2, b2 = None, None
        
        if self.use_checkpoint and self.training:
            h = checkpoint(self._enc_block2, h, g2, b2, use_reentrant=False)
        else:
            h = self._enc_block2(h, g2, b2)
    
        # H/4 - precompute conditioning before checkpoint
        if self.use_film:
            H4, W4 = (h.shape[-2] // 2, h.shape[-1] // 2)
            g4e, b4e = self._split_gb(self.enc_meta_4(meta), h.size(1) * 2, H4, W4, squash=True)
            g4m, b4m = self.map_film_4(m_H4).chunk(2, dim=1)
            g4, b4   = self._combine_gb(torch.cat([g4e, b4e], 1), torch.cat([g4m, b4m], 1))
        else:
            g4, b4 = None, None
        
        if self.use_checkpoint and self.training:
            h = checkpoint(self._enc_block3, h, g4, b4, use_reentrant=False)
        else:
            h = self._enc_block3(h, g4, b4)
    
        # H/8 (if used)
        if self.d3 is not None:
            if self.use_film:
                H8, W8 = (h.shape[-2] // 2, h.shape[-1] // 2)
                g8e, b8e = self._split_gb(self.enc_meta_8(meta), h.size(1), H8, W8, squash=True)
                g8m, b8m = self.map_film_8(m_H8).chunk(2, dim=1)
                g8, b8   = self._combine_gb(torch.cat([g8e, b8e], 1), torch.cat([g8m, b8m], 1))
            else:
                g8, b8 = None, None
            
            if self.use_checkpoint and self.training:
                h = checkpoint(self._enc_block4, h, g8, b8, use_reentrant=False)
            else:
                h = self._enc_block4(h, g8, b8)
                    
        # heads
        if self.mode == "kl":
            mu     = self.to_mu(h)
            logvar = self.to_logvar(h).clamp_(self.logvar_min, self.logvar_max)
            return mu, logvar
        else:
            z = self.to_z(h)
            return z, None

    def reparam(self, mu, logvar):
        if (self.sample_tau <= 0) or (not self.training):
            return mu
        std = torch.exp(0.5 * logvar).clamp_max(1.0)
        return mu + self.sample_tau * torch.randn_like(std) * std
    
    def _dec_block8(self, h):
        """Decoder block H/8 - wrapped for checkpointing."""
        return self.d_b8(h)
    
    def _dec_block4(self, h):
        """Decoder block H/4 - wrapped for checkpointing."""
        return self.d_b4(h)
    
    def _dec_block2(self, h):
        """Decoder block H/2 - wrapped for checkpointing."""
        return self.d_b2(h)
    
    def _dec_block1(self, h):
        """Decoder block H - wrapped for checkpointing."""
        return self.d_b1(h)
        
    def decode(self, z):
        # project z up
        h = self.dec_in(z)
    
        if hasattr(self, 'd_b8'):  # down_factor == 8
            # H/8
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block8, h, use_reentrant=False)
            else:
                h = self._dec_block8(h)
    
            # H/4
            h = self.up4(h)
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block4, h, use_reentrant=False)
            else:
                h = self._dec_block4(h)
    
            # H/2
            h = self.up2(h)
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block2, h, use_reentrant=False)
            else:
                h = self._dec_block2(h)
    
            # H
            h = self.up1(h)
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block1, h, use_reentrant=False)
            else:
                h = self._dec_block1(h)
    
        else:  # down_factor == 4
            # H/4
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block4, h, use_reentrant=False)
            else:
                h = self._dec_block4(h)
            
            # H/2
            h = self.up2(h)
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block2, h, use_reentrant=False)
            else:
                h = self._dec_block2(h)
            
            # H
            h = self.up1(h)
            if self.use_checkpoint and self.training:
                h = checkpoint(self._dec_block1, h, use_reentrant=False)
            else:
                h = self._dec_block1(h)
            
        out_lin = self.dec_out(h)
        out = torch.sigmoid(out_lin)
        return out


    def forward(self, x, maps_b, meta):
        B, C, H, W = x.shape
        mu, logvar = self.encode_stats(x, maps_b, meta)
        z = mu if self.mode != "kl" else self.reparam(mu, logvar)
        z_scaled = z * self.latent_scale
        recon = self.decode(z_scaled)
        if self.mode != "kl":
            logvar = torch.zeros_like(mu)  # keeps your loss happy (KL=0 if beta_kl=0)
        return recon, mu, logvar
        
    @torch.no_grad()
    def reconstruct(self, x, maps_b, meta, deterministic=True):
        B, C, H, W = x.shape
        mu, logvar = self.encode_stats(x, maps_b, meta)
        if self.mode == "kl" and not deterministic and self.sample_tau > 0:
            z = self.reparam(mu, logvar)
            # log.info(f"JUST TESTING {z.shape}")
        else:
            z = mu  # deterministic path (or AE)
        z_scaled = z * self.latent_scale
        return self.decode(z_scaled)
    
    @torch.no_grad()
    def encode_latents(self, x, maps_b, meta, deterministic=True):
        mu, logvar = self.encode_stats(x, maps_b, meta)
        if self.mode == "kl" and not deterministic and self.sample_tau > 0:
            z = self.reparam(mu, logvar)
        else:
            z = mu
        return z * self.latent_scale

    @torch.no_grad()
    def decode_latents(self, z_scaled):
        return self.decode(z_scaled)

    @torch.no_grad()
    def calibrate_latent_scale(self, loader, device, S_torch, max_batches=50):
        self.eval()
        cnt, acc = 0, 0.0
        for i, (x, m) in enumerate(loader):
            x = x.to(device, non_blocking=True).float()
            m = m.to(device, non_blocking=True).float()
            maps = S_torch.unsqueeze(0).expand(x.size(0), -1, -1, -1)
            mu, _ = self.encode_stats(x, maps, m)
            zs = mu.float().detach().flatten(2).std(dim=2).mean().item()
            acc += zs; cnt += 1
            if i + 1 >= max_batches: break
        mean_std = max(acc / max(1, cnt), 1e-6)
        self.latent_scale.data = torch.tensor(1.0 / mean_std, device=self.latent_scale.device, dtype=self.latent_scale.dtype)
