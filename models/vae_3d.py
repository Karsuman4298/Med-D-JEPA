import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from monai.networks.nets import AutoencoderKL, PatchDiscriminator
    from monai.losses import PatchAdversarialLoss, PerceptualLoss
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    AutoencoderKL = None
    PatchDiscriminator = None
    PatchAdversarialLoss = None
    PerceptualLoss = None


def gradient_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    3D spatial gradient loss (L_grad) from Med-D-JEPA Section 3.2.
    Penalizes blurred anatomical boundaries across all 3 spatial planes (Z, Y, X).
    """
    dz_pred = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
    dz_targ = torch.abs(target[:, :, 1:, :, :] - target[:, :, :-1, :, :])

    dy_pred = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
    dy_targ = torch.abs(target[:, :, :, 1:, :] - target[:, :, :, :-1, :])

    dx_pred = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])
    dx_targ = torch.abs(target[:, :, :, :, 1:] - target[:, :, :, :, :-1])

    return F.l1_loss(dz_pred, dz_targ) + F.l1_loss(dy_pred, dy_targ) + F.l1_loss(dx_pred, dx_targ)


def patchify_3d(z: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    """
    Partitions continuous 3D latent volume (B, C, H, W, D) into non-overlapping
    3D patches of size (p, p, p) and flattens into sequence (B, N, C * p^3).
    Example: (B, 4, 16, 16, 16) with p=2 -> (B, 512, 32)
    """
    B, C, H, W, D = z.shape
    p = patch_size
    gh, gw, gd = H // p, W // p, D // p

    x = z.view(B, C, gh, p, gw, p, gd, p)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
    return x.view(B, gh * gw * gd, C * (p ** 3))


def unpatchify_3d(tokens: torch.Tensor, grid_shape: tuple = (8, 8, 8), patch_size: int = 2, latent_channels: int = 4) -> torch.Tensor:
    """
    Inverse operation of patchify_3d:
    Rearranges token sequence (B, N, C * p^3) back into (B, C, H, W, D).
    Example: (B, 512, 32) with grid (8, 8, 8), p=2, C=4 -> (B, 4, 16, 16, 16)
    """
    B = tokens.shape[0]
    gh, gw, gd = grid_shape
    p = patch_size
    C = latent_channels

    x = tokens.view(B, gh, gw, gd, C, p, p, p)
    x = x.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
    return x.view(B, C, gh * p, gw * p, gd * p)


class Stage1MedVAEGAN(nn.Module):
    """
    Stage I 3D KL-VAE-GAN compression model from Med-D-JEPA.
    Combines 3D AutoencoderKL with 3D PatchDiscriminator and exact 5-loss optimization.
    """
    def __init__(self, in_channels: int = 1, latent_channels: int = 4):
        super().__init__()
        if not MONAI_AVAILABLE:
            raise ImportError(
                "MONAI is required for Stage1MedVAEGAN. "
                "Please install with: pip install monai[all]"
            )

        # 8x spatial downsampling (e.g. 128^3 -> 16^3)
        self.vae = AutoencoderKL(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=in_channels,
            num_channels=(32, 64, 128, 256),
            latent_channels=latent_channels,
            num_res_blocks=2,
            norm_num_groups=16,
            attention_levels=(False, False, False, True),
        )

        self.discriminator = PatchDiscriminator(
            spatial_dims=3,
            num_layers_d=3,
            num_channels=32,
            in_channels=in_channels,
        )

        # 2.5D slice-wise perceptual loss (AlexNet)
        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3,
            network_type="alex",
            is_fake_3d=True
        )
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")

    def encode(self, x: torch.Tensor):
        """Encodes 3D MRI volume into latent representation."""
        z_mu, z_sigma = self.vae.encode(x)
        return z_mu, z_sigma

    def decode(self, z: torch.Tensor):
        """Decodes 3D continuous latents into synthetic 3D MRI volume."""
        return self.vae.decode(z)

    def compute_kl(self, z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
        """Kullback-Leibler divergence with standard normal prior."""
        # MONAI AutoencoderKL returns z_sigma as standard deviation
        kl = 0.5 * torch.sum(
            z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2) + 1e-8) - 1,
            dim=[1, 2, 3, 4]
        )
        return torch.mean(kl)

    def train_step(self, x: torch.Tensor, opt_g: torch.optim.Optimizer, opt_d: torch.optim.Optimizer, global_step: int):
        """
        Executes one training step with exact paper weights:
        lambda_1 = 2.0 (L_recon)
        lambda_2 = 0.5 (L_lpips)
        lambda_3 = 1.0 (L_grad)
        lambda_4 = 1e-7 (L_reg, warmed up after step 1000)
        lambda_5 = 1.0 (L_adv)
        """
        # --- 1. Generator (VAE) Update ---
        opt_g.zero_grad()
        recon, z_mu, z_sigma = self.vae(x)

        l_recon = F.l1_loss(recon, x)
        l_lpips = self.perceptual_loss(recon, x)
        l_grad = gradient_loss_3d(recon, x)
        l_kl = self.compute_kl(z_mu, z_sigma)

        # KL warmup schedule (Appendix A): starts at step 1000, linear over 1000 steps
        kl_scale = min(1.0, max(0.0, (global_step - 1000) / 1000.0))
        lambda_kl = 1e-7 * kl_scale

        disc_out_fake = self.discriminator(recon)
        logits_fake = disc_out_fake[-1] if isinstance(disc_out_fake, list) else disc_out_fake
        l_adv = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)

        loss_g = (2.0 * l_recon) + (0.5 * l_lpips) + (1.0 * l_grad) + (lambda_kl * l_kl) + (1.0 * l_adv)
        loss_g.backward()
        opt_g.step()

        # --- 2. Discriminator Update ---
        opt_d.zero_grad()
        disc_out_real = self.discriminator(x)
        logits_real = disc_out_real[-1] if isinstance(disc_out_real, list) else disc_out_real

        disc_out_fake_det = self.discriminator(recon.detach())
        logits_fake_det = disc_out_fake_det[-1] if isinstance(disc_out_fake_det, list) else disc_out_fake_det

        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        loss_d_fake = self.adv_loss(logits_fake_det, target_is_real=False, for_discriminator=True)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        loss_d.backward()
        opt_d.step()

        return {
            "loss_g": loss_g.item(),
            "loss_d": loss_d.item(),
            "l_recon": l_recon.item(),
            "l_lpips": l_lpips.item(),
            "l_grad": l_grad.item(),
            "l_kl": l_kl.item(),
            "lambda_kl": lambda_kl,
        }

