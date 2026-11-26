import inspect
import os
import torch
import torch.nn as nn

VAE = {
    "KL16_MAR": {
        "spatial_dims": 2,
        "vae_embed_dim": 16,
        "ckpt_path": "vae/kl16.ckpt",
        "kwargs": dict(embed_dim=16, ch_mult=(1, 1, 2, 2, 4)),
        "scaling_factor": 0.2325,
        "shift": 0.0,
        "vae_stride": 16
    },
    "MED_VAE_3D": {
        "spatial_dims": 3,
        "vae_embed_dim": 4,
        "ckpt_path": "vae/med_vae3d.pt",
        "kwargs": dict(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(32, 64, 128, 256),
            latent_channels=4,
            num_res_blocks=2,
            norm_num_groups=16,
            attention_levels=(False, False, False, True),
        ),
        "scaling_factor": 1.0,
        "shift": 0.0,
        "vae_stride": 8
    }
}


def get_pretrained_vae(args):
    vae_dict = VAE[args.vae]
    spatial_dims = vae_dict.get("spatial_dims", 2)
    args.spatial_dims = spatial_dims

    if args.use_cached and not getattr(args, "online_eval", False) and not getattr(args, "evaluate", False):
        vae = nn.Identity()
    else:
        ckpt_path = os.path.join(
            args.local_diffusers_model_root, vae_dict["ckpt_path"])

        if spatial_dims == 3:
            from monai.networks.nets import AutoencoderKL
            vae_kwargs = dict(vae_dict["kwargs"])
            vae_sig = inspect.signature(AutoencoderKL.__init__).parameters
            if "channels" not in vae_sig and "num_channels" in vae_sig and "channels" in vae_kwargs:
                vae_kwargs["num_channels"] = vae_kwargs.pop("channels")
            vae = AutoencoderKL(**vae_kwargs)
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                # Handle state_dict wrapped in Stage1MedVAEGAN
                if "vae" in state_dict:
                    state_dict = state_dict["vae"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                vae.load_state_dict(state_dict)
                print(f"[Stage I] Loaded 3D VAE from {ckpt_path}")
            else:
                print(f"[Stage I] Warning: {ckpt_path} not found. Running with initialized weights.")
            vae = vae.cuda().eval()
        else:
            # Used for loading 2D KL16 from MAR
            from models.vae import AutoencoderKL
            vae = AutoencoderKL(ckpt_path=ckpt_path, **vae_dict["kwargs"]).cuda().eval()

    for param in vae.parameters():
        param.requires_grad = False

    args.vae_embed_dim = vae_dict["vae_embed_dim"]
    args.scaling_factor = vae_dict['scaling_factor']
    args.shift = vae_dict["shift"]
    args.vae_stride = vae_dict["vae_stride"]

    return vae
