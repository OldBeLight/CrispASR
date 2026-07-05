#!/usr/bin/env python3
"""
Convert Semantic-DACVAE (Aratako/Semantic-DACVAE-Japanese-32dim) → GGUF
for the CrispASR irodori-tts backend's audio decoder.

The DACVAE decoder has the same architecture as descript/dac_44khz:
  out_proj: Conv1d(codebook_dim → 1024, k=1)   ← VAE bottleneck projection
  in_conv:  Conv1d(1024 → 1536, k=7)
  4 DecoderBlocks: strides [8, 8, 4, 2]
  out_snake + out_conv + Tanh → PCM

Usage:
    python models/convert-dacvae-to-gguf.py \\
        --model Aratako/Semantic-DACVAE-Japanese-32dim \\
        --output /mnt/storage/gguf-models/dacvae-ja-32dim-f16.gguf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from gguf import GGUFWriter
except ImportError:
    sys.exit("pip install gguf")

try:
    import torch
except ImportError:
    sys.exit("pip install torch")


def to_f16(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float16).numpy()


def to_f32(t: torch.Tensor) -> np.ndarray:
    return t.detach().to(torch.float32).numpy()


def write_tensor(writer: GGUFWriter, name: str, data: np.ndarray):
    if data.size <= 512 or data.ndim == 1:
        writer.add_tensor(name, data.astype(np.float32))
    else:
        writer.add_tensor(name, data.astype(np.float16))


def main():
    parser = argparse.ArgumentParser(description="Convert DACVAE to GGUF")
    parser.add_argument("--model", type=str, required=True,
                        help="HF repo ID or local path to DACVAE weights.pth")
    parser.add_argument("--output", type=str, required=True, help="Output GGUF path")
    args = parser.parse_args()

    # Load DACVAE model
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from dacvae import DACVAE
    except ImportError:
        sys.exit("pip install dacvae (or git+https://github.com/facebookresearch/dacvae.git)")

    print(f"Loading DACVAE: {args.model}")
    model = DACVAE.load(args.model).eval()
    state = model.state_dict()

    # Extract config
    sample_rate = int(model.sample_rate)
    hop_length = int(model.hop_length)
    latent_dim = int(model.latent_dim)
    decoder_dim = int(model.decoder_dim)
    decoder_rates = list(model.decoder_rates)
    codebook_dim = int(model.codebook_dim)

    print(f"  sample_rate={sample_rate}, hop_length={hop_length}")
    print(f"  latent_dim={latent_dim}, codebook_dim={codebook_dim}")
    print(f"  decoder_dim={decoder_dim}, decoder_rates={decoder_rates}")

    # Infer decoder channels: decoder_dim // 2^i for each block
    n_blocks = len(decoder_rates)
    decoder_channels = [decoder_dim // (2 ** i) for i in range(n_blocks + 1)]
    print(f"  decoder_channels={decoder_channels}")

    # Create GGUF
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output_path), "dacvae")

    # Write config
    writer.add_uint32("dacvae.sample_rate", sample_rate)
    writer.add_uint32("dacvae.hop_length", hop_length)
    writer.add_uint32("dacvae.latent_dim", latent_dim)
    writer.add_uint32("dacvae.codebook_dim", codebook_dim)
    writer.add_uint32("dacvae.decoder_dim", decoder_dim)
    writer.add_uint32("dacvae.n_decoder_blocks", n_blocks)

    # ── VAE bottleneck out_proj: NormConv1d(codebook_dim → latent_dim, k=1) ──
    # NormConv1d stores weight_v, weight_g (weight norm decomposition)
    # After weight_norm removal: weight = g * v / ||v||
    # We need to reconstruct the effective weight
    def get_conv_weight(prefix):
        """Get effective weight from weight-normed conv."""
        if f"{prefix}.weight" in state:
            return state[f"{prefix}.weight"], state.get(f"{prefix}.bias")
        # Weight norm decomposition
        v = state[f"{prefix}.weight_v"]
        g = state[f"{prefix}.weight_g"]
        # weight = g * v / ||v||
        norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
        norm = norm.unsqueeze(-1) if v.ndim == 3 else norm
        weight = g * v / (norm + 1e-12)
        bias = state.get(f"{prefix}.bias")
        return weight, bias

    # out_proj
    w, b = get_conv_weight("quantizer.out_proj.conv")
    print(f"  out_proj: weight {list(w.shape)}, bias {list(b.shape) if b is not None else None}")
    write_tensor(writer, "dacvae.out_proj.w", to_f16(w))
    if b is not None:
        write_tensor(writer, "dacvae.out_proj.b", to_f32(b))

    # ── Decoder ──
    # in_conv: Conv1d(latent_dim → decoder_dim, k=7)
    w, b = get_conv_weight("decoder.model.0.conv")
    print(f"  in_conv: weight {list(w.shape)}")
    write_tensor(writer, "dacvae.dec.in_conv.w", to_f16(w))
    if b is not None:
        write_tensor(writer, "dacvae.dec.in_conv.b", to_f32(b))

    # DecoderBlocks
    for blk_idx in range(n_blocks):
        blk_prefix = f"decoder.model.{blk_idx + 1}.block"

        # The DecoderBlock structure in DACVAE is more complex than standard DAC:
        # block[0] = Snake1d (input_dim)
        # block[1] = NormConvTranspose1d (input_dim → output_dim, k=2*stride, stride=stride)
        # block[2] = ELU
        # block[3] = NormConvTranspose1d (input_dim/3 → output_dim/3, k=2*wm_stride, stride=wm_stride)
        # block[4] = ResidualUnit(output_dim, d=1, Snake)
        # block[5] = ResidualUnit(output_dim, d=3, Snake)
        # block[6] = ResidualUnit(output_dim/3, k=3, ELU, causal, true_skip)
        # block[7] = ResidualUnit(output_dim/3, k=3, ELU, causal, true_skip)
        # block[8] = ResidualUnit(output_dim, d=9, Snake)
        # block[9] = ResidualUnit or Identity (last_kernel_size)
        # block[10] = ELU
        # block[11] = NormConv1d (downsample)

        # For now, write all decoder block weights with their original structure
        # The C++ code will need to handle the DACVAE-specific block layout
        prefix = f"decoder.model.{blk_idx + 1}"
        block_tensors = {k: v for k, v in state.items() if k.startswith(prefix)}
        for tname, tval in sorted(block_tensors.items()):
            short = tname.replace(prefix + ".", "")
            gguf_name = f"dacvae.dec.blk.{blk_idx}.{short}"
            # Reconstruct weight-normed weights
            if "weight_v" in tname:
                base = tname.replace("weight_v", "")
                g_key = base + "weight_g"
                if g_key in state:
                    v = tval
                    g = state[g_key]
                    norm = torch.linalg.norm(v.reshape(v.shape[0], -1), dim=1, keepdim=True)
                    if v.ndim == 3:
                        norm = norm.unsqueeze(-1)
                    eff_w = g * v / (norm + 1e-12)
                    gguf_name = gguf_name.replace(".weight_v", ".weight")
                    write_tensor(writer, gguf_name, to_f16(eff_w))
                    continue
            if "weight_g" in tname:
                continue  # handled with weight_v above
            if "alpha" in short:
                write_tensor(writer, gguf_name, to_f32(tval))
            elif tval.ndim <= 1 or tval.numel() <= 512:
                write_tensor(writer, gguf_name, to_f32(tval))
            else:
                write_tensor(writer, gguf_name, to_f16(tval))

    # Final layers: wm_model.encoder_block (used for watermark-free output path)
    # When alpha=0 (Irodori disables watermark), decode just runs:
    #   for layer in self.model: x = layer(x)
    #   return self.wm_model.encoder_block.forward_no_conv(x)
    # forward_no_conv runs through pre[:-1] (Snake activation only, no conv)
    # So we just need the Snake alpha from the encoder_block
    wm_prefix = "decoder.wm_model.encoder_block.pre"
    for tname, tval in state.items():
        if tname.startswith(wm_prefix) and "alpha" in tname:
            short = tname.replace("decoder.", "")
            write_tensor(writer, f"dacvae.{short}", to_f32(tval))

    # Finalize
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = output_path.stat().st_size
    print(f"\nWrote {output_path} ({file_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
