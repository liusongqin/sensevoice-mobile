#!/usr/bin/env python3
"""
Export SenseVoice model to ONNX format for mobile deployment.

This script converts a SenseVoice PyTorch model to ONNX format that can
be loaded by ONNX Runtime on Android/iOS devices.

Usage:
    python export_onnx.py --model_dir /path/to/SenseVoiceSmall --output_dir ./output

Requirements:
    pip install -r requirements.txt
"""

import argparse
import os
import shutil

import numpy as np
import torch
from funasr import AutoModel


def export_model(model_dir: str, output_dir: str, quantize: bool = False):
    """Export SenseVoice model to ONNX format.

    Args:
        model_dir: Path to the SenseVoice model directory
        output_dir: Directory to save the exported ONNX model and assets
        quantize: Whether to apply dynamic quantization
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    model = AutoModel(model=model_dir, device="cpu")

    inner_model = model.model
    inner_model.eval()

    # Create dummy inputs matching the model's expected format
    # speech_feats: [batch, num_frames, feature_dim]
    # speech_lengths: [batch]
    batch_size = 1
    num_frames = 100
    feature_dim = 560  # 80 mel bins * 7 LFR frames

    dummy_speech = torch.randn(batch_size, num_frames, feature_dim)
    dummy_lengths = torch.tensor([num_frames], dtype=torch.long)

    onnx_path = os.path.join(output_dir, "model.onnx")

    print("Exporting to ONNX format...")
    # Try export with the model's forward signature
    try:
        # SenseVoice model forward: speech, speech_lengths, language, text_norm
        dummy_language = torch.tensor([0], dtype=torch.long)
        dummy_text_norm = torch.tensor([15], dtype=torch.long)

        torch.onnx.export(
            inner_model,
            (dummy_speech, dummy_lengths, dummy_language, dummy_text_norm),
            onnx_path,
            input_names=["speech", "speech_lengths", "language", "text_norm"],
            output_names=["logits"],
            dynamic_axes={
                "speech": {0: "batch", 1: "num_frames"},
                "speech_lengths": {0: "batch"},
                "logits": {0: "batch", 1: "num_frames"},
            },
            opset_version=14,
            do_constant_folding=True,
        )
    except Exception:
        # Fallback: try with just speech and speech_lengths
        torch.onnx.export(
            inner_model,
            (dummy_speech, dummy_lengths),
            onnx_path,
            input_names=["speech", "speech_lengths"],
            output_names=["logits"],
            dynamic_axes={
                "speech": {0: "batch", 1: "num_frames"},
                "speech_lengths": {0: "batch"},
                "logits": {0: "batch", 1: "num_frames"},
            },
            opset_version=14,
            do_constant_folding=True,
        )

    print(f"ONNX model saved to {onnx_path}")

    if quantize:
        print("Applying dynamic quantization...")
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized_path = os.path.join(output_dir, "model_quantized.onnx")
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QUInt8,
        )
        print(f"Quantized model saved to {quantized_path}")

    # Copy tokens.txt
    tokens_src = os.path.join(model_dir, "tokens.txt")
    if os.path.exists(tokens_src):
        shutil.copy2(tokens_src, os.path.join(output_dir, "tokens.txt"))
        print("Copied tokens.txt")

    # Copy or generate CMVN file
    cmvn_src = os.path.join(model_dir, "am.mvn")
    if os.path.exists(cmvn_src):
        shutil.copy2(cmvn_src, os.path.join(output_dir, "am.mvn"))
        print("Copied am.mvn")

    # Verify the exported model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    print("Input names and shapes:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.shape} ({inp.type})")
    print("Output names and shapes:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.shape} ({out.type})")

    print(f"\nExport complete! Files saved to {output_dir}")
    print("\nTo use with the Android app:")
    print(f"  1. Copy {output_dir}/model.onnx to android/app/src/main/assets/")
    print(f"  2. Copy {output_dir}/tokens.txt to android/app/src/main/assets/")
    print(f"  3. Copy {output_dir}/am.mvn to android/app/src/main/assets/")


def main():
    parser = argparse.ArgumentParser(
        description="Export SenseVoice model to ONNX format for mobile deployment"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the SenseVoice model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save exported files (default: ./output)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization for smaller model size",
    )
    args = parser.parse_args()

    export_model(args.model_dir, args.output_dir, args.quantize)


if __name__ == "__main__":
    main()
