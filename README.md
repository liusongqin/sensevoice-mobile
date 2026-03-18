# SenseVoice Mobile

Android app for on-device speech recognition using the [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) model with ONNX Runtime.

## Features

- Real-time speech recognition from microphone
- Audio file transcription (WAV format)
- Language, emotion, and event detection
- Fully on-device inference (no network required)

## Quick Start

### 1. Download Model Files

You can directly use the official ONNX model files from ModelScope — **no model conversion needed**.

Download the following files from [SenseVoiceSmall-onnx](https://www.modelscope.cn/models/iic/SenseVoiceSmall-onnx/files):

- `model.onnx` (or `model_quant.onnx` for a smaller quantized version)
- `tokens.txt`
- `am.mvn`

### 2. Place Model Files

Copy the downloaded files into the Android assets directory:

```
android/app/src/main/assets/
├── model.onnx
├── tokens.txt
└── am.mvn
```

> **Note:** If using `model_quant.onnx`, rename it to `model.onnx` before placing it in the assets directory.

### 3. Build and Run

Open the `android/` directory in Android Studio, then build and run the app.

## Model Export (Optional)

If you prefer to export the model yourself from a PyTorch checkpoint, you can use the provided export script:

```bash
pip install -r scripts/requirements.txt
python scripts/export_onnx.py --model_dir /path/to/SenseVoiceSmall --output_dir ./output
```

For a smaller model with dynamic quantization:

```bash
python scripts/export_onnx.py --model_dir /path/to/SenseVoiceSmall --output_dir ./output --quantize
```

Then copy the generated files from `./output/` to `android/app/src/main/assets/`.

## Requirements

- Android 8.0 (API 26) or higher
- Android Studio
