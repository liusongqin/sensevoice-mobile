# SenseVoice Mobile

在 Android 手机上运行 [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) 语音识别模型的移动端应用。

## 功能

- 🎙️ **实时录音识别** — 通过麦克风录音并实时转文字
- 📂 **音频文件识别** — 选择本地 WAV 音频文件进行识别
- 🌍 **多语言支持** — 自动检测中文、英文、日文、韩文等语言
- 😊 **情感识别** — 识别说话者的情感（中性、开心、悲伤等）
- 🔊 **事件检测** — 检测语音事件（语音、音乐、噪声等）
- 📱 **纯端侧推理** — 使用 ONNX Runtime 在设备本地运行，无需网络

## 项目结构

```
sensevoice-mobile/
├── android/                          # Android 应用项目
│   ├── app/
│   │   ├── build.gradle
│   │   └── src/main/
│   │       ├── AndroidManifest.xml
│   │       ├── assets/               # 放置模型文件（需自行导出）
│   │       │   ├── model.onnx        # ONNX 模型（需导出）
│   │       │   ├── tokens.txt        # 词表文件（需复制）
│   │       │   └── am.mvn            # CMVN 统计文件（需复制）
│   │       ├── java/com/sensevoice/mobile/
│   │       │   ├── MainActivity.kt           # 主界面
│   │       │   ├── SenseVoiceModel.kt        # 模型推理封装
│   │       │   ├── AudioFeatureExtractor.kt  # 音频特征提取（FBank+LFR+CMVN）
│   │       │   ├── AudioRecorder.kt          # 麦克风录音
│   │       │   ├── AudioFileReader.kt        # WAV 文件读取
│   │       │   └── TokenDecoder.kt           # CTC 解码 + 文本后处理
│   │       └── res/                          # 资源文件
│   ├── build.gradle
│   └── settings.gradle
├── scripts/
│   ├── export_onnx.py               # 模型导出脚本
│   └── requirements.txt
└── README.md
```

## 快速开始

### 1. 导出模型

首先需要将 SenseVoice PyTorch 模型导出为 ONNX 格式：

```bash
# 安装依赖
cd scripts
pip install -r requirements.txt

# 导出模型（需指定 SenseVoice 模型目录）
python export_onnx.py --model_dir /path/to/SenseVoiceSmall --output_dir ./output

# 可选：量化模型以减小体积
python export_onnx.py --model_dir /path/to/SenseVoiceSmall --output_dir ./output --quantize
```

### 2. 放置模型文件

将导出的文件复制到 Android 项目的 assets 目录：

```bash
cp output/model.onnx android/app/src/main/assets/
cp output/tokens.txt android/app/src/main/assets/
cp output/am.mvn android/app/src/main/assets/
```

### 3. 构建 Android 应用

使用 Android Studio 或命令行构建：

```bash
cd android

# 使用 Gradle 构建（需要 Android SDK）
./gradlew assembleDebug

# APK 输出位置
# android/app/build/outputs/apk/debug/app-debug.apk
```

或者使用 Android Studio：
1. 打开 `android/` 目录作为项目
2. 等待 Gradle 同步完成
3. 点击 Run 按钮安装到设备

### 4. 使用应用

1. 安装 APK 到手机
2. 授予麦克风权限
3. 点击 **"开始录音"** 录制语音，再点击 **"停止录音"** 进行识别
4. 或点击 **"选择音频文件"** 选择本地 WAV 文件进行识别
5. 识别结果将显示语言、情感、事件和文本内容

## 技术架构

### 音频处理流程

```
原始音频 (16kHz, 16-bit, mono)
    │
    ▼
预加重 (coefficient: 0.97)
    │
    ▼
分帧 (25ms 窗口, 10ms 帧移)
    │
    ▼
汉宁窗 + FFT
    │
    ▼
Mel 滤波器组 (80 维 FBank 特征)
    │
    ▼
对数能量
    │
    ▼
LFR (合并 7 帧, 每 6 帧取一次) → 560 维特征
    │
    ▼
CMVN 归一化
    │
    ▼
ONNX 模型推理 → CTC logits
    │
    ▼
CTC 贪心解码 → 文本 + 语言/情感/事件标签
```

### 核心组件

| 组件 | 说明 |
|------|------|
| `AudioFeatureExtractor` | 从原始音频提取 FBank 特征，应用 LFR 和 CMVN |
| `SenseVoiceModel` | ONNX Runtime 模型推理封装 |
| `TokenDecoder` | CTC 贪心解码，解析语言/情感/事件标签 |
| `AudioRecorder` | 使用 AudioRecord API 录制 16kHz PCM 音频 |
| `AudioFileReader` | 读取 WAV 文件并重采样到 16kHz |

### 依赖

- **ONNX Runtime Android** 1.17.0 — 模型推理引擎
- **AndroidX** — UI 组件和兼容性
- **Material Design** — 现代 UI 组件

## 对应的 Python 代码

本项目实现了与以下 Python 代码等效的移动端功能：

```python
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

model_dir = "/path/to/SenseVoiceSmall"
file = f"{model_dir}/example/zh.mp3"

model = AutoModel(
    model=model_dir,
    device="cpu",
)

res = model.generate(
    input=file,
)
print(f"{file}: {res}")
```

## 系统要求

- **Android** 8.0 (API 26) 及以上
- **存储空间** ≈ 200MB（含模型文件）
- **RAM** ≥ 2GB（推荐 4GB）

## 许可证

本项目仅供学习和研究使用。SenseVoice 模型的使用请遵循其原始许可证。
