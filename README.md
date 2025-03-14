# 语音转文本工具

一个功能强大的语音转文本工具，支持单文件和批量转录，集成了多种语音识别模型和AI摘要功能。

![应用截图](docs/images/screenshot.png)

## 功能特点

- **多模型支持**：集成了Whisper和FunASR两种主流语音识别模型
- **单文件转录**：支持单个音频文件的转录，并提供实时进度显示
- **批量转录**：支持批量处理多个音频文件，自动保存转录结果
- **AI摘要**：集成Ollama和自定义API接口，可对转录文本生成摘要
- **多种格式**：支持MP3、WAV、M4A、FLAC、OGG等多种音频格式
- **友好界面**：简洁直观的图形用户界面，易于操作
- **命令行支持**：提供命令行接口，方便脚本集成和自动化处理
- **API服务**：内置REST API服务，可作为独立的语音识别服务器

## 安装方法

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/yourusername/speech-to-text-tool.git
cd speech-to-text-tool
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 运行应用

```bash
python speech_to_text_ui.py
```

## 使用说明

### 图形界面使用

#### 单文件转录

1. 切换到"单文件转录"标签页
2. 点击"浏览..."选择音频文件
3. 设置输出文件路径或使用默认路径
4. 点击"开始转录"按钮
5. 等待转录完成，结果将显示在文本框中并保存到指定文件

#### 批量转录

1. 切换到"批量转录"标签页
2. 点击"浏览..."选择包含音频文件的目录
3. 设置输出目录或使用默认目录
4. 点击"开始转录"按钮
5. 等待所有文件转录完成，结果将保存到指定目录

#### AI摘要生成

1. 切换到"AI摘要"标签页
2. 选择AI模型类型（Ollama或自定义API）
3. 配置模型参数
4. 在文本框中输入或加载要摘要的文本
5. 设置摘要长度和语言
6. 点击"生成摘要"按钮
7. 查看生成的摘要，可选择保存到文件

### 命令行使用

单文件转录：

```bash
python speech_to_text_cli.py -i input.mp3 -o output.txt -m whisper --whisper-size base
```

批量转录：

```bash
python speech_to_text_cli.py -i input_dir -o output_dir -m whisper --whisper-size base --batch
```

使用FunASR模型：

```bash
python speech_to_text_cli.py -i input.mp3 -o output.txt -m funasr --funasr-model paraformer-zh
```

### API服务使用

启动API服务：

```bash
python speech_to_text_api.py --host 0.0.0.0 --port 5000
```

转录API调用示例：

```bash
curl -X POST -F "file=@audio.mp3" -F "model_type=whisper" -F "model_size=base" http://localhost:5000/api/transcribe
```

## 模型配置

### Whisper模型

- **模型大小**：tiny, base, small, medium, large
- **语言**：自动检测，默认中文

### FunASR模型

- **模型名称**：paraformer-zh, paraformer-zh-streaming
- **模型版本**：v2.0.4
- **功能选项**：VAD, 标点, 说话人识别

### AI摘要模型

- **Ollama**：支持本地部署的开源大语言模型
- **自定义API**：支持OpenAI兼容的API接口

## 常见问题

**Q: 如何提高转录准确率？**
A: 对于Whisper模型，选择更大的模型（如medium或large）通常能提高准确率。对于中文内容，FunASR的paraformer-zh模型通常有更好的表现。

**Q: 转录速度慢怎么办？**
A: 如果有CUDA支持的GPU，程序会自动使用GPU加速。对于长音频，可以尝试使用FunASR的streaming模型，或者Whisper的tiny/base模型以提高速度。

**Q: 支持哪些音频格式？**
A: 支持大多数常见音频格式，包括MP3、WAV、M4A、FLAC、OGG等。如果遇到不支持的格式，可以使用ffmpeg先转换为WAV格式。

## 贡献指南

欢迎贡献代码、报告问题或提出新功能建议！请遵循以下步骤：

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个Pull Request

## 许可证

本项目采用MIT许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [OpenAI Whisper](https://github.com/openai/whisper) - 提供强大的语音识别模型
- [FunASR](https://github.com/alibaba-damo-academy/FunASR) - 提供高性能中文语音识别模型
- [Ollama](https://github.com/ollama/ollama) - 提供本地大语言模型部署方案
- [PySide6](https://doc.qt.io/qtforpython-6/) - 提供GUI框架