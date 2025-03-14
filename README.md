# Speech2Txt

## 项目简介
Speech2Txt是一个基于OpenAI Whisper模型的语音转文本工具，支持将各种音频格式转换为带时间戳的文本。该工具提供了命令行界面和图形用户界面两种使用方式，方便不同场景下的使用需求。

## 功能特点
- 支持多种音频格式（mp3, wav, m4a, flac, ogg等）
- 生成带时间戳的转录文本
- 支持单文件转换和批量处理
- 提供多种Whisper模型大小选择（tiny, base, small, medium, large）
- 图形用户界面，操作简单直观
- 命令行界面，适合脚本集成
- 支持GPU加速（如果可用）

## 技术栈
- 语言：Python 3.8+
- 核心模型：OpenAI Whisper
- 图形界面：PySide6 (Qt for Python)
- 音频处理：ffmpeg (Whisper依赖)
- 深度学习框架：PyTorch

## 安装指南

### 前提条件
- Python 3.8+
- ffmpeg（必须安装并添加到系统PATH）
- CUDA支持（可选，用于GPU加速）

### 安装步骤

```bash
# 克隆仓库
git clone https://github.com/prog-le/speech2txt.git
cd speech2txt

# 创建虚拟环境（推荐conda）
conda create -n speech2txt python=3.10
conda activate speech2txt

# 安装依赖
pip install -r requirements.txt

# 或者直接安装包
pip install .
```

## 使用说明

### 图形界面
```bash
python speech_to_text_ui.py
```
- 1. 选择音频文件
- 2. 选择输出文件
- 3. 选择模型大小
- 4. 点击开始转换
- 5. 转换完成后，会在指定位置生成转换后的文本文件

### 命令行
```bash
python speech_to_text_whisper.py
```
- 1. 输入音频文件路径
- 2. 输入输出文件路径
- 3. 选择模型大小（tiny, base, small, medium, large）
- 4. 转换完成后，会在指定位置生成转换后的文本文件

## 项目结构
```
speech2txt/
├── requirements.txt               # 项目依赖
├── .gitignore                     # Git忽略文件
├── LICENSE                        # 许可证
├── CONTRIBUTING.md                # 贡献指南
└── README.md                      # 项目说明
```

## 模型说明
Whisper模型有不同大小，根据您的需求和硬件选择合适的模型：

| 模型大小 | 参数数量 | 内存需求 | 相对速度 | 精度 |
|---------|---------|---------|---------|------|
| tiny    | 39M     | 低      | 最快    | 最低 |
| base    | 74M     | 中低    | 快      | 中低 |
| small   | 244M    | 中      | 中      | 中   |
| medium  | 769M    | 中高    | 慢      | 高   |
| large   | 1550M   | 高      | 最慢    | 最高 |

## 许可证
该项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 致谢
- [OpenAI Whisper](https://github.com/openai/whisper) - 提供强大的语音识别模型
- [PySide6](https://wiki.qt.io/Qt_for_Python) - 提供Python的Qt绑定
- [PyTorch](https://pytorch.org/) - 提供深度学习框架支持 