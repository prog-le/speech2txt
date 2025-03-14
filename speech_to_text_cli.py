#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别终端交互程序
提供命令行界面的语音转文字功能
"""

import os
import sys
import argparse
import glob
import time
from datetime import datetime

# 检查必要的依赖
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    print("警告: 无法导入Whisper模块")
    print("Whisper功能将不可用。请安装所需依赖:")
    print("pip install whisper")
    WHISPER_AVAILABLE = False

# 导入FunASR模块
try:
    from funasr_asr import FunASRModel
    FUNASR_AVAILABLE = True
except ImportError:
    print("警告: 无法导入FunASR模块")
    print("FunASR功能将不可用。请安装所需依赖:")
    print("pip install funasr")
    FUNASR_AVAILABLE = False

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def log_message(message, log_file=None):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_msg + "\n")

def transcribe_with_whisper(audio_file, output_file, model_size="base", log_file=None):
    """使用Whisper模型转录音频文件"""
    if not WHISPER_AVAILABLE:
        log_message("错误: Whisper模块不可用", log_file)
        return False
    
    try:
        log_message(f"加载Whisper {model_size}模型...", log_file)
        model = whisper.load_model(model_size)
        
        log_message(f"开始转录: {audio_file}", log_file)
        start_time = time.time()
        
        result = model.transcribe(audio_file, language="zh")
        text = result["text"]
        
        # 保存转录结果
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        elapsed_time = time.time() - start_time
        log_message(f"转录完成: {output_file} (用时: {elapsed_time:.2f}秒)", log_file)
        return True
    
    except Exception as e:
        import traceback
        error_msg = f"转录过程中出错: {str(e)}"
        log_message(error_msg, log_file)
        log_message(traceback.format_exc(), log_file)
        return False

def transcribe_with_funasr(audio_file, output_file, model_name="paraformer-zh", 
                          model_revision="v2.0.4", use_vad=True, use_punc=True, 
                          use_spk=False, model_hub="ms", log_file=None):
    """使用FunASR模型转录音频文件"""
    if not FUNASR_AVAILABLE:
        log_message("错误: FunASR模块不可用", log_file)
        return False
    
    try:
        log_message(f"加载FunASR模型: {model_name}...", log_file)
        
        asr_model = FunASRModel(
            model=model_name,
            model_revision=model_revision,
            use_vad=use_vad,
            use_punc=use_punc,
            use_spk=use_spk,
            model_hub=model_hub
        )
        
        log_message(f"开始转录: {audio_file}", log_file)
        start_time = time.time()
        
        text = asr_model.transcribe(audio_file)
        
        # 保存转录结果
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)
        
        elapsed_time = time.time() - start_time
        log_message(f"转录完成: {output_file} (用时: {elapsed_time:.2f}秒)", log_file)
        return True
    
    except Exception as e:
        import traceback
        error_msg = f"转录过程中出错: {str(e)}"
        log_message(error_msg, log_file)
        log_message(traceback.format_exc(), log_file)
        return False

def batch_transcribe(files, output_dir, model_type, model_settings, log_file=None):
    """批量转录音频文件"""
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            log_message(f"创建输出目录: {output_dir}", log_file)
        except Exception as e:
            log_message(f"创建输出目录失败: {str(e)}", log_file)
            return False
    
    total_files = len(files)
    log_message(f"开始批量转录 {total_files} 个文件", log_file)
    
    success_count = 0
    failed_files = []
    
    for i, audio_file in enumerate(files):
        # 设置状态
        file_name = os.path.basename(audio_file)
        log_message(f"正在转录 ({i+1}/{total_files}): {file_name}", log_file)
        
        # 设置输出文件路径
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join(output_dir, f"{base_name}.txt")
        
        # 根据模型类型转录
        success = False
        if model_type == "whisper":
            success = transcribe_with_whisper(
                audio_file, 
                output_file, 
                model_settings.get("size", "base"),
                log_file
            )
        elif model_type == "funasr":
            success = transcribe_with_funasr(
                audio_file,
                output_file,
                model_settings.get("model", "paraformer-zh"),
                model_settings.get("revision", "v2.0.4"),
                model_settings.get("use_vad", True),
                model_settings.get("use_punc", True),
                model_settings.get("use_spk", False),
                model_settings.get("hub", "ms"),
                log_file
            )
        
        if success:
            success_count += 1
        else:
            failed_files.append(audio_file)
    
    # 输出统计信息
    log_message(f"批量转录完成: 成功 {success_count}/{total_files}", log_file)
    if failed_files:
        log_message("以下文件转录失败:", log_file)
        for file in failed_files:
            log_message(f"  - {file}", log_file)
    
    return success_count == total_files

def scan_directory(directory, extensions=None):
    """扫描目录获取音频文件"""
    if extensions is None:
        extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    
    return sorted(files)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="语音识别终端交互程序")
    
    # 基本参数
    parser.add_argument("-i", "--input", help="输入音频文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件或目录路径")
    parser.add_argument("-m", "--model", choices=["whisper", "funasr"], default="whisper", help="使用的模型类型")
    parser.add_argument("-b", "--batch", action="store_true", help="批量处理模式")
    parser.add_argument("-l", "--log", help="日志文件路径")
    parser.add_argument("-e", "--extensions", help="音频文件扩展名，逗号分隔，例如: mp3,wav,flac")
    
    # 交互模式
    parser.add_argument("--interactive", action="store_true", help="启用交互模式")
    
    # API服务模式
    parser.add_argument("--api", action="store_true", help="启动API服务")
    parser.add_argument("--host", default="127.0.0.1", help="API服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="API服务器端口")
    parser.add_argument("--debug", action="store_true", help="启用API服务调试模式")
    
    # Whisper模型参数
    parser.add_argument("--whisper-size", choices=["tiny", "base", "small", "medium", "large"], default="base", help="Whisper模型大小")
    
    # FunASR模型参数
    parser.add_argument("--funasr-model", choices=["paraformer-zh", "paraformer-zh-streaming"], default="paraformer-zh", help="FunASR模型名称")
    parser.add_argument("--funasr-revision", default="v2.0.4", help="FunASR模型版本")
    parser.add_argument("--funasr-hub", choices=["ms", "hf"], default="ms", help="FunASR模型仓库")
    parser.add_argument("--no-vad", action="store_true", help="不使用语音活动检测")
    parser.add_argument("--no-punc", action="store_true", help="不使用标点符号预测")
    parser.add_argument("--use-spk", action="store_true", help="使用说话人识别")
    
    return parser.parse_args()

def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')

def select_file(prompt, file_types=None):
    """选择文件"""
    if file_types is None:
        file_types = "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg)|*.mp3;*.wav;*.m4a;*.flac;*.ogg|所有文件 (*.*)|*.*"
    
    print(prompt)
    print("请输入文件路径，或按回车键取消:")
    file_path = input("> ").strip()
    
    if not file_path:
        return None
    
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在: {file_path}")
        return None
    
    if not os.path.isfile(file_path):
        print(f"错误: 不是文件: {file_path}")
        return None
    
    return file_path

def select_directory(prompt):
    """选择目录"""
    print(prompt)
    print("请输入目录路径，或按回车键取消:")
    dir_path = input("> ").strip()
    
    if not dir_path:
        return None
    
    if not os.path.exists(dir_path):
        print(f"目录 '{dir_path}' 不存在，是否创建? (y/n)")
        choice = input("> ").strip().lower()
        if choice == 'y':
            try:
                os.makedirs(dir_path)
                print(f"已创建目录: {dir_path}")
            except Exception as e:
                print(f"创建目录失败: {str(e)}")
                return None
        else:
            return None
    
    if not os.path.isdir(dir_path):
        print(f"错误: 不是目录: {dir_path}")
        return None
    
    return dir_path

def select_audio_files():
    """选择音频文件"""
    print("选择音频文件方式:")
    print("1. 选择单个文件")
    print("2. 选择目录")
    print("3. 输入文件列表")
    print("0. 取消")
    
    choice = input("> ").strip()
    
    if choice == "1":
        # 选择单个文件
        file_path = select_file("请选择音频文件")
        if file_path:
            return [file_path]
        return []
    
    elif choice == "2":
        # 选择目录
        dir_path = select_directory("请选择音频文件目录")
        if not dir_path:
            return []
        
        # 扫描目录
        files = scan_directory(dir_path)
        if not files:
            print(f"在目录 '{dir_path}' 中未找到音频文件")
            return []
        
        print(f"找到 {len(files)} 个音频文件")
        return files
    
    elif choice == "3":
        # 输入文件列表
        print("请输入文件列表，每行一个文件路径，输入空行结束:")
        files = []
        while True:
            line = input().strip()
            if not line:
                break
            
            if os.path.exists(line) and os.path.isfile(line):
                files.append(line)
            else:
                print(f"警告: 文件不存在或不是文件: {line}")
        
        if not files:
            print("没有有效的音频文件")
            return []
        
        print(f"添加了 {len(files)} 个文件")
        return files
    
    else:
        return []

def select_model_type():
    """选择模型类型"""
    available_models = []
    
    if WHISPER_AVAILABLE:
        available_models.append("whisper")
    
    if FUNASR_AVAILABLE:
        available_models.append("funasr")
    
    if not available_models:
        print("错误: 没有可用的模型")
        return None
    
    print("请选择模型类型:")
    for i, model in enumerate(available_models, 1):
        print(f"{i}. {model}")
    print("0. 取消")
    
    while True:
        choice = input("> ").strip()
        if choice == "0":
            return None
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(available_models):
                return available_models[index]
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")

def configure_whisper_model():
    """配置Whisper模型"""
    if not WHISPER_AVAILABLE:
        print("错误: Whisper模块不可用")
        return {}
    
    print("请选择Whisper模型大小:")
    sizes = ["tiny", "base", "small", "medium", "large"]
    for i, size in enumerate(sizes, 1):
        print(f"{i}. {size}")
    print("0. 取消")
    
    while True:
        choice = input("> ").strip()
        if choice == "0":
            return {}
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(sizes):
                return {"size": sizes[index]}
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")

def configure_funasr_model():
    """配置FunASR模型"""
    if not FUNASR_AVAILABLE:
        print("错误: FunASR模块不可用")
        return {}
    
    model_settings = {}
    
    # 选择模型名称
    print("请选择FunASR模型名称:")
    models = ["paraformer-zh", "paraformer-zh-streaming"]
    for i, model in enumerate(models, 1):
        print(f"{i}. {model}")
    print("0. 取消")
    
    while True:
        choice = input("> ").strip()
        if choice == "0":
            return {}
        
        try:
            index = int(choice) - 1
            if 0 <= index < len(models):
                model_settings["model"] = models[index]
                break
            else:
                print("无效的选择，请重试")
        except ValueError:
            print("请输入数字")
    
    # 输入模型版本
    print("请输入模型版本 (默认: v2.0.4):")
    revision = input("> ").strip()
    model_settings["revision"] = revision or "v2.0.4"
    
    # 选择模型仓库
    print("请选择模型仓库:")
    print("1. ModelScope (ms)")
    print("2. HuggingFace (hf)")
    print("0. 取消")
    
    while True:
        choice = input("> ").strip()
        if choice == "0":
            return {}
        
        if choice == "1":
            model_settings["hub"] = "ms"
            break
        elif choice == "2":
            model_settings["hub"] = "hf"
            break
        else:
            print("无效的选择，请重试")
    
    # 功能选项
    print("是否使用语音活动检测 (VAD)? (y/n, 默认: y)")
    choice = input("> ").strip().lower()
    model_settings["use_vad"] = choice != "n"
    
    print("是否使用标点符号预测? (y/n, 默认: y)")
    choice = input("> ").strip().lower()
    model_settings["use_punc"] = choice != "n"
    
    print("是否使用说话人识别? (y/n, 默认: n)")
    choice = input("> ").strip().lower()
    model_settings["use_spk"] = choice == "y"
    
    return model_settings

def configure_ai_model():
    """配置AI模型"""
    print("\n配置AI模型:")
    print("1. Ollama")
    print("2. 自定义API")
    print("0. 取消")
    
    choice = input("> ").strip()
    
    if choice == "1":
        # Ollama
        print("\n配置Ollama:")
        
        model = input("模型名称 [llama3]: ").strip() or "llama3"
        api_url = input("API URL [http://localhost:11434]: ").strip() or "http://localhost:11434"
        
        return {
            "type": "ollama",
            "model": model,
            "api_url": api_url
        }
    
    elif choice == "2":
        # 自定义API
        print("\n配置自定义API:")
        
        api_url = input("API URL: ").strip()
        if not api_url:
            print("错误: 必须提供API URL")
            return None
        
        api_key = input("API Key (可选): ").strip()
        model = input("模型名称: ").strip()
        if not model:
            print("错误: 必须提供模型名称")
            return None
        
        return {
            "type": "custom",
            "model": model,
            "api_url": api_url,
            "api_key": api_key
        }
    
    else:
        return None

def interactive_mode():
    """交互式模式"""
    while True:
        clear_screen()
        print("语音识别终端交互程序")
        print("=" * 30)
        print("1. 单文件转录")
        print("2. 批量转录")
        print("3. 模型设置")
        print("4. 启动API服务")
        print("5. 帮助信息")
        print("0. 退出程序")
        print("=" * 30)
        
        choice = input("请选择操作: ").strip()
        
        if choice == "1":
            # 单文件转录
            single_file_transcription()
        elif choice == "2":
            # 批量转录
            batch_transcription()
        elif choice == "3":
            # 模型设置
            configure_model()
        elif choice == "4":
            # 启动API服务
            start_api_server()
        elif choice == "5":
            # 帮助信息
            show_help()
        elif choice == "0":
            # 退出程序
            print("感谢使用，再见!")
            return 0
        else:
            print("无效的选择，请重试")
            input("按回车键继续...")
    
    return 0

def start_api_server():
    """启动API服务"""
    clear_screen()
    print("启动API服务")
    print("=" * 30)
    
    # 获取服务器设置
    host = input("服务器主机地址 [127.0.0.1]: ").strip() or "127.0.0.1"
    
    while True:
        port_str = input("服务器端口 [5000]: ").strip() or "5000"
        try:
            port = int(port_str)
            break
        except ValueError:
            print("错误: 端口必须是数字")
    
    debug = input("启用调试模式? (y/n) [n]: ").strip().lower() == 'y'
    
    print("\n正在启动API服务...")
    print(f"服务地址: http://{host}:{port}")
    print("按Ctrl+C停止服务")
    
    try:
        # 导入API服务模块
        from speech_to_text_api import app
        
        # 启动服务
        app.run(host=host, port=port, debug=debug)
    
    except ImportError:
        print("\n错误: 无法导入API服务模块")
        print("请确保已安装Flask:")
        print("pip install flask")
    
    except KeyboardInterrupt:
        print("\n服务已停止")
    
    except Exception as e:
        print(f"\n启动API服务时出错: {str(e)}")
    
    input("\n按回车键返回主菜单...")

def main():
    """主函数"""
    args = parse_arguments()
    
    # 如果指定了交互模式，则进入交互模式
    if args.interactive or len(sys.argv) == 1:
        return interactive_mode()
    
    # 如果指定了API服务模式，则启动API服务
    if args.api:
        try:
            from speech_to_text_api import app
            
            print(f"启动API服务: http://{args.host}:{args.port}")
            app.run(host=args.host, port=args.port, debug=args.debug)
            return 0
        
        except ImportError:
            print("错误: 无法导入API服务模块")
            print("请确保已安装Flask:")
            print("pip install flask")
            return 1
        
        except Exception as e:
            print(f"启动API服务时出错: {str(e)}")
            return 1
    
    # 以下是命令行参数模式
    
    # 设置日志文件
    log_file = args.log
    if log_file:
        # 确保日志目录存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
            except Exception as e:
                print(f"创建日志目录失败: {str(e)}")
                log_file = None
    
    # 检查模型可用性
    if args.model == "whisper" and not WHISPER_AVAILABLE:
        log_message("错误: 选择了Whisper模型，但Whisper模块不可用", log_file)
        return 1
    
    if args.model == "funasr" and not FUNASR_AVAILABLE:
        log_message("错误: 选择了FunASR模型，但FunASR模块不可用", log_file)
        return 1
    
    # 检查输入路径
    if not args.input:
        log_message("错误: 未指定输入路径", log_file)
        return 1
    
    if not os.path.exists(args.input):
        log_message(f"错误: 输入路径不存在: {args.input}", log_file)
        return 1
    
    # 准备模型设置
    model_settings = {}
    if args.model == "whisper":
        model_settings["size"] = args.whisper_size
    elif args.model == "funasr":
        model_settings["model"] = args.funasr_model
        model_settings["revision"] = args.funasr_revision
        model_settings["hub"] = args.funasr_hub
        model_settings["use_vad"] = not args.no_vad
        model_settings["use_punc"] = not args.no_punc
        model_settings["use_spk"] = args.use_spk
    
    # 处理文件扩展名
    extensions = None
    if args.extensions:
        extensions = [f".{ext.strip()}" for ext in args.extensions.split(",")]
    
    # 批量处理模式
    if args.batch or os.path.isdir(args.input):
        # 如果输入是目录，则扫描目录获取音频文件
        if os.path.isdir(args.input):
            files = scan_directory(args.input, extensions)
            if not files:
                log_message(f"错误: 在目录 '{args.input}' 中未找到音频文件", log_file)
                return 1
        else:
            # 如果输入是文件，则将其作为文件列表文件读取
            try:
                with open(args.input, "r", encoding="utf-8") as f:
                    files = [line.strip() for line in f if line.strip()]
                
                # 验证文件是否存在
                valid_files = []
                for file in files:
                    if os.path.exists(file):
                        valid_files.append(file)
                    else:
                        log_message(f"警告: 文件不存在: {file}", log_file)
                
                files = valid_files
                if not files:
                    log_message("错误: 没有有效的音频文件", log_file)
                    return 1
            except Exception as e:
                log_message(f"读取文件列表失败: {str(e)}", log_file)
                return 1
        
        # 设置输出目录
        output_dir = args.output or "output"
        
        # 批量转录
        success = batch_transcribe(files, output_dir, args.model, model_settings, log_file)
        return 0 if success else 1
    
    # 单文件处理模式
    else:
        # 检查输入文件
        if not os.path.isfile(args.input):
            log_message(f"错误: 输入不是文件: {args.input}", log_file)
            return 1
        
        # 设置输出文件
        if not args.output:
            base_name = os.path.splitext(args.input)[0]
            output_file = f"{base_name}.txt"
        else:
            if os.path.isdir(args.output):
                base_name = os.path.splitext(os.path.basename(args.input))[0]
                output_file = os.path.join(args.output, f"{base_name}.txt")
            else:
                output_file = args.output
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                log_message(f"创建输出目录失败: {str(e)}", log_file)
                return 1
        
        # 根据模型类型转录
        success = False
        if args.model == "whisper":
            success = transcribe_with_whisper(
                args.input, 
                output_file, 
                args.whisper_size,
                log_file
            )
        elif args.model == "funasr":
            success = transcribe_with_funasr(
                args.input,
                output_file,
                args.funasr_model,
                args.funasr_revision,
                not args.no_vad,
                not args.no_punc,
                args.use_spk,
                args.funasr_hub,
                log_file
            )
        
        return 0 if success else 1

def generate_summary(text, ai_settings, max_length=200, language="chinese"):
    """生成文本摘要"""
    if not ai_settings:
        print("错误: 未配置AI模型")
        return None
    
    try:
        from ai_summary import AIModelManager
        
        # 创建AI模型管理器
        if ai_settings["type"] == "ollama":
            ai_manager = AIModelManager(
                model_type="ollama",
                model_name=ai_settings["model"],
                api_url=ai_settings["api_url"]
            )
        else:  # custom
            ai_manager = AIModelManager(
                model_type="custom",
                model_name=ai_settings["model"],
                api_url=ai_settings["api_url"],
                api_key=ai_settings.get("api_key")
            )
        
        # 测试连接
        print("正在测试AI连接...")
        if not ai_manager.test_connection():
            print("错误: 无法连接到AI模型")
            return None
        
        # 生成摘要
        print("正在生成摘要...")
        summary = ai_manager.generate_summary(text, max_length, language)
        
        return summary
    
    except ImportError:
        print("错误: 无法导入AI摘要模块")
        return None
    except Exception as e:
        print(f"生成摘要时出错: {str(e)}")
        return None

if __name__ == "__main__":
    sys.exit(main()) 