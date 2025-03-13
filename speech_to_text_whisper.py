import os
import torch
import whisper
import tqdm
from datetime import timedelta
import glob

def convert_audio_to_text(audio_file_path, output_file_path=None, model_size="base"):
    """使用Whisper模型将音频转换为文本，并添加时间戳"""
    try:
        # 规范化文件路径
        audio_file_path = os.path.abspath(os.path.normpath(audio_file_path))
        
        # 检查文件是否存在
        if not os.path.exists(audio_file_path):
            print(f"错误: 文件 '{audio_file_path}' 不存在")
            return None
            
        # 设置默认输出文件路径
        if output_file_path is None:
            base_name = os.path.splitext(audio_file_path)[0]
            output_file_path = f"{base_name}.txt"
        
        # 加载Whisper模型
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 使用tqdm显示加载模型的进度
        print(f"加载Whisper模型 ({model_size})...")
        with tqdm.tqdm(total=100, desc="加载模型") as pbar:
            pbar.update(10)
            model = whisper.load_model(model_size, device=device)
            pbar.update(90)
        
        # 转录音频
        print(f"正在处理音频文件: {os.path.basename(audio_file_path)}...")
        
        # 使用verbose=False来禁用Whisper内置的进度条，我们将使用自定义进度条
        result = model.transcribe(
            audio_file_path, 
            verbose=False,
            word_timestamps=True,  # 启用单词级时间戳
            fp16=torch.cuda.is_available()  # 如果有CUDA，使用FP16加速
        )
        
        # 获取带时间戳的文本
        segments = result["segments"]
        
        # 创建带时间戳的转录文本
        transcription_with_timestamps = ""
        raw_transcription = ""
        
        # 使用tqdm显示处理进度
        print("处理时间戳和格式化输出...")
        for segment in tqdm.tqdm(segments, desc="处理段落"):
            # 获取开始时间并格式化为HH:MM:SS
            start_time = str(timedelta(seconds=int(segment['start']))).split('.')[0]
            if start_time.startswith('0:'):  # 去掉前导的0:
                start_time = start_time[2:]
            if len(start_time.split(':')) == 2:  # 如果只有MM:SS，添加00:
                start_time = f"00:{start_time}"
            
            # 添加时间戳和文本
            segment_text = segment['text'].strip()
            transcription_with_timestamps += f"[{start_time}] {segment_text}\n"
            raw_transcription += segment_text + " "
        
        # 保存转录结果到文件
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(transcription_with_timestamps)
            
        print(f"转录完成! 结果已保存到: {output_file_path}")
        return raw_transcription
        
    except Exception as e:
        print(f"转录过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def process_batch(directory, pattern="*.mp3", output_dir=None, model_size="base"):
    """批量处理目录中的音频文件"""
    # 获取所有匹配的文件
    files = glob.glob(os.path.join(directory, pattern))
    
    if not files:
        print(f"在目录 '{directory}' 中没有找到匹配 '{pattern}' 的文件")
        return
    
    print(f"找到 {len(files)} 个文件待处理")
    
    # 如果指定了输出目录，确保它存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 处理每个文件
    for i, file_path in enumerate(files):
        print(f"\n处理文件 {i+1}/{len(files)}: {os.path.basename(file_path)}")
        
        # 设置输出路径
        if output_dir:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.txt")
        else:
            output_path = None  # 使用默认路径（与音频文件同目录）
        
        # 转换音频
        convert_audio_to_text(file_path, output_path, model_size)

def main():
    """主函数，处理命令行输入"""
    print("语音转文本工具 (Whisper版本)")
    print("=" * 50)
    print("支持的功能:")
    print("1. 单个音频文件转换")
    print("2. 批量处理目录中的音频文件")
    print("3. 选择不同的模型大小")
    print("输入 'q' 或 'exit' 退出程序")
    print("=" * 50)
    
    # 默认模型大小
    model_size = "base"
    
    while True:
        print("\n请选择操作模式:")
        print("1. 处理单个文件")
        print("2. 批量处理目录")
        print("3. 更改模型大小 (当前: " + model_size + ")")
        print("q. 退出程序")
        
        choice = input("请输入选项 [1/2/3/q]: ").strip().lower()
        
        if choice in ['q', 'exit']:
            print("程序已退出")
            break
        
        elif choice == '1':
            # 单个文件处理
            audio_path = input("\n请输入音频文件路径: ").strip()
            
            # 检查是否退出
            if audio_path.lower() in ['q', 'exit']:
                continue
            
            # 处理路径中的引号和空格
            audio_path = audio_path.strip('"\'')
            
            # 转换为绝对路径并规范化
            audio_path = os.path.abspath(os.path.normpath(audio_path))
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                print(f"错误: 文件 '{audio_path}' 不存在")
                print("提示: 请确保路径格式正确，Windows路径示例: C:\\Users\\username\\file.wav 或 C:/Users/username/file.wav")
                print("      请不要在路径两端加引号")
                continue
            
            # 获取输出文件路径（可选）
            output_path = input("请输入输出文本文件路径 (留空则使用默认路径): ").strip()
            if output_path:
                output_path = output_path.strip('"\'')
                output_path = os.path.abspath(os.path.normpath(output_path))
            else:
                output_path = None
            
            # 执行转换
            result = convert_audio_to_text(audio_path, output_path, model_size)
            
            if result:
                print("\n转录结果预览 (不含时间戳):")
                print("-" * 50)
                print(result[:500] + ("..." if len(result) > 500 else ""))
                print("-" * 50)
                print("完整结果（含时间戳）已保存到文件中")
        
        elif choice == '2':
            # 批量处理
            directory = input("\n请输入音频文件所在目录: ").strip()
            
            # 检查是否退出
            if directory.lower() in ['q', 'exit']:
                continue
            
            # 处理路径中的引号和空格
            directory = directory.strip('"\'')
            
            # 转换为绝对路径并规范化
            directory = os.path.abspath(os.path.normpath(directory))
            
            # 检查目录是否存在
            if not os.path.isdir(directory):
                print(f"错误: 目录 '{directory}' 不存在")
                continue
            
            # 获取文件模式
            pattern = input("请输入要处理的文件模式 (例如: *.mp3, *.wav, 留空则处理所有音频文件): ").strip()
            if not pattern:
                pattern = "*.mp3 *.wav *.m4a *.flac *.ogg"
            
            # 获取输出目录（可选）
            output_dir = input("请输入输出文本文件目录 (留空则与音频文件同目录): ").strip()
            if output_dir:
                output_dir = output_dir.strip('"\'')
                output_dir = os.path.abspath(os.path.normpath(output_dir))
            else:
                output_dir = None
            
            # 执行批量处理
            for p in pattern.split():
                process_batch(directory, p, output_dir, model_size)
        
        elif choice == '3':
            # 更改模型大小
            print("\n可用的模型大小:")
            print("tiny   - 最小模型，速度最快，精度最低")
            print("base   - 基础模型，速度和精度平衡 (默认)")
            print("small  - 小型模型，精度较高")
            print("medium - 中型模型，精度高")
            print("large  - 大型模型，精度最高，速度最慢")
            
            new_size = input("请选择模型大小 [tiny/base/small/medium/large]: ").strip().lower()
            
            if new_size in ['tiny', 'base', 'small', 'medium', 'large']:
                model_size = new_size
                print(f"模型大小已更改为: {model_size}")
            else:
                print(f"无效的选择: {new_size}，保持当前模型大小: {model_size}")
        
        else:
            print(f"无效的选择: {choice}")

if __name__ == "__main__":
    main() 