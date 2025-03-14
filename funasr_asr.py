import os
import sys
import tempfile
import subprocess

# 更可靠的依赖检查方法
def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# 检查必要的依赖
required_packages = ['funasr']
missing_packages = []

for package in required_packages:
    if not is_package_installed(package):
        missing_packages.append(package)

if missing_packages:
    print(f"错误: 缺少必要的依赖项: {', '.join(missing_packages)}")
    print("请使用以下命令安装缺失的依赖:")
    print(f"pip install {' '.join(missing_packages)}")
    
    # 提供一个模拟类，这样应用程序仍然可以启动
    class FunASRModel:
        def __init__(self, **kwargs):
            self.missing_packages = missing_packages
            
        def transcribe(self, audio_path):
            return f"无法转录: 缺少必要的依赖项: {', '.join(self.missing_packages)}"
else:
    # 如果所有依赖都已安装，导入FunASR
    try:
        from funasr import AutoModel
        
        class FunASRModel:
            """FunASR语音识别模型"""
            
            def __init__(self, model="paraformer-zh", model_revision="v2.0.4", 
                         use_vad=True, use_punc=True, use_spk=False, model_hub="ms"):
                """
                初始化FunASR模型
                
                Args:
                    model: 模型名称，例如 'paraformer-zh'
                    model_revision: 模型版本，例如 'v2.0.4'
                    use_vad: 是否使用语音活动检测
                    use_punc: 是否使用标点符号预测
                    use_spk: 是否使用说话人识别
                    model_hub: 模型仓库，'ms'或'hf'
                """
                self.model = model
                self.model_revision = model_revision
                self.use_vad = use_vad
                self.use_punc = use_punc
                self.use_spk = use_spk
                self.model_hub = model_hub
                self.asr_pipeline = None
                
                # 加载模型
                self.load_model()
            
            def load_model(self):
                """加载模型"""
                try:
                    print(f"正在加载FunASR模型: {self.model}...")
                    
                    # 设置模型参数
                    model_params = {
                        "model": self.model,
                        "model_revision": self.model_revision,
                        "model_hub": self.model_hub,
                        "batch_size": 1  # 明确设置batch_size为1
                    }
                    
                    # 添加VAD、标点和说话人识别模型
                    if self.use_vad:
                        model_params["vad_model"] = "fsmn-vad"
                        model_params["vad_model_revision"] = self.model_revision
                    
                    if self.use_punc:
                        model_params["punc_model"] = "ct-punc"
                        model_params["punc_model_revision"] = self.model_revision
                    
                    if self.use_spk:
                        model_params["spk_model"] = "cam++"
                        model_params["spk_model_revision"] = self.model_revision
                    
                    # 创建模型
                    self.asr_pipeline = AutoModel(**model_params)
                    
                    print("FunASR模型加载成功!")
                    return True
                
                except Exception as e:
                    import traceback
                    print(f"加载FunASR模型失败: {str(e)}")
                    print(traceback.format_exc())
                    return False
            
            def transcribe(self, audio_path):
                """
                转录音频文件
                
                Args:
                    audio_path: 音频文件路径
                
                Returns:
                    转录结果文本
                """
                if not self.asr_pipeline:
                    return "错误: 模型未加载"
                
                try:
                    if not os.path.exists(audio_path):
                        return f"错误: 文件不存在: {audio_path}"
                    
                    print(f"开始转录文件: {audio_path}")
                    
                    # 预处理音频
                    processed_audio = self._preprocess_audio(audio_path)
                    if processed_audio:
                        audio_path = processed_audio
                    
                    # 转录
                    result = self.asr_pipeline.generate(input=audio_path)
                    
                    # 清理临时文件
                    if processed_audio and processed_audio != audio_path:
                        try:
                            os.remove(processed_audio)
                        except:
                            pass
                    
                    # 提取文本
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and "text" in result[0]:
                            return result[0]["text"]
                        else:
                            return str(result[0])
                    else:
                        return str(result)
                
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"转录过程中出错: {str(e)}")
                    print(error_trace)
                    return f"转录过程中出错: {str(e)}"
            
            def _preprocess_audio(self, audio_path):
                """预处理音频文件"""
                try:
                    # 检查是否需要转换
                    import tempfile
                    import subprocess
                    
                    # 创建临时WAV文件
                    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
                    
                    # 使用ffmpeg转换为16kHz, 单声道WAV
                    cmd = ['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', '-f', 'wav', temp_wav]
                    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    
                    print(f"音频预处理成功: {audio_path} -> {temp_wav}")
                    return temp_wav
                
                except Exception as e:
                    print(f"音频预处理失败: {str(e)}")
                    return None
    
    except ImportError as e:
        print(f"导入FunASR时出错: {e}")
        print("请确保已正确安装FunASR及其依赖项:")
        print("pip install funasr")
        
        # 提供一个模拟类，这样应用程序仍然可以启动
        class FunASRModel:
            def __init__(self, **kwargs):
                pass
                
            def transcribe(self, audio_path):
                return f"无法转录: FunASR导入失败: {e}" 