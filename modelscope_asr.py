import os
import sys

# 更可靠的依赖检查方法
def is_package_installed(package_name):
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

# 检查必要的依赖
required_packages = ['numpy', 'soundfile', 'addict', 'pyyaml', 'datasets', 'transformers', 'modelscope', 'simplejson']
missing_packages = []

for package in required_packages:
    # 特殊处理pyyaml，因为它的导入名称是yaml而不是pyyaml
    if package == 'pyyaml':
        if not is_package_installed('yaml'):
            missing_packages.append(package)
    else:
        if not is_package_installed(package):
            missing_packages.append(package)

if missing_packages:
    print(f"错误: 缺少必要的依赖项: {', '.join(missing_packages)}")
    print("请使用以下命令安装缺失的依赖:")
    print(f"pip install {' '.join(missing_packages)}")
    
    # 不立即退出，而是提供一个模拟类，这样应用程序仍然可以启动
    class ModelScopeASR:
        def __init__(self, model_id=None, device='cpu'):
            self.model_id = model_id
            self.device = device
            self.missing_packages = missing_packages
            
        def load_model(self):
            return False
            
        def transcribe(self, audio_path):
            return f"无法转录: 缺少必要的依赖项: {', '.join(self.missing_packages)}"
else:
    # 如果所有依赖都已安装，导入ModelScope
    import numpy as np
    import soundfile as sf
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        
        class ModelScopeASR:
            """ModelScope语音识别模型封装类"""
            
            def __init__(self, model_id=None, device='cpu'):
                """
                初始化ModelScope ASR模型
                
                Args:
                    model_id: 模型ID，例如 'Jieer2024/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn-onnx'
                    device: 运行设备，'cpu'或'cuda'
                """
                # 默认使用一个已知工作良好的模型
                self.model_id = model_id or 'damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn'
                self.device = device
                self.inference_pipeline = None
                
            def load_model(self):
                """加载模型"""
                try:
                    print(f"正在加载ModelScope模型: {self.model_id}...")
                    
                    # 检查simplejson是否已安装
                    try:
                        import simplejson
                    except ImportError:
                        print("错误: 缺少simplejson模块，这是ModelScope的必要依赖")
                        print("请安装: pip install simplejson")
                        return False
                        
                    # 尝试修复配置文件问题
                    try:
                        from modelscope.hub.snapshot_download import snapshot_download
                        from modelscope.utils.config import Config
                        import json
                        
                        # 下载模型
                        model_dir = snapshot_download(self.model_id)
                        config_path = os.path.join(model_dir, 'configuration.json')
                        
                        # 检查配置文件是否存在
                        if os.path.exists(config_path):
                            # 读取配置
                            with open(config_path, 'r', encoding='utf-8') as f:
                                config = json.load(f)
                            
                            # 检查是否缺少framework字段
                            if 'framework' not in config:
                                print("检测到配置文件缺少framework字段，尝试修复...")
                                # 添加framework字段
                                config['framework'] = 'pytorch'  # 或者'tensorflow'，取决于模型
                                
                                # 保存修改后的配置
                                with open(config_path, 'w', encoding='utf-8') as f:
                                    json.dump(config, f, indent=2)
                                
                                print("配置文件已修复")
                    except Exception as e:
                        print(f"尝试修复配置文件时出错: {e}")
                        # 继续尝试加载模型，可能会失败
                        
                    # 创建管道
                    self.inference_pipeline = pipeline(
                        task=Tasks.auto_speech_recognition,
                        model=self.model_id,
                        device=self.device
                    )
                    print("ModelScope模型加载成功!")
                    return True
                except Exception as e:
                    print(f"加载ModelScope模型失败: {str(e)}")
                    import traceback
                    print(traceback.format_exc())
                    
                    # 提供更具体的错误信息和解决方案
                    error_str = str(e)
                    if "No module named 'simplejson'" in error_str:
                        print("\n解决方案: 请安装simplejson模块")
                        print("pip install simplejson\n")
                    elif "Attribute framework is missing from configuration.json" in error_str:
                        print("\n解决方案: 配置文件缺少framework字段")
                        print("请尝试手动修改配置文件，添加 'framework': 'pytorch' 字段\n")
                    
                    return False
            
            def transcribe(self, audio_path):
                """
                转录音频文件
                
                Args:
                    audio_path: 音频文件路径
                    
                Returns:
                    转录结果文本
                """
                if not self.inference_pipeline:
                    if not self.load_model():
                        return "模型加载失败，无法进行转录"
                
                try:
                    # 检查文件是否存在
                    if not os.path.exists(audio_path):
                        return f"错误: 文件不存在: {audio_path}"
                        
                    print(f"开始转录文件: {audio_path}")
                    # 读取音频文件
                    rec_result = self.inference_pipeline(audio_path)
                    
                    # 返回转录结果
                    if 'text' in rec_result:
                        return rec_result['text']
                    else:
                        return str(rec_result)
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    print(f"转录过程中出错: {str(e)}")
                    print(error_trace)
                    return f"转录过程中出错: {str(e)}"
    
    except ImportError as e:
        print(f"导入ModelScope时出错: {e}")
        print("请确保已正确安装ModelScope及其依赖项:")
        print("pip install modelscope addict pyyaml datasets transformers simplejson")
        
        # 提供一个模拟类，这样应用程序仍然可以启动
        class ModelScopeASR:
            def __init__(self, model_id=None, device='cpu'):
                self.model_id = model_id
                self.device = device
                
            def load_model(self):
                return False
                
            def transcribe(self, audio_path):
                return f"无法转录: ModelScope导入失败: {e}" 