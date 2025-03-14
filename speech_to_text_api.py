#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音识别API服务
提供REST API接口的语音转文字功能
"""

import os
import sys
import time
import json
import tempfile
import argparse
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename

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

# 导入AI摘要模块
try:
    from ai_summary import AIModelManager
    AI_SUMMARY_AVAILABLE = True
except ImportError:
    print("警告: 无法导入AI摘要模块")
    print("AI摘要功能将不可用")
    AI_SUMMARY_AVAILABLE = False

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 创建Flask应用
app = Flask(__name__)

# 配置
app.config['UPLOAD_FOLDER'] = os.path.join(tempfile.gettempdir(), 'speech_to_text_uploads')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为100MB
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'flac', 'ogg', 'm4a'}

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 日志文件
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'api_server.log')

# 模型实例
whisper_models = {}
funasr_model = None
ai_manager = None

def log_message(message):
    """记录日志消息"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(log_msg + "\n")

def allowed_file(filename):
    """检查文件是否允许上传"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_whisper_model(model_size):
    """获取Whisper模型实例"""
    global whisper_models
    
    if not WHISPER_AVAILABLE:
        return None
    
    if model_size not in whisper_models:
        try:
            log_message(f"加载Whisper {model_size}模型...")
            whisper_models[model_size] = whisper.load_model(model_size)
            log_message(f"Whisper {model_size}模型加载成功")
        except Exception as e:
            log_message(f"加载Whisper {model_size}模型失败: {str(e)}")
            return None
    
    return whisper_models[model_size]

def get_funasr_model(model_name="paraformer-zh", model_revision="v2.0.4", 
                    use_vad=True, use_punc=True, use_spk=False, model_hub="ms"):
    """获取FunASR模型实例"""
    global funasr_model
    
    if not FUNASR_AVAILABLE:
        return None
    
    if funasr_model is None:
        try:
            log_message(f"加载FunASR {model_name}模型...")
            funasr_model = FunASRModel(
                model=model_name,
                model_revision=model_revision,
                use_vad=use_vad,
                use_punc=use_punc,
                use_spk=use_spk,
                model_hub=model_hub
            )
            log_message("FunASR模型加载成功")
        except Exception as e:
            log_message(f"加载FunASR模型失败: {str(e)}")
            return None
    
    return funasr_model

def get_ai_manager(model_type="ollama", model_name="llama3", api_url=None, api_key=None):
    """获取AI模型管理器实例"""
    global ai_manager
    
    if not AI_SUMMARY_AVAILABLE:
        return None
    
    if ai_manager is None:
        try:
            log_message(f"初始化AI模型管理器: {model_type}/{model_name}...")
            ai_manager = AIModelManager(
                model_type=model_type,
                model_name=model_name,
                api_url=api_url,
                api_key=api_key
            )
            log_message("AI模型管理器初始化成功")
        except Exception as e:
            log_message(f"初始化AI模型管理器失败: {str(e)}")
            return None
    
    return ai_manager

@app.route('/')
def index():
    """API首页"""
    return jsonify({
        "name": "语音识别API服务",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/transcribe": "转录音频文件",
            "POST /api/batch_transcribe": "批量转录音频文件",
            "POST /api/summarize": "生成文本摘要",
            "GET /api/status": "获取服务状态"
        }
    })

@app.route('/api/status')
def status():
    """获取服务状态"""
    return jsonify({
        "status": "running",
        "models": {
            "whisper": WHISPER_AVAILABLE,
            "funasr": FUNASR_AVAILABLE,
            "ai_summary": AI_SUMMARY_AVAILABLE
        },
        "loaded_models": {
            "whisper": list(whisper_models.keys()),
            "funasr": funasr_model is not None,
            "ai_manager": ai_manager is not None
        }
    })

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    """转录音频文件"""
    # 检查是否有文件
    if 'file' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    file = request.files['file']
    
    # 检查文件名
    if file.filename == '':
        return jsonify({"error": "没有选择文件"}), 400
    
    # 检查文件类型
    if not allowed_file(file.filename):
        return jsonify({"error": "不支持的文件类型"}), 400
    
    # 获取模型参数
    model_type = request.form.get('model_type', 'whisper')
    
    # 保存文件
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    log_message(f"接收到文件: {filename}, 模型类型: {model_type}")
    
    try:
        # 根据模型类型转录
        if model_type.lower() == 'whisper':
            model_size = request.form.get('model_size', 'base')
            
            model = get_whisper_model(model_size)
            if model is None:
                return jsonify({"error": "Whisper模型不可用"}), 500
            
            log_message(f"使用Whisper {model_size}模型转录: {filename}")
            start_time = time.time()
            
            result = model.transcribe(file_path, language="zh")
            text = result["text"]
            
            elapsed_time = time.time() - start_time
            log_message(f"转录完成: {filename}, 耗时: {elapsed_time:.2f}秒")
            
            # 返回结果
            return jsonify({
                "text": text,
                "model": f"whisper-{model_size}",
                "filename": filename,
                "elapsed_time": elapsed_time
            })
        
        elif model_type.lower() == 'funasr':
            model_name = request.form.get('model_name', 'paraformer-zh')
            model_revision = request.form.get('model_revision', 'v2.0.4')
            use_vad = request.form.get('use_vad', 'true').lower() == 'true'
            use_punc = request.form.get('use_punc', 'true').lower() == 'true'
            use_spk = request.form.get('use_spk', 'false').lower() == 'true'
            model_hub = request.form.get('model_hub', 'ms')
            
            model = get_funasr_model(
                model_name=model_name,
                model_revision=model_revision,
                use_vad=use_vad,
                use_punc=use_punc,
                use_spk=use_spk,
                model_hub=model_hub
            )
            
            if model is None:
                return jsonify({"error": "FunASR模型不可用"}), 500
            
            log_message(f"使用FunASR {model_name}模型转录: {filename}")
            start_time = time.time()
            
            text = model.transcribe(file_path)
            
            elapsed_time = time.time() - start_time
            log_message(f"转录完成: {filename}, 耗时: {elapsed_time:.2f}秒")
            
            # 返回结果
            return jsonify({
                "text": text,
                "model": f"funasr-{model_name}",
                "filename": filename,
                "elapsed_time": elapsed_time
            })
        
        else:
            return jsonify({"error": f"不支持的模型类型: {model_type}"}), 400
    
    except Exception as e:
        log_message(f"转录过程中出错: {str(e)}")
        return jsonify({"error": f"转录过程中出错: {str(e)}"}), 500
    
    finally:
        # 清理临时文件
        try:
            os.remove(file_path)
        except:
            pass

@app.route('/api/batch_transcribe', methods=['POST'])
def batch_transcribe():
    """批量转录音频文件"""
    # 检查是否有文件
    if 'files[]' not in request.files:
        return jsonify({"error": "没有上传文件"}), 400
    
    files = request.files.getlist('files[]')
    
    # 检查文件
    if len(files) == 0:
        return jsonify({"error": "没有选择文件"}), 400
    
    # 获取模型参数
    model_type = request.form.get('model_type', 'whisper')
    
    # 保存文件并转录
    results = []
    
    for file in files:
        # 检查文件名
        if file.filename == '':
            continue
        
        # 检查文件类型
        if not allowed_file(file.filename):
            results.append({
                "filename": file.filename,
                "error": "不支持的文件类型"
            })
            continue
        
        # 保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        log_message(f"接收到文件: {filename}, 模型类型: {model_type}")
        
        try:
            # 根据模型类型转录
            if model_type.lower() == 'whisper':
                model_size = request.form.get('model_size', 'base')
                
                model = get_whisper_model(model_size)
                if model is None:
                    results.append({
                        "filename": filename,
                        "error": "Whisper模型不可用"
                    })
                    continue
                
                log_message(f"使用Whisper {model_size}模型转录: {filename}")
                start_time = time.time()
                
                result = model.transcribe(file_path, language="zh")
                text = result["text"]
                
                elapsed_time = time.time() - start_time
                log_message(f"转录完成: {filename}, 耗时: {elapsed_time:.2f}秒")
                
                # 添加结果
                results.append({
                    "filename": filename,
                    "text": text,
                    "model": f"whisper-{model_size}",
                    "elapsed_time": elapsed_time
                })
            
            elif model_type.lower() == 'funasr':
                model_name = request.form.get('model_name', 'paraformer-zh')
                model_revision = request.form.get('model_revision', 'v2.0.4')
                use_vad = request.form.get('use_vad', 'true').lower() == 'true'
                use_punc = request.form.get('use_punc', 'true').lower() == 'true'
                use_spk = request.form.get('use_spk', 'false').lower() == 'true'
                model_hub = request.form.get('model_hub', 'ms')
                
                model = get_funasr_model(
                    model_name=model_name,
                    model_revision=model_revision,
                    use_vad=use_vad,
                    use_punc=use_punc,
                    use_spk=use_spk,
                    model_hub=model_hub
                )
                
                if model is None:
                    results.append({
                        "filename": filename,
                        "error": "FunASR模型不可用"
                    })
                    continue
                
                log_message(f"使用FunASR {model_name}模型转录: {filename}")
                start_time = time.time()
                
                text = model.transcribe(file_path)
                
                elapsed_time = time.time() - start_time
                log_message(f"转录完成: {filename}, 耗时: {elapsed_time:.2f}秒")
                
                # 添加结果
                results.append({
                    "filename": filename,
                    "text": text,
                    "model": f"funasr-{model_name}",
                    "elapsed_time": elapsed_time
                })
            
            else:
                results.append({
                    "filename": filename,
                    "error": f"不支持的模型类型: {model_type}"
                })
            
        except Exception as e:
            log_message(f"转录文件 {filename} 时出错: {str(e)}")
            results.append({
                "filename": filename,
                "error": f"转录过程中出错: {str(e)}"
            })
        
        finally:
            # 清理临时文件
            try:
                os.remove(file_path)
            except:
                pass
    
    # 返回结果
    return jsonify({
        "results": results,
        "total": len(files),
        "success": len([r for r in results if "error" not in r]),
        "failed": len([r for r in results if "error" in r])
    })

@app.route('/api/summarize', methods=['POST'])
def summarize():
    """生成文本摘要"""
    # 获取请求数据
    data = request.json
    
    if not data or 'text' not in data:
        return jsonify({"error": "缺少文本内容"}), 400
    
    text = data['text']
    
    # 获取AI模型设置
    model_type = data.get('model_type', 'ollama')
    model_name = data.get('model_name', 'llama3')
    api_url = data.get('api_url', 'http://localhost:11434')
    api_key = data.get('api_key', '')
    
    # 获取摘要设置
    max_length = data.get('max_length', 200)
    language = data.get('language', 'chinese')
    
    # 初始化AI模型管理器
    manager = get_ai_manager(
        model_type=model_type,
        model_name=model_name,
        api_url=api_url,
        api_key=api_key
    )
    
    if manager is None:
        return jsonify({"error": "AI模型管理器不可用"}), 500
    
    try:
        log_message(f"使用{model_type}/{model_name}生成摘要")
        start_time = time.time()
        
        # 生成摘要
        summary = manager.generate_summary(text, max_length, language)
        
        elapsed_time = time.time() - start_time
        log_message(f"摘要生成完成, 耗时: {elapsed_time:.2f}秒")
        
        # 返回结果
        return jsonify({
            "summary": summary,
            "model": f"{model_type}-{model_name}",
            "elapsed_time": elapsed_time
        })
    
    except Exception as e:
        log_message(f"生成摘要时出错: {str(e)}")
        return jsonify({"error": f"生成摘要时出错: {str(e)}"}), 500

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="语音识别API服务")
    
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=5000, help="服务器端口")
    parser.add_argument("--debug", action="store_true", help="启用调试模式")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    log_message(f"启动API服务: http://{args.host}:{args.port}")
    
    app.run(host=args.host, port=args.port, debug=args.debug) 