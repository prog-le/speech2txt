import os
import sys
import torch
import whisper
import glob
import time
import json
from datetime import timedelta

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, 
                              QComboBox, QProgressBar, QTabWidget, QGroupBox, QListWidget,
                              QMessageBox, QCheckBox, QRadioButton, QButtonGroup, QSplitter, QDialog, QProgressDialog)
from PySide6.QtCore import Qt, Signal, QObject, Slot, QThread
from PySide6.QtGui import QFont, QTextCursor, QIntValidator

# 导入AI摘要模块
from ai_summary import AIModelManager

# 导入FunASR模块
try:
    from funasr_asr import FunASRModel
    FUNASR_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入FunASR模块: {e}")
    print("FunASR功能将不可用。请安装所需依赖:")
    print("pip install funasr")
    FUNASR_AVAILABLE = False

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 自定义信号类
class WorkerSignals(QObject):
    """工作线程信号"""
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(object)
    error = Signal(str)
    log = Signal(str)

# 转录工作线程
class TranscriptionWorker(QThread):
    def __init__(self, audio_file, output_file, model_type, model_settings, parent=None):
        super().__init__(parent)
        self.audio_file = audio_file
        self.output_file = output_file
        self.model_type = model_type
        self.model_settings = model_settings
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            self.signals.status.emit("正在加载模型...")
            self.signals.progress.emit(10)
            self.signals.log.emit(f"开始转录: {self.audio_file}")
            
            if self.model_type == "Whisper":
                # 使用Whisper模型
                model_size = self.model_settings.get("size", "base")
                self.signals.log.emit(f"加载Whisper {model_size}模型...")
                
                model = whisper.load_model(model_size)
                
                self.signals.status.emit("正在转录...")
                self.signals.progress.emit(30)
                
                result = model.transcribe(self.audio_file, language="zh")
                text = result["text"]
                
            # elif self.model_type == "ModelScope":
            #     # 使用ModelScope模型
            #     model_id = self.model_settings.get("model_id", "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn")
            #     device = self.model_settings.get("device", "cpu")
            #     
            #     self.signals.log.emit(f"加载ModelScope模型: {model_id}...")
            #     
            #     asr_model = ModelScopeASR(model_id=model_id, device=device)
            #     
            #     self.signals.status.emit("正在转录...")
            #     self.signals.progress.emit(30)
            #     
            #     text = asr_model.transcribe(self.audio_file)
            
            else:  # FunASR
                # 使用FunASR模型
                model_name = self.model_settings.get("model", "paraformer-zh")
                model_revision = self.model_settings.get("revision", "v2.0.4")
                use_vad = self.model_settings.get("use_vad", True)
                use_punc = self.model_settings.get("use_punc", True)
                use_spk = self.model_settings.get("use_spk", False)
                model_hub = self.model_settings.get("hub", "ms")
                
                self.signals.log.emit(f"加载FunASR模型: {model_name}...")
                
                asr_model = FunASRModel(
                    model=model_name,
                    model_revision=model_revision,
                    use_vad=use_vad,
                    use_punc=use_punc,
                    use_spk=use_spk,
                    model_hub=model_hub
                )
                
                self.signals.status.emit("正在转录...")
                self.signals.progress.emit(30)
                
                text = asr_model.transcribe(self.audio_file)
            
            self.signals.progress.emit(80)
            
            # 保存转录结果
            with open(self.output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            self.signals.progress.emit(100)
            self.signals.status.emit("转录完成")
            self.signals.finished.emit(text)
            self.signals.log.emit(f"转录完成: {self.output_file}")
            
        except Exception as e:
            import traceback
            error_msg = f"转录过程中出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(traceback.format_exc())

# 批量转录工作线程
class BatchTranscriptionWorker(QThread):
    def __init__(self, files, output_dir, model_type, model_settings, parent=None):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.model_type = model_type
        self.model_settings = model_settings
        self.signals = WorkerSignals()
        self.is_cancelled = False
        
    def run(self):
        try:
            total_files = len(self.files)
            self.signals.log.emit(f"开始批量转录 {total_files} 个文件")
            
            # 加载模型
            self.signals.status.emit("正在加载模型...")
            self.signals.progress.emit(0)
            
            if self.model_type == "Whisper":
                # 使用Whisper模型
                model_size = self.model_settings.get("size", "base")
                self.signals.log.emit(f"加载Whisper {model_size}模型...")
                
                model = whisper.load_model(model_size)
                
                # 处理每个文件
                for i, audio_file in enumerate(self.files):
                    if self.is_cancelled:
                        break
                    
                    # 计算进度
                    progress = int((i / total_files) * 100)
                    self.signals.progress.emit(progress)
                    
                    # 设置状态
                    file_name = os.path.basename(audio_file)
                    self.signals.status.emit(f"正在转录 ({i+1}/{total_files}): {file_name}")
                    self.signals.log.emit(f"开始转录: {audio_file}")
                    
                    try:
                        # 转录
                        result = model.transcribe(audio_file, language="zh")
                        text = result["text"]
                        
                        # 保存结果
                        base_name = os.path.splitext(os.path.basename(audio_file))[0]
                        output_file = os.path.join(self.output_dir, f"{base_name}.txt")
                        
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(text)
                        
                        self.signals.log.emit(f"转录完成: {output_file}")
                    except Exception as e:
                        self.signals.log.emit(f"转录文件 {audio_file} 时出错: {str(e)}")
                
            # elif self.model_type == "ModelScope":
            #     # 使用ModelScope模型
            #     model_id = self.model_settings.get("model_id", "damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn")
            #     device = self.model_settings.get("device", "cpu")
            #     
            #     self.signals.log.emit(f"加载ModelScope模型: {model_id}...")
            #     
            #     asr_model = ModelScopeASR(model_id=model_id, device=device)
            #     
            #     # 处理每个文件
            #     for i, audio_file in enumerate(self.files):
            #         if self.is_cancelled:
            #             break
            #         
            #         # 计算进度
            #         progress = int((i / total_files) * 100)
            #         self.signals.progress.emit(progress)
            #         
            #         # 设置状态
            #         file_name = os.path.basename(audio_file)
            #         self.signals.status.emit(f"正在转录 ({i+1}/{total_files}): {file_name}")
            #         
            #         try:
            #             # 转录
            #             text = asr_model.transcribe(audio_file)
            #             
            #             # 保存结果
            #             base_name = os.path.splitext(os.path.basename(audio_file))[0]
            #             output_file = os.path.join(self.output_dir, f"{base_name}.txt")
            #             
            #             with open(output_file, "w", encoding="utf-8") as f:
            #                 f.write(text)
            #             
            #             self.signals.log.emit(f"转录完成: {output_file}")
            #         except Exception as e:
            #             self.signals.log.emit(f"转录文件 {audio_file} 时出错: {str(e)}")
            
            else:  # FunASR
                # 使用FunASR模型
                model_name = self.model_settings.get("model", "paraformer-zh")
                model_revision = self.model_settings.get("revision", "v2.0.4")
                use_vad = self.model_settings.get("use_vad", True)
                use_punc = self.model_settings.get("use_punc", True)
                use_spk = self.model_settings.get("use_spk", False)
                model_hub = self.model_settings.get("hub", "ms")
                
                self.signals.log.emit(f"加载FunASR模型: {model_name}...")
                
                asr_model = FunASRModel(
                    model=model_name,
                    model_revision=model_revision,
                    use_vad=use_vad,
                    use_punc=use_punc,
                    use_spk=use_spk,
                    model_hub=model_hub
                )
                
                # 处理每个文件
                for i, audio_file in enumerate(self.files):
                    if self.is_cancelled:
                        break
                    
                    # 计算进度
                    progress = int((i / total_files) * 100)
                    self.signals.progress.emit(progress)
                    
                    # 设置状态
                    file_name = os.path.basename(audio_file)
                    self.signals.status.emit(f"正在转录 ({i+1}/{total_files}): {file_name}")
                    self.signals.log.emit(f"开始转录: {audio_file}")
                    
                    try:
                        # 转录
                        text = asr_model.transcribe(audio_file)
                        
                        # 保存结果
                        base_name = os.path.splitext(os.path.basename(audio_file))[0]
                        output_file = os.path.join(self.output_dir, f"{base_name}.txt")
                        
                        with open(output_file, "w", encoding="utf-8") as f:
                            f.write(text)
                        
                        self.signals.log.emit(f"转录完成: {output_file}")
                    except Exception as e:
                        self.signals.log.emit(f"转录文件 {audio_file} 时出错: {str(e)}")
            
            # 完成
            if not self.is_cancelled:
                self.signals.progress.emit(100)
                self.signals.status.emit("批量转录完成")
                self.signals.finished.emit("完成")
            
        except Exception as e:
            import traceback
            error_msg = f"批量转录过程中出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(traceback.format_exc())
    
    def cancel(self):
        self.is_cancelled = True

# 添加AI摘要工作线程
class AISummaryWorker(QThread):
    def __init__(self, text, ai_manager, custom_prompt=None, parent=None):
        super().__init__(parent)
        self.text = text
        self.ai_manager = ai_manager
        self.custom_prompt = custom_prompt
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            self.signals.status.emit("正在生成AI摘要...")
            self.signals.progress.emit(10)
            
            # 调用AI模型生成摘要
            self.signals.log.emit(f"使用{self.ai_manager.platform}平台的{self.ai_manager.model}模型生成摘要")
            self.signals.log.emit(f"API URL: {self.ai_manager.api_url}")
            
            result = self.ai_manager.get_summary(self.text, self.custom_prompt)
            
            self.signals.progress.emit(90)
            
            if "error" in result:
                error_msg = result["error"]
                self.signals.error.emit(f"摘要生成失败: {error_msg}")
                self.signals.log.emit(f"摘要生成失败: {error_msg}")
                return
                
            summary = result["summary"]
            self.signals.progress.emit(100)
            self.signals.status.emit("摘要生成完成!")
            self.signals.finished.emit(summary)
            self.signals.log.emit("摘要生成完成")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            error_msg = f"摘要生成过程中出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(error_trace)

# 添加获取模型的工作线程
class GetModelsWorker(QObject):
    finished = Signal(list)  # 发送模型列表
    error = Signal(str)      # 发送错误信息
    
    def __init__(self, ai_manager, ollama_url):
        super().__init__()
        self.ai_manager = ai_manager
        self.ollama_url = ollama_url
        
    def run(self):
        try:
            models = self.ai_manager.get_ollama_models(self.ollama_url)
            self.finished.emit(models)
        except Exception as e:
            self.error.emit(str(e))

# 添加摘要生成工作线程
class SummaryWorker(QThread):
    def __init__(self, text, model_type, model_settings, summary_length, summary_lang, parent=None):
        super().__init__(parent)
        self.text = text
        self.model_type = model_type
        self.model_settings = model_settings
        self.summary_length = summary_length
        self.summary_lang = summary_lang
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            self.signals.status.emit("正在生成摘要...")
            self.signals.log.emit("开始生成摘要")
            
            # 创建AI模型管理器
            from ai_summary import AIModelManager
            
            if self.model_type == "Ollama":
                ai_manager = AIModelManager(
                    model_type="ollama",
                    model_name=self.model_settings.get("model", "llama3"),
                    api_url=self.model_settings.get("api_url", "http://localhost:11434")
                )
            else:  # 自定义
                ai_manager = AIModelManager(
                    model_type="custom",
                    model_name=self.model_settings.get("model", ""),
                    api_url=self.model_settings.get("api_url", ""),
                    api_key=self.model_settings.get("api_key", "")
                )
            
            # 生成摘要
            summary = ai_manager.generate_summary(
                self.text, 
                self.summary_length, 
                self.summary_lang
            )
            
            self.signals.finished.emit(summary)
            self.signals.log.emit("摘要生成完成")
            
        except Exception as e:
            import traceback
            error_msg = f"生成摘要时出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(traceback.format_exc())

# 主窗口类
class MainWindow(QMainWindow):  # 继承QMainWindow
    def __init__(self):
        super().__init__()
        
        # 初始化属性
        self.init_attributes()
        
        # 初始化UI
        self.init_ui()
        
        # 初始日志
        self.log("应用程序已启动")
        
        # 延迟初始化
        QApplication.processEvents()
        self.delayed_init()
    
    def init_attributes(self):
        # 初始化变量
        self.transcription_worker = None
        self.batch_worker = None
        self.summary_worker = None
        
        # 初始化AI模型管理器
        self.ai_manager = AIModelManager()
    
    def init_ui(self):
        """初始化UI"""
        # 设置窗口标题和大小
        self.setWindowTitle("语音转文本工具")
        self.resize(800, 600)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建标签页
        self.tabs = QTabWidget()
        self.single_tab = QWidget()
        self.batch_tab = QWidget()
        self.settings_tab = QWidget()
        self.ai_tab = QWidget()
        self.log_tab = QWidget()
        
        self.tabs.addTab(self.single_tab, "单文件转录")
        self.tabs.addTab(self.batch_tab, "批量转录")
        self.tabs.addTab(self.settings_tab, "设置")
        self.tabs.addTab(self.ai_tab, "AI摘要")
        self.tabs.addTab(self.log_tab, "日志")
        
        main_layout.addWidget(self.tabs)
        
        # 首先设置日志标签页，因为其他标签页可能会调用log方法
        self.setup_log_tab()
        
        # 然后设置其他标签页
        self.setup_single_tab()
        self.setup_batch_tab()
        self.setup_settings_tab()
        self.setup_ai_tab()
    
    def delayed_init(self):
        """延迟初始化，在所有UI元素都初始化完成后执行"""
        # 刷新Ollama模型列表
        try:
            self.refresh_ollama_models()
        except Exception as e:
            self.log(f"刷新Ollama模型列表时出错: {str(e)}")

    def setup_single_tab(self):
        """设置单文件标签页"""
        layout = QVBoxLayout(self.single_tab)
        
        # 文件选择组
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 音频文件选择
        audio_file_layout = QHBoxLayout()
        audio_file_label = QLabel("音频文件:")
        self.audio_file_path = QLineEdit()
        self.audio_file_path.setReadOnly(True)
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_audio_file)
        
        audio_file_layout.addWidget(audio_file_label)
        audio_file_layout.addWidget(self.audio_file_path)
        audio_file_layout.addWidget(browse_button)
        file_layout.addLayout(audio_file_layout)
        
        # 输出文件选择
        output_file_layout = QHBoxLayout()
        output_file_label = QLabel("输出文件:")
        self.output_file_path = QLineEdit()
        self.use_default_output = QCheckBox("使用默认输出路径")
        self.use_default_output.setChecked(True)
        self.output_file_path.setEnabled(False)
        browse_output_button = QPushButton("浏览...")
        browse_output_button.clicked.connect(self.browse_output_file)
        browse_output_button.setEnabled(False)
        
        # 连接复选框信号
        self.use_default_output.stateChanged.connect(self.toggle_output_path)
        
        output_file_layout.addWidget(output_file_label)
        output_file_layout.addWidget(self.output_file_path)
        output_file_layout.addWidget(self.use_default_output)
        output_file_layout.addWidget(browse_output_button)
        file_layout.addLayout(output_file_layout)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QHBoxLayout()
        
        model_label = QLabel("选择模型大小:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.model_combo.setCurrentText("base")
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 进度显示
        progress_group = QGroupBox("转换进度")
        progress_layout = QVBoxLayout()
        
        self.single_progress_bar = QProgressBar()
        self.single_progress_bar.setRange(0, 100)
        self.single_progress_bar.setValue(0)
        
        self.single_status_label = QLabel("就绪")
        
        progress_layout.addWidget(self.single_progress_bar)
        progress_layout.addWidget(self.single_status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # 预览
        preview_group = QGroupBox("转换结果预览")
        preview_layout = QVBoxLayout()
        
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        
        preview_layout.addWidget(self.preview_text)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        # 操作按钮
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("开始转换")
        self.start_button.clicked.connect(self.start_transcription)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_single_transcription)
        self.cancel_button.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def toggle_output_path(self, state):
        """切换输出路径启用状态"""
        use_default = (state == Qt.Checked)
        self.output_file_path.setEnabled(not use_default)
        
        # 同时启用/禁用浏览按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "浏览..." and button != self.findChild(QPushButton, "", 1):  # 跳过第一个浏览按钮
                button.setEnabled(not use_default)
        
        # 如果选择使用默认路径，则根据音频文件路径生成默认输出路径
        if use_default and self.audio_file_path.text():
            audio_path = self.audio_file_path.text()
            default_output = os.path.splitext(audio_path)[0] + ".txt"
            self.output_file_path.setText(default_output)

    def browse_audio_file(self):
        """浏览并选择音频文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg);;所有文件 (*.*)"
        )
        
        if file_path:
            self.audio_file_path.setText(file_path)
            self.log(f"已选择音频文件: {file_path}")
            
            # 如果使用默认输出路径，则自动生成输出文件路径
            if self.use_default_output.isChecked():
                default_output = os.path.splitext(file_path)[0] + ".txt"
                self.output_file_path.setText(default_output)
    
    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "选择输出文件", 
            "", 
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if file_path:
            self.output_file_path.setText(file_path)
    
    def browse_audio_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择音频文件目录"
        )
        if dir_path:
            self.batch_dir_path.setText(dir_path)
    
    def browse_output_dir(self):
        """浏览输出目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输出目录",
            ""
        )
        
        if dir_path:
            self.batch_output_dir.setText(dir_path)
    
    def scan_files(self):
        directory = self.batch_dir_path.text().strip()
        if not directory:
            QMessageBox.warning(self, "警告", "请先选择音频文件目录")
            return
        
        if not os.path.isdir(directory):
            QMessageBox.warning(self, "警告", f"目录 '{directory}' 不存在")
            return
        
        pattern = self.pattern_input.text().strip()
        if not pattern:
            pattern = "*.mp3 *.wav *.m4a *.flac *.ogg"
        
        self.files_list.clear()
        
        all_files = []
        for p in pattern.split():
            files = glob.glob(os.path.join(directory, p))
            all_files.extend(files)
        
        if not all_files:
            QMessageBox.information(self, "信息", f"在目录 '{directory}' 中没有找到匹配 '{pattern}' 的文件")
            return
        
        for file_path in all_files:
            self.files_list.addItem(os.path.basename(file_path))
        
        self.log(f"找到 {len(all_files)} 个匹配的文件")
    
    def start_transcription(self):
        """开始转录"""
        # 获取音频文件路径
        audio_file = self.audio_file_path.text().strip()
        if not audio_file:
            QMessageBox.warning(self, "警告", "请选择音频文件")
            return
        
        # 获取输出文件路径
        output_file = self.output_file_path.text().strip()
        if not output_file:
            QMessageBox.warning(self, "警告", "请指定输出文件")
            return
        
        # 获取模型设置
        model_type = self.model_combo.currentText()
        model_settings = {}
        
        if model_type == "Whisper":
            model_settings["size"] = self.model_combo.currentText()
        elif model_type == "FunASR":
            model_settings["model"] = self.funasr_model_combo.currentText()
            model_settings["revision"] = self.funasr_revision_input.text().strip()
            model_settings["use_vad"] = self.funasr_vad_check.isChecked()
            model_settings["use_punc"] = self.funasr_punc_check.isChecked()
            model_settings["use_spk"] = self.funasr_spk_check.isChecked()
            model_settings["hub"] = self.funasr_hub_combo.currentText()
        
        # 更新UI状态
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.single_progress_bar.setValue(0)
        self.single_status_label.setText("准备中...")
        
        # 创建并启动工作线程
        self.worker = TranscriptionWorker(audio_file, output_file, model_type, model_settings)
        
        # 连接信号
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.status.connect(self.update_status)
        self.worker.signals.finished.connect(self.on_transcription_finished)
        self.worker.signals.error.connect(self.on_error)
        self.worker.signals.log.connect(self.log)
        
        # 启动线程
        self.worker.start()
        self.log(f"开始转录: {audio_file}")
    
    def cancel_single_transcription(self):
        if self.worker and self.worker.isRunning():
            # 终止线程
            self.worker.terminate()
            self.worker.wait()
            
            # 更新UI
            self.single_status_label.setText("已取消")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            
            self.log("转录已取消")
    
    def update_progress(self, value):
        self.single_progress_bar.setValue(value)
    
    def update_status(self, status):
        self.single_status_label.setText(status)
        self.statusBar().showMessage(status)
    
    def on_transcription_finished(self, text):
        """单文件转录完成处理"""
        self.single_progress_bar.setValue(100)
        self.single_status_label.setText("转录完成")
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        
        # 获取输出文件路径
        output_file = self.output_file_path.text()
        
        # 显示转录结果
        self.preview_text.setText(text)
        
        # 询问是否生成摘要
        reply = QMessageBox.question(
            self,
            "转录完成",
            "转录已完成，是否要生成摘要？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 切换到AI摘要标签页
            self.tabs.setCurrentWidget(self.ai_tab)
            
            # 加载转录文本
            self.preview_text.setText(text)
    
    def on_error(self, error_message):
        QMessageBox.critical(self, "错误", error_message)
        
        # 恢复按钮状态
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
    
    def log(self, message):
        """记录日志"""
        if hasattr(self, 'log_text'):
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_entry = f"[{timestamp}] {message}"
            
            self.log_text.append(log_entry)
            self.log_text.moveCursor(QTextCursor.End)
        else:
            # 如果log_text尚未初始化，则将消息打印到控制台
            print(f"[LOG] {message}")

    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        self.log("日志已清除")

    def browse_text_file(self):
        """浏览文本文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择文本文件",
            "",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )
        
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                self.preview_text.setText(text)
                self.log(f"已加载文本文件: {file_path}")
            
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载文本文件时出错: {str(e)}")
                self.log(f"加载文本文件时出错: {str(e)}")

    def load_text_content(self):
        """加载文本文件内容"""
        file_path = self.text_file_path.text().strip()
        if not file_path:
            QMessageBox.warning(self, "警告", "请先选择文本文件")
            return
        
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "警告", f"文件 '{file_path}' 不存在")
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            self.original_text.setText(content)
            self.log(f"已加载文本文件: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载文件时出错: {str(e)}")
            self.log(f"加载文件时出错: {str(e)}")

    def on_ai_model_type_changed(self, model_type):
        """AI模型类型变更处理"""
        if model_type == "Ollama":
            # 显示Ollama设置
            self.ollama_settings.setVisible(True)
            self.custom_api_settings.setVisible(False)
        elif model_type == "自定义":
            # 显示自定义API设置
            self.ollama_settings.setVisible(False)
            self.custom_api_settings.setVisible(True)
        
        self.log(f"已切换到{model_type}模型")

    def start_batch_transcription(self):
        """开始批量转录"""
        # 获取输入目录
        input_dir = self.input_dir_path.text().strip()
        if not input_dir:
            QMessageBox.warning(self, "警告", "请选择输入目录")
            return
        
        # 获取输出目录
        output_dir = self.output_dir_path.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "警告", "请指定输出目录")
            return
        
        # 创建输出目录
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                QMessageBox.critical(self, "错误", f"创建输出目录失败: {str(e)}")
                return
        
        # 获取文件列表
        files = self.batch_files
        if not files:
            QMessageBox.warning(self, "警告", "没有找到音频文件")
            return
        
        # 获取模型设置
        model_type = self.model_type_combo.currentText()
        model_settings = {}
        
        if model_type == "Whisper":
            model_settings["size"] = self.whisper_size_combo.currentText()
        elif model_type == "FunASR":
            model_settings["model"] = self.funasr_model_combo.currentText()
            model_settings["revision"] = self.funasr_revision_input.text().strip()
            model_settings["use_vad"] = self.funasr_vad_check.isChecked()
            model_settings["use_punc"] = self.funasr_punc_check.isChecked()
            model_settings["use_spk"] = self.funasr_spk_check.isChecked()
            model_settings["hub"] = self.funasr_hub_combo.currentText()
        
        # 更新UI状态
        self.start_batch_button.setEnabled(False)
        self.cancel_batch_button.setEnabled(True)
        self.batch_progress_bar.setValue(0)
        self.batch_status_label.setText("准备中...")
        
        # 创建并启动工作线程
        self.batch_worker = BatchTranscriptionWorker(files, output_dir, model_type, model_settings)
        
        # 连接信号
        self.batch_worker.signals.progress.connect(self.update_batch_progress)
        self.batch_worker.signals.status.connect(self.update_batch_status)
        self.batch_worker.signals.finished.connect(self.on_batch_transcription_finished)
        self.batch_worker.signals.error.connect(self.on_error)
        self.batch_worker.signals.log.connect(self.log)
        
        # 启动线程
        self.batch_worker.start()
        self.log(f"开始批量转录: {len(files)}个文件")

    def cancel_batch_transcription(self):
        """取消批量转录"""
        if self.batch_worker and self.batch_worker.isRunning():
            # 设置取消标志
            self.batch_worker.is_cancelled = True
            
            # 等待线程完成当前任务
            self.batch_worker.wait()
            
            # 更新UI
            self.update_status("批量转录已取消")
            self.start_batch_button.setEnabled(True)
            self.cancel_batch_button.setEnabled(False)
            
            self.log("批量转录已取消")

    def add_files(self):
        """添加音频文件到列表"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg);;所有文件 (*)"
        )
        
        if file_paths:
            for file_path in file_paths:
                # 检查是否已存在
                exists = False
                for i in range(self.files_list.count()):
                    if self.files_list.item(i).text() == file_path:
                        exists = True
                        break
                
                if not exists:
                    self.files_list.addItem(file_path)
            
            self.log(f"已添加 {len(file_paths)} 个文件到列表")

    def remove_selected_files(self):
        """从列表中移除选中的文件"""
        selected_items = self.files_list.selectedItems()
        if not selected_items:
            return
        
        for item in selected_items:
            row = self.files_list.row(item)
            self.files_list.takeItem(row)
        
        self.log(f"已从列表中移除 {len(selected_items)} 个文件")

    def clear_files(self):
        """清空文件列表"""
        if self.files_list.count() == 0:
            return
        
        reply = QMessageBox.question(
            self,
            "确认清空",
            "确定要清空文件列表吗?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.files_list.clear()
            self.log("已清空文件列表")

    def scan_directory(self):
        """扫描目录添加音频文件"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择音频文件目录",
            ""
        )
        
        if not dir_path:
            return
        
        # 获取文件模式
        file_patterns = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg"]
        
        # 扫描文件
        files = []
        for pattern in file_patterns:
            files.extend(glob.glob(os.path.join(dir_path, pattern)))
        
        if not files:
            QMessageBox.information(self, "提示", f"在目录 '{dir_path}' 中未找到音频文件")
            return
        
        # 添加到列表
        count = 0
        for file_path in files:
            # 检查是否已存在
            exists = False
            for i in range(self.files_list.count()):
                if self.files_list.item(i).text() == file_path:
                    exists = True
                    break
            
            if not exists:
                self.files_list.addItem(file_path)
                count += 1
        
        self.log(f"已从目录 '{dir_path}' 添加 {count} 个文件到列表")
        
        if count > 0:
            QMessageBox.information(self, "扫描完成", f"已添加 {count} 个音频文件到列表")

    def update_batch_progress(self, value):
        """更新批量转录进度条"""
        self.batch_progress_bar.setValue(value)

    def on_batch_transcription_finished(self, result):
        """批量转录完成回调"""
        # 恢复按钮状态
        self.start_batch_button.setEnabled(True)
        self.cancel_batch_button.setEnabled(False)
        
        # 更新进度条和状态
        self.batch_progress_bar.setValue(100)
        self.update_status("批量转录完成")
        
        self.log("批量转录完成")
        
        # 显示完成消息
        QMessageBox.information(self, "完成", "批量转录已完成")

    def test_ai_connection(self):
        """测试AI连接"""
        self.log("正在测试AI连接...")
        
        model_type = self.ai_model_type.currentText()
        
        if model_type == "Ollama":
            model = self.ollama_model_combo.currentText()
            api_url = self.ollama_url_input.text().strip()
            
            if not model:
                QMessageBox.warning(self, "警告", "请输入Ollama模型名称")
                return
            
            if not api_url:
                QMessageBox.warning(self, "警告", "请输入Ollama URL")
                return
            
            # 创建AI模型管理器
            ai_manager = AIModelManager(
                model_type="ollama",
                model_name=model,
                api_url=api_url
            )
        
        elif model_type == "自定义":
            api_url = self.custom_api_url_input.text().strip()
            api_key = self.custom_api_key_input.text().strip()
            model = self.custom_api_model_input.text().strip()
            
            if not api_url:
                QMessageBox.warning(self, "警告", "请输入API URL")
                return
            
            if not model:
                QMessageBox.warning(self, "警告", "请输入模型名称")
                return
            
            # 创建AI模型管理器
            ai_manager = AIModelManager(
                model_type="custom",
                model_name=model,
                api_url=api_url,
                api_key=api_key
            )
        
        # 测试连接
        try:
            self.log(f"正在连接到{model_type}模型...")
            result = ai_manager.test_connection()
            
            if result:
                QMessageBox.information(self, "成功", f"成功连接到{model_type}模型")
                self.log(f"成功连接到{model_type}模型")
            else:
                QMessageBox.warning(self, "失败", f"无法连接到{model_type}模型")
                self.log(f"无法连接到{model_type}模型")
        
        except Exception as e:
            QMessageBox.critical(self, "错误", f"测试连接时出错: {str(e)}")
            self.log(f"测试连接时出错: {str(e)}")

    def generate_summary(self):
        """生成摘要"""
        # 获取转录文本
        text = self.preview_text.toPlainText().strip()
        
        if not text:
            QMessageBox.warning(self, "警告", "没有可用的转录文本")
            return
        
        # 获取AI模型设置
        model_type = self.ai_model_type.currentText()
        model_settings = {}
        
        if model_type == "Ollama":
            model = self.ollama_model_combo.currentText().strip()  # 使用下拉列表
            api_url = self.ollama_url_input.text().strip()
            
            if not model:
                QMessageBox.warning(self, "警告", "请选择或输入Ollama模型名称")
                return
            
            if not api_url:
                QMessageBox.warning(self, "警告", "请输入Ollama URL")
                return
            
            model_settings = {
                "model": model,
                "api_url": api_url
            }
        
        elif model_type == "自定义":
            api_url = self.custom_api_url_input.text().strip()
            api_key = self.custom_api_key_input.text().strip()
            model = self.custom_api_model_input.text().strip()
            
            if not api_url:
                QMessageBox.warning(self, "警告", "请输入API URL")
                return
            
            if not model:
                QMessageBox.warning(self, "警告", "请输入模型名称")
                return
            
            model_settings = {
                "model": model,
                "api_url": api_url,
                "api_key": api_key
            }
        
        # 获取摘要设置
        try:
            summary_length = int(self.summary_length_input.text().strip())
        except ValueError:
            summary_length = 200
        
        summary_lang = "chinese" if self.summary_lang_combo.currentText() == "中文" else "english"
        
        # 显示进度对话框
        self.progress_dialog = QProgressDialog("正在生成摘要...", "取消", 0, 0, self)
        self.progress_dialog.setWindowTitle("生成摘要")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        
        # 创建并启动工作线程
        self.summary_worker = SummaryWorker(
            text, 
            model_type, 
            model_settings, 
            summary_length, 
            summary_lang,
            self  # 传递self作为父对象，确保线程不会被过早销毁
        )
        
        # 连接取消按钮
        self.progress_dialog.canceled.connect(self.cancel_summary)
        
        # 连接信号
        self.summary_worker.signals.status.connect(self.progress_dialog.setLabelText)
        self.summary_worker.signals.finished.connect(self.on_summary_finished)
        self.summary_worker.signals.error.connect(self.on_summary_error)
        self.summary_worker.signals.log.connect(self.log)
        
        # 启动线程
        self.summary_worker.start()
        
        # 显示进度对话框
        self.progress_dialog.exec_()

    def cancel_summary(self):
        """取消摘要生成"""
        if hasattr(self, 'summary_worker') and self.summary_worker and self.summary_worker.isRunning():
            # 终止线程
            self.summary_worker.terminate()
            self.summary_worker.wait()
            
            # 更新UI
            self.log("摘要生成已取消")

    def on_summary_finished(self, summary):
        """摘要生成完成处理"""
        # 关闭进度对话框
        self.progress_dialog.close()
        
        if summary:
            # 创建摘要对话框
            summary_dialog = QDialog(self)
            summary_dialog.setWindowTitle("文本摘要")
            summary_dialog.resize(600, 400)
            
            dialog_layout = QVBoxLayout(summary_dialog)
            
            summary_text = QTextEdit()
            summary_text.setReadOnly(True)
            summary_text.setText(summary)
            
            save_button = QPushButton("保存摘要")
            
            dialog_layout.addWidget(summary_text)
            dialog_layout.addWidget(save_button)
            
            # 保存摘要
            def save_summary():
                file_path, _ = QFileDialog.getSaveFileName(
                    summary_dialog,
                    "保存摘要",
                    "",
                    "文本文件 (*.txt)"
                )
                
                if file_path:
                    try:
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(summary)
                    
                        QMessageBox.information(summary_dialog, "成功", f"摘要已保存到: {file_path}")
                    except Exception as e:
                        QMessageBox.critical(summary_dialog, "错误", f"保存摘要时出错: {str(e)}")
            
            save_button.clicked.connect(save_summary)
            
            summary_dialog.exec_()
            
            self.log("摘要生成成功")
        else:
            QMessageBox.warning(self, "失败", "生成摘要失败")
            self.log("生成摘要失败")

    def on_summary_error(self, error_msg):
        """摘要生成错误处理"""
        # 关闭进度对话框
        self.progress_dialog.close()
        
        QMessageBox.critical(self, "错误", error_msg)

    def setup_log_tab(self):
        """设置日志标签页"""
        # 创建布局
        layout = QVBoxLayout(self.log_tab)
        
        # 日志文本框
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        layout.addWidget(self.log_text)
        
        # 清除按钮
        clear_button = QPushButton("清除日志")
        clear_button.clicked.connect(self.clear_log)
        
        layout.addWidget(clear_button)

    def apply_styles(self):
        """应用样式"""
        # 设置字体
        app_font = QFont()
        app_font.setPointSize(10)
        QApplication.setFont(app_font)
        
        # 设置按钮样式
        button_style = """
        QPushButton {
            background-color: #4a86e8;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #3a76d8;
        }
        QPushButton:pressed {
            background-color: #2a66c8;
        }
        QPushButton:disabled {
            background-color: #cccccc;
            color: #888888;
        }
        """
        
        # 设置进度条样式
        progress_style = """
        QProgressBar {
            border: 1px solid #cccccc;
            border-radius: 5px;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: #4a86e8;
            border-radius: 5px;
        }
        """
        
        # 应用样式
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        # 确保进度条已经创建
        if hasattr(self, 'single_progress_bar'):
            self.single_progress_bar.setStyleSheet(progress_style)
        
        if hasattr(self, 'batch_progress_bar'):
            self.batch_progress_bar.setStyleSheet(progress_style)

    def closeEvent(self, event):
        """窗口关闭事件处理"""
        # 停止所有线程
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        
        if hasattr(self, 'batch_worker') and self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.terminate()
            self.batch_worker.wait()
        
        if hasattr(self, 'summary_worker') and self.summary_worker and self.summary_worker.isRunning():
            self.summary_worker.terminate()
            self.summary_worker.wait()
        
        # 接受关闭事件
        event.accept()

    def setup_batch_tab(self):
        """设置批量标签页"""
        layout = QVBoxLayout(self.batch_tab)
        
        # 目录选择组
        dir_group = QGroupBox("目录选择")
        dir_layout = QVBoxLayout()
        
        # 输入目录选择
        input_dir_layout = QHBoxLayout()
        input_dir_label = QLabel("输入目录:")
        self.input_dir_path = QLineEdit()
        self.input_dir_path.setReadOnly(True)
        browse_input_button = QPushButton("浏览...")
        browse_input_button.clicked.connect(self.browse_input_dir)
        
        input_dir_layout.addWidget(input_dir_label)
        input_dir_layout.addWidget(self.input_dir_path)
        input_dir_layout.addWidget(browse_input_button)
        dir_layout.addLayout(input_dir_layout)
        
        # 输出目录选择
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("输出目录:")
        self.output_dir_path = QLineEdit()
        self.use_default_output_dir = QCheckBox("使用默认输出目录")
        self.use_default_output_dir.setChecked(True)
        self.output_dir_path.setEnabled(False)
        browse_output_button = QPushButton("浏览...")
        browse_output_button.clicked.connect(self.browse_output_dir)
        browse_output_button.setEnabled(False)
        
        # 连接复选框信号
        self.use_default_output_dir.stateChanged.connect(self.toggle_output_dir)
        
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.output_dir_path)
        output_dir_layout.addWidget(self.use_default_output_dir)
        output_dir_layout.addWidget(browse_output_button)
        dir_layout.addLayout(output_dir_layout)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # ... (其余部分保持不变)

    def toggle_output_dir(self, state):
        """切换输出目录启用状态"""
        use_default = (state == Qt.Checked)
        self.output_dir_path.setEnabled(not use_default)
        
        # 同时启用/禁用浏览按钮
        for button in self.findChildren(QPushButton):
            if button.text() == "浏览..." and button != self.findChild(QPushButton, "", 1) and button != self.findChild(QPushButton, "", 2):
                button.setEnabled(not use_default)
        
        # 如果选择使用默认路径，则根据输入目录路径生成默认输出路径
        if use_default and self.input_dir_path.text():
            input_dir = self.input_dir_path.text()
            default_output_dir = os.path.join(os.path.dirname(input_dir), os.path.basename(input_dir) + "_output")
            self.output_dir_path.setText(default_output_dir)

    def browse_input_dir(self):
        """浏览并选择输入目录"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "选择输入目录",
            ""
        )
        
        if dir_path:
            self.input_dir_path.setText(dir_path)
            self.log(f"已选择输入目录: {dir_path}")
            
            # 如果使用默认输出目录，则自动生成输出目录路径
            if self.use_default_output_dir.isChecked():
                default_output_dir = os.path.join(os.path.dirname(dir_path), os.path.basename(dir_path) + "_output")
                self.output_dir_path.setText(default_output_dir)
            
            # 更新文件列表
            self.update_batch_file_list()

    def setup_ai_tab(self):
        """设置AI摘要标签页"""
        layout = QVBoxLayout(self.ai_tab)
        
        # 模型设置组
        model_group = QGroupBox("AI模型设置")
        model_layout = QVBoxLayout()
        
        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("模型类型:")
        self.ai_model_type = QComboBox()
        self.ai_model_type.addItems(["Ollama", "自定义"])
        
        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.ai_model_type)
        model_layout.addLayout(model_type_layout)
        
        # Ollama设置
        self.ollama_settings = QWidget()
        ollama_layout = QVBoxLayout(self.ollama_settings)
        
        # Ollama URL
        ollama_url_layout = QHBoxLayout()
        ollama_url_label = QLabel("Ollama URL:")
        self.ollama_url_input = QLineEdit("http://localhost:11434")
        
        ollama_url_layout.addWidget(ollama_url_label)
        ollama_url_layout.addWidget(self.ollama_url_input)
        ollama_layout.addLayout(ollama_url_layout)
        
        # Ollama模型
        ollama_model_layout = QHBoxLayout()
        ollama_model_label = QLabel("模型名称:")
        self.ollama_model_combo = QComboBox()  # 使用下拉列表代替输入框
        self.ollama_model_combo.setEditable(True)  # 允许编辑
        self.ollama_model_combo.addItem("llama3")  # 默认添加llama3
        
        # 刷新模型列表按钮
        refresh_models_button = QPushButton("刷新模型列表")
        refresh_models_button.clicked.connect(self.refresh_ollama_models)
        
        ollama_model_layout.addWidget(ollama_model_label)
        ollama_model_layout.addWidget(self.ollama_model_combo)
        ollama_model_layout.addWidget(refresh_models_button)
        ollama_layout.addLayout(ollama_model_layout)
        
        model_layout.addWidget(self.ollama_settings)
        
        # 自定义API设置
        self.custom_api_settings = QWidget()
        custom_api_layout = QVBoxLayout(self.custom_api_settings)
        
        # API URL
        custom_api_url_layout = QHBoxLayout()
        custom_api_url_label = QLabel("API URL:")
        self.custom_api_url_input = QLineEdit()
        
        custom_api_url_layout.addWidget(custom_api_url_label)
        custom_api_url_layout.addWidget(self.custom_api_url_input)
        custom_api_layout.addLayout(custom_api_url_layout)
        
        # API密钥
        custom_api_key_layout = QHBoxLayout()
        custom_api_key_label = QLabel("API密钥:")
        self.custom_api_key_input = QLineEdit()
        self.custom_api_key_input.setEchoMode(QLineEdit.Password)
        
        custom_api_key_layout.addWidget(custom_api_key_label)
        custom_api_key_layout.addWidget(self.custom_api_key_input)
        custom_api_layout.addLayout(custom_api_key_layout)
        
        # 模型名称
        custom_api_model_layout = QHBoxLayout()
        custom_api_model_label = QLabel("模型名称:")
        self.custom_api_model_input = QLineEdit()
        
        custom_api_model_layout.addWidget(custom_api_model_label)
        custom_api_model_layout.addWidget(self.custom_api_model_input)
        custom_api_layout.addLayout(custom_api_model_layout)
        
        model_layout.addWidget(self.custom_api_settings)
        
        # 初始隐藏自定义API设置
        self.custom_api_settings.setVisible(False)
        
        # 测试连接按钮
        test_button = QPushButton("测试连接")
        test_button.clicked.connect(self.test_ai_connection)
        model_layout.addWidget(test_button)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 摘要设置组
        summary_group = QGroupBox("摘要设置")
        summary_layout = QVBoxLayout()
        
        # 摘要长度
        summary_length_layout = QHBoxLayout()
        summary_length_label = QLabel("摘要长度:")
        self.summary_length_input = QLineEdit("200")
        self.summary_length_input.setValidator(QIntValidator(50, 1000))
        
        summary_length_layout.addWidget(summary_length_label)
        summary_length_layout.addWidget(self.summary_length_input)
        summary_layout.addLayout(summary_length_layout)
        
        # 摘要语言
        summary_lang_layout = QHBoxLayout()
        summary_lang_label = QLabel("摘要语言:")
        self.summary_lang_combo = QComboBox()
        self.summary_lang_combo.addItems(["中文", "英文"])
        
        summary_lang_layout.addWidget(summary_lang_label)
        summary_lang_layout.addWidget(self.summary_lang_combo)
        summary_layout.addLayout(summary_lang_layout)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # 文本区域组
        text_group = QGroupBox("文本内容")
        text_layout = QVBoxLayout()
        
        # 文本预览
        self.preview_text = QTextEdit()
        text_layout.addWidget(self.preview_text)
        
        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 加载文本按钮
        load_text_button = QPushButton("加载文本文件")
        load_text_button.clicked.connect(self.browse_text_file)
        button_layout.addWidget(load_text_button)
        
        # 生成摘要按钮
        generate_button = QPushButton("生成摘要")
        generate_button.clicked.connect(self.generate_summary)
        button_layout.addWidget(generate_button)
        
        text_layout.addLayout(button_layout)
        
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        
        # 连接信号
        self.ai_model_type.currentTextChanged.connect(self.on_ai_model_type_changed)

    def refresh_ollama_models(self):
        """刷新Ollama模型列表"""
        self.log("正在获取Ollama模型列表...")
        
        # 保存当前选择的模型
        current_model = self.ollama_model_combo.currentText()
        
        # 清空模型列表
        self.ollama_model_combo.clear()
        
        # 获取Ollama模型列表
        try:
            import requests
            api_url = self.ollama_url_input.text().strip()
            if not api_url:
                api_url = "http://localhost:11434"
            
            response = requests.get(f"{api_url}/api/tags")
            
            if response.status_code == 200:
                models_data = response.json()
                models = [model["name"] for model in models_data.get("models", [])]
                
                if models:
                    self.ollama_model_combo.addItems(models)
                    self.log(f"获取到 {len(models)} 个Ollama模型")
                    
                    # 恢复之前选择的模型
                    index = self.ollama_model_combo.findText(current_model)
                    if index >= 0:
                        self.ollama_model_combo.setCurrentIndex(index)
                
                else:
                    self.ollama_model_combo.addItem("llama3")  # 默认添加llama3
                    self.log("未找到Ollama模型，请确保Ollama服务已启动")
            else:
                self.ollama_model_combo.addItem("llama3")  # 默认添加llama3
                self.log(f"获取Ollama模型列表失败: {response.status_code}")
        
        except Exception as e:
            self.ollama_model_combo.addItem("llama3")  # 默认添加llama3
            self.log(f"获取Ollama模型列表时出错: {str(e)}")

    def setup_settings_tab(self):
        """设置设置标签页"""
        layout = QVBoxLayout(self.settings_tab)
        
        # 模型设置组
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout()
        
        # 模型类型选择
        model_type_layout = QHBoxLayout()
        model_type_label = QLabel("模型类型:")
        self.model_type_combo = QComboBox()
        
        # 检查FunASR是否可用
        if FUNASR_AVAILABLE:
            self.model_type_combo.addItems(["Whisper", "FunASR"])
        else:
            self.model_type_combo.addItems(["Whisper"])
        
        model_type_layout.addWidget(model_type_label)
        model_type_layout.addWidget(self.model_type_combo)
        model_layout.addLayout(model_type_layout)
        
        # Whisper模型设置
        self.whisper_settings = QWidget()
        whisper_layout = QVBoxLayout(self.whisper_settings)
        
        whisper_size_layout = QHBoxLayout()
        whisper_size_label = QLabel("模型大小:")
        self.whisper_size_combo = QComboBox()
        self.whisper_size_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.whisper_size_combo.setCurrentText("base")
        
        whisper_size_layout.addWidget(whisper_size_label)
        whisper_size_layout.addWidget(self.whisper_size_combo)
        whisper_layout.addLayout(whisper_size_layout)
        
        model_layout.addWidget(self.whisper_settings)
        
        # FunASR模型设置
        self.funasr_settings = QWidget()
        funasr_layout = QVBoxLayout(self.funasr_settings)
        
        # FunASR模型名称
        funasr_model_layout = QHBoxLayout()
        funasr_model_label = QLabel("模型名称:")
        self.funasr_model_combo = QComboBox()
        self.funasr_model_combo.addItems(["paraformer-zh", "paraformer-zh-streaming"])
        
        funasr_model_layout.addWidget(funasr_model_label)
        funasr_model_layout.addWidget(self.funasr_model_combo)
        funasr_layout.addLayout(funasr_model_layout)
        
        # FunASR模型版本
        funasr_revision_layout = QHBoxLayout()
        funasr_revision_label = QLabel("模型版本:")
        self.funasr_revision_input = QLineEdit("v2.0.4")
        
        funasr_revision_layout.addWidget(funasr_revision_label)
        funasr_revision_layout.addWidget(self.funasr_revision_input)
        funasr_layout.addLayout(funasr_revision_layout)
        
        # FunASR模型仓库
        funasr_hub_layout = QHBoxLayout()
        funasr_hub_label = QLabel("模型仓库:")
        self.funasr_hub_combo = QComboBox()
        self.funasr_hub_combo.addItems(["ms", "hf"])
        
        funasr_hub_layout.addWidget(funasr_hub_label)
        funasr_hub_layout.addWidget(self.funasr_hub_combo)
        funasr_layout.addLayout(funasr_hub_layout)
        
        # FunASR功能选项
        funasr_options_layout = QHBoxLayout()
        self.funasr_vad_check = QCheckBox("使用VAD")
        self.funasr_vad_check.setChecked(True)
        self.funasr_punc_check = QCheckBox("使用标点")
        self.funasr_punc_check.setChecked(True)
        self.funasr_spk_check = QCheckBox("使用说话人识别")
        self.funasr_spk_check.setChecked(False)
        
        funasr_options_layout.addWidget(self.funasr_vad_check)
        funasr_options_layout.addWidget(self.funasr_punc_check)
        funasr_options_layout.addWidget(self.funasr_spk_check)
        funasr_layout.addLayout(funasr_options_layout)
        
        model_layout.addWidget(self.funasr_settings)
        
        # 初始隐藏FunASR设置
        self.funasr_settings.setVisible(False)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 设备信息
        device_group = QGroupBox("设备信息")
        device_layout = QVBoxLayout()
        
        try:
            device_info = QLabel(f"当前设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            if torch.cuda.is_available():
                device_info.setText(f"当前设备: CUDA ({torch.cuda.get_device_name()})")
        except ImportError:
            device_info = QLabel("当前设备: 未知 (PyTorch未安装)")
        
        device_layout.addWidget(device_info)
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        # 连接信号
        self.model_type_combo.currentTextChanged.connect(self.on_model_type_changed)

    def on_model_type_changed(self, model_type):
        """模型类型变更处理"""
        if model_type == "Whisper":
            # 显示Whisper设置
            self.whisper_settings.setVisible(True)
            self.funasr_settings.setVisible(False)
        elif model_type == "FunASR":
            # 显示FunASR设置
            self.whisper_settings.setVisible(False)
            self.funasr_settings.setVisible(True)
        
        self.log(f"已切换到{model_type}模型")

# 应用程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())