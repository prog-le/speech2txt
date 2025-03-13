import os
import sys
import torch
import whisper
import glob
import time
from datetime import timedelta

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                              QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, 
                              QComboBox, QProgressBar, QTabWidget, QGroupBox, QListWidget,
                              QMessageBox)
from PySide6.QtCore import Qt, Signal, QObject, Slot, QThread
from PySide6.QtGui import QFont, QTextCursor  # 正确导入QTextCursor

# 设置环境变量
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 自定义信号类
class WorkerSignals(QObject):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal(str)
    error = Signal(str)
    log = Signal(str)

# 转录工作线程
class TranscriptionWorker(QThread):
    def __init__(self, audio_file, output_file, model_size, hf_endpoint, parent=None):
        super().__init__(parent)
        self.audio_file = audio_file
        self.output_file = output_file
        self.model_size = model_size
        self.hf_endpoint = hf_endpoint
        self.signals = WorkerSignals()
        
    def run(self):
        try:
            # 设置HuggingFace端点
            if self.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.hf_endpoint
                self.signals.log.emit(f"设置HF端点: {self.hf_endpoint}")
            
            # 规范化文件路径
            audio_file_path = os.path.abspath(os.path.normpath(self.audio_file))
            
            # 检查文件是否存在
            if not os.path.exists(audio_file_path):
                self.signals.error.emit(f"错误: 文件 '{audio_file_path}' 不存在")
                return
                
            # 设置默认输出文件路径
            if not self.output_file:
                base_name = os.path.splitext(audio_file_path)[0]
                output_file_path = f"{base_name}.txt"
            else:
                output_file_path = self.output_file
            
            # 加载Whisper模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.signals.log.emit(f"使用设备: {device}")
            
            # 加载模型
            self.signals.status.emit(f"正在加载 {self.model_size} 模型...")
            self.signals.progress.emit(10)
            
            model = whisper.load_model(self.model_size, device=device)
            
            self.signals.progress.emit(40)
            self.signals.log.emit(f"模型 {self.model_size} 加载完成")
            
            # 转录音频
            self.signals.status.emit(f"正在处理音频文件: {os.path.basename(audio_file_path)}...")
            self.signals.progress.emit(50)
            
            # 使用verbose=False来禁用Whisper内置的进度条
            result = model.transcribe(
                audio_file_path, 
                verbose=False,
                word_timestamps=True,  # 启用单词级时间戳
                fp16=torch.cuda.is_available()  # 如果有CUDA，使用FP16加速
            )
            
            self.signals.progress.emit(80)
            self.signals.log.emit("音频处理完成，正在生成时间戳...")
            
            # 获取带时间戳的文本
            segments = result["segments"]
            
            # 创建带时间戳的转录文本
            transcription_with_timestamps = ""
            raw_transcription = ""
            
            # 处理时间戳
            self.signals.status.emit("正在生成时间戳...")
            for i, segment in enumerate(segments):
                # 更新进度
                progress = 80 + int((i / len(segments)) * 15)
                self.signals.progress.emit(progress)
                
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
                
            self.signals.progress.emit(100)
            self.signals.status.emit(f"转录完成! 结果已保存到: {output_file_path}")
            self.signals.finished.emit(raw_transcription)
            self.signals.log.emit(f"转录完成! 结果已保存到: {output_file_path}")
            
        except Exception as e:
            import traceback
            error_msg = f"转录过程中出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(traceback.format_exc())

# 批量转录工作线程
class BatchTranscriptionWorker(QThread):
    def __init__(self, directory, pattern, output_dir, model_size, hf_endpoint, parent=None):
        super().__init__(parent)
        self.directory = directory
        self.pattern = pattern
        self.output_dir = output_dir
        self.model_size = model_size
        self.hf_endpoint = hf_endpoint
        self.signals = WorkerSignals()
        self.is_running = True
    
    def stop(self):
        self.is_running = False
        
    def run(self):
        try:
            # 设置HuggingFace端点
            if self.hf_endpoint:
                os.environ["HF_ENDPOINT"] = self.hf_endpoint
                self.signals.log.emit(f"设置HF端点: {self.hf_endpoint}")
            
            # 获取所有匹配的文件
            all_files = []
            for p in self.pattern.split():
                files = glob.glob(os.path.join(self.directory, p))
                all_files.extend(files)
            
            if not all_files:
                self.signals.error.emit(f"在目录 '{self.directory}' 中没有找到匹配 '{self.pattern}' 的文件")
                return
            
            self.signals.log.emit(f"找到 {len(all_files)} 个文件待处理")
            
            # 如果指定了输出目录，确保它存在
            if self.output_dir and not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            
            # 加载模型（只加载一次）
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.signals.log.emit(f"使用设备: {device}")
            
            self.signals.status.emit(f"正在加载 {self.model_size} 模型...")
            self.signals.progress.emit(5)
            
            model = whisper.load_model(self.model_size, device=device)
            
            self.signals.progress.emit(10)
            self.signals.log.emit(f"模型 {self.model_size} 加载完成")
            
            # 处理每个文件
            for i, file_path in enumerate(all_files):
                if not self.is_running:
                    self.signals.log.emit("批量处理已取消")
                    break
                    
                file_name = os.path.basename(file_path)
                self.signals.log.emit(f"处理文件 {i+1}/{len(all_files)}: {file_name}")
                
                # 设置输出路径
                if self.output_dir:
                    base_name = os.path.splitext(file_name)[0]
                    output_path = os.path.join(self.output_dir, f"{base_name}.txt")
                else:
                    base_name = os.path.splitext(file_path)[0]
                    output_path = f"{base_name}.txt"
                
                # 更新状态
                self.signals.status.emit(f"处理文件 {i+1}/{len(all_files)}: {file_name}")
                
                # 计算当前文件的进度基准
                base_progress = 10 + (i / len(all_files)) * 90
                self.signals.progress.emit(int(base_progress))
                
                try:
                    # 转录音频
                    self.signals.log.emit(f"正在处理: {file_name}")
                    
                    result = model.transcribe(
                        file_path, 
                        verbose=False,
                        word_timestamps=True,
                        fp16=torch.cuda.is_available()
                    )
                    
                    # 获取带时间戳的文本
                    segments = result["segments"]
                    
                    # 创建带时间戳的转录文本
                    transcription_with_timestamps = ""
                    
                    # 处理时间戳
                    for segment in segments:
                        # 获取开始时间并格式化为HH:MM:SS
                        start_time = str(timedelta(seconds=int(segment['start']))).split('.')[0]
                        if start_time.startswith('0:'):
                            start_time = start_time[2:]
                        if len(start_time.split(':')) == 2:
                            start_time = f"00:{start_time}"
                        
                        # 添加时间戳和文本
                        segment_text = segment['text'].strip()
                        transcription_with_timestamps += f"[{start_time}] {segment_text}\n"
                    
                    # 保存转录结果到文件
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(transcription_with_timestamps)
                        
                    self.signals.log.emit(f"文件 {file_name} 处理完成，结果已保存到: {output_path}")
                    
                except Exception as e:
                    self.signals.log.emit(f"处理文件 {file_name} 时出错: {str(e)}")
            
            if self.is_running:
                self.signals.progress.emit(100)
                self.signals.status.emit("批量转换完成!")
                self.signals.finished.emit("批量转换完成!")
                
        except Exception as e:
            import traceback
            error_msg = f"批量转录过程中出错: {str(e)}"
            self.signals.error.emit(error_msg)
            self.signals.log.emit(error_msg)
            self.signals.log.emit(traceback.format_exc())

# 主窗口类
class MainWindow(QMainWindow):  # 继承QMainWindow
    def __init__(self):
        super().__init__()
        
        # 初始化变量
        self.transcription_worker = None
        self.batch_worker = None
        
        # 初始化UI
        self.init_ui()
        
        # 记录日志
        self.log("应用程序已启动")
        self.log(f"当前设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            self.log(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    def init_ui(self):
        # 设置窗口属性
        self.setWindowTitle("语音转文本工具")
        self.setGeometry(100, 100, 800, 600)
        
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
        self.log_tab = QWidget()
        
        self.tabs.addTab(self.single_tab, "单文件转换")
        self.tabs.addTab(self.batch_tab, "批量转换")
        self.tabs.addTab(self.settings_tab, "设置")
        self.tabs.addTab(self.log_tab, "日志")
        
        main_layout.addWidget(self.tabs)
        
        # 设置各个标签页
        self.setup_single_tab()
        self.setup_batch_tab()
        self.setup_settings_tab()
        self.setup_log_tab()
        
        # 应用样式
        self.apply_styles()
        
        # 设置状态栏
        self.statusBar().showMessage("就绪")
    
    def setup_single_tab(self):
        # 创建布局
        layout = QVBoxLayout(self.single_tab)
        
        # 文件选择
        file_group = QGroupBox("文件选择")
        file_layout = QVBoxLayout()
        
        # 音频文件选择
        audio_layout = QHBoxLayout()
        audio_label = QLabel("音频文件:")
        self.single_file_path = QLineEdit()
        browse_button = QPushButton("浏览...")
        browse_button.clicked.connect(self.browse_audio_file)
        
        audio_layout.addWidget(audio_label)
        audio_layout.addWidget(self.single_file_path)
        audio_layout.addWidget(browse_button)
        
        file_layout.addLayout(audio_layout)
        
        # 输出文件选择
        output_layout = QHBoxLayout()
        output_label = QLabel("输出文件:")
        self.single_output_path = QLineEdit()
        output_browse_button = QPushButton("浏览...")
        output_browse_button.clicked.connect(self.browse_output_file)
        
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.single_output_path)
        output_layout.addWidget(output_browse_button)
        
        file_layout.addLayout(output_layout)
        
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
        self.start_button.clicked.connect(self.start_single_transcription)
        
        self.cancel_button = QPushButton("取消")
        self.cancel_button.clicked.connect(self.cancel_single_transcription)
        self.cancel_button.setEnabled(False)
        
        button_layout.addStretch()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.cancel_button)
        
        layout.addLayout(button_layout)
    
    def setup_batch_tab(self):
        # 创建布局
        layout = QVBoxLayout(self.batch_tab)
        
        # 目录选择
        dir_group = QGroupBox("目录选择")
        dir_layout = QVBoxLayout()
        
        # 音频文件目录选择
        audio_dir_layout = QHBoxLayout()
        audio_dir_label = QLabel("音频文件目录:")
        self.batch_dir_path = QLineEdit()
        dir_browse_button = QPushButton("浏览...")
        dir_browse_button.clicked.connect(self.browse_audio_dir)
        
        audio_dir_layout.addWidget(audio_dir_label)
        audio_dir_layout.addWidget(self.batch_dir_path)
        audio_dir_layout.addWidget(dir_browse_button)
        
        dir_layout.addLayout(audio_dir_layout)
        
        # 文件模式
        pattern_layout = QHBoxLayout()
        pattern_label = QLabel("文件模式:")
        self.pattern_input = QLineEdit("*.mp3 *.wav *.m4a *.flac *.ogg")
        
        pattern_layout.addWidget(pattern_label)
        pattern_layout.addWidget(self.pattern_input)
        
        dir_layout.addLayout(pattern_layout)
        
        # 输出目录选择
        output_dir_layout = QHBoxLayout()
        output_dir_label = QLabel("输出目录:")
        self.batch_output_dir = QLineEdit()
        output_dir_browse_button = QPushButton("浏览...")
        output_dir_browse_button.clicked.connect(self.browse_output_dir)
        
        output_dir_layout.addWidget(output_dir_label)
        output_dir_layout.addWidget(self.batch_output_dir)
        output_dir_layout.addWidget(output_dir_browse_button)
        
        dir_layout.addLayout(output_dir_layout)
        
        # 模型选择
        model_layout = QHBoxLayout()
        batch_model_label = QLabel("选择模型大小:")
        self.batch_model_combo = QComboBox()
        self.batch_model_combo.addItems(["tiny", "base", "small", "medium", "large"])
        self.batch_model_combo.setCurrentText("base")
        
        model_layout.addWidget(batch_model_label)
        model_layout.addWidget(self.batch_model_combo)
        
        dir_layout.addLayout(model_layout)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 进度显示
        batch_progress_group = QGroupBox("批量转换进度")
        batch_progress_layout = QVBoxLayout()
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setRange(0, 100)
        self.batch_progress_bar.setValue(0)
        
        self.batch_status_label = QLabel("就绪")
        
        batch_progress_layout.addWidget(self.batch_progress_bar)
        batch_progress_layout.addWidget(self.batch_status_label)
        
        batch_progress_group.setLayout(batch_progress_layout)
        layout.addWidget(batch_progress_group)
        
        # 文件列表
        files_group = QGroupBox("文件列表")
        files_layout = QVBoxLayout()
        
        self.files_list = QListWidget()
        
        files_layout.addWidget(self.files_list)
        
        files_group.setLayout(files_layout)
        layout.addWidget(files_group)
        
        # 操作按钮
        batch_button_layout = QHBoxLayout()
        
        self.scan_button = QPushButton("扫描文件")
        self.scan_button.clicked.connect(self.scan_files)
        
        self.batch_start_button = QPushButton("开始批量转换")
        self.batch_start_button.clicked.connect(self.start_batch_transcription)
        
        self.batch_cancel_button = QPushButton("取消")
        self.batch_cancel_button.clicked.connect(self.cancel_batch_transcription)
        self.batch_cancel_button.setEnabled(False)
        
        batch_button_layout.addWidget(self.scan_button)
        batch_button_layout.addStretch()
        batch_button_layout.addWidget(self.batch_start_button)
        batch_button_layout.addWidget(self.batch_cancel_button)
        
        layout.addLayout(batch_button_layout)
    
    def setup_settings_tab(self):
        # 创建布局
        layout = QVBoxLayout(self.settings_tab)
        
        # HuggingFace端点设置
        hf_group = QGroupBox("HuggingFace端点设置")
        hf_layout = QVBoxLayout()
        
        hf_label = QLabel("选择或输入HuggingFace端点:")
        self.hf_combo = QComboBox()
        self.hf_combo.setEditable(True)
        self.hf_combo.addItems([
            "https://huggingface.co",
            "https://hf-mirror.com"
        ])
        
        hf_layout.addWidget(hf_label)
        hf_layout.addWidget(self.hf_combo)
        
        hf_group.setLayout(hf_layout)
        layout.addWidget(hf_group)
        
        # 其他设置
        other_group = QGroupBox("其他设置")
        other_layout = QVBoxLayout()
        
        device_info = QLabel(f"当前设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            device_info.setText(f"当前设备: CUDA ({torch.cuda.get_device_name()})")
        
        other_layout.addWidget(device_info)
        
        other_group.setLayout(other_layout)
        layout.addWidget(other_group)
        
        # 关于信息
        about_group = QGroupBox("关于")
        about_layout = QVBoxLayout()
        
        about_text = QLabel("语音转文本工具 - 基于Whisper模型")
        about_text.setAlignment(Qt.AlignCenter)
        
        version_text = QLabel("版本: 1.0.0")
        version_text.setAlignment(Qt.AlignCenter)
        
        about_layout.addWidget(about_text)
        about_layout.addWidget(version_text)
        
        about_group.setLayout(about_layout)
        layout.addWidget(about_group)
        
        # 添加空白区域
        layout.addStretch()
    
    def setup_log_tab(self):
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
        
        self.single_progress_bar.setStyleSheet(progress_style)
        self.batch_progress_bar.setStyleSheet(progress_style)
    
    def browse_audio_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择音频文件", 
            "", 
            "音频文件 (*.mp3 *.wav *.m4a *.flac *.ogg);;所有文件 (*)"
        )
        if file_path:
            self.single_file_path.setText(file_path)
            
            # 自动设置输出文件路径
            base_name = os.path.splitext(file_path)[0]
            self.single_output_path.setText(f"{base_name}.txt")
    
    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "选择输出文件", 
            "", 
            "文本文件 (*.txt);;所有文件 (*)"
        )
        if file_path:
            self.single_output_path.setText(file_path)
    
    def browse_audio_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择音频文件目录"
        )
        if dir_path:
            self.batch_dir_path.setText(dir_path)
    
    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self, 
            "选择输出目录"
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
    
    def start_single_transcription(self):
        audio_file = self.single_file_path.text().strip()
        if not audio_file:
            QMessageBox.warning(self, "警告", "请先选择音频文件")
            return
        
        if not os.path.exists(audio_file):
            QMessageBox.warning(self, "警告", f"文件 '{audio_file}' 不存在")
            return
        
        output_file = self.single_output_path.text().strip()
        model_size = self.model_combo.currentText()
        hf_endpoint = self.hf_combo.currentText() if self.hf_combo.currentText() else None
        
        # 禁用开始按钮，启用取消按钮
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        
        # 重置进度条和预览
        self.single_progress_bar.setValue(0)
        self.preview_text.clear()
        
        # 创建并启动工作线程
        self.transcription_worker = TranscriptionWorker(
            audio_file, output_file, model_size, hf_endpoint
        )
        
        # 连接信号
        self.transcription_worker.signals.progress.connect(self.update_single_progress)
        self.transcription_worker.signals.status.connect(self.update_single_status)
        self.transcription_worker.signals.finished.connect(self.on_single_transcription_finished)
        self.transcription_worker.signals.error.connect(self.on_error)
        self.transcription_worker.signals.log.connect(self.log)
        
        # 启动线程
        self.transcription_worker.start()
        
        self.log(f"开始转录文件: {audio_file}")
    
    def cancel_single_transcription(self):
        if self.transcription_worker and self.transcription_worker.isRunning():
            # 终止线程
            self.transcription_worker.terminate()
            self.transcription_worker.wait()
            
            # 更新UI
            self.single_status_label.setText("已取消")
            self.start_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            
            self.log("转录已取消")
    
    def start_batch_transcription(self):
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
        
        output_dir = self.batch_output_dir.text().strip()
        model_size = self.batch_model_combo.currentText()
        hf_endpoint = self.hf_combo.currentText() if self.hf_combo.currentText() else None
        
        # 禁用开始按钮，启用取消按钮
        self.batch_start_button.setEnabled(False)
        self.scan_button.setEnabled(False)
        self.batch_cancel_button.setEnabled(True)
        
        # 重置进度条
        self.batch_progress_bar.setValue(0)
        
        # 创建并启动工作线程
        self.batch_worker = BatchTranscriptionWorker(
            directory, pattern, output_dir, model_size, hf_endpoint
        )
        
        # 连接信号
        self.batch_worker.signals.progress.connect(self.update_batch_progress)
        self.batch_worker.signals.status.connect(self.update_batch_status)
        self.batch_worker.signals.finished.connect(self.on_batch_transcription_finished)
        self.batch_worker.signals.error.connect(self.on_error)
        self.batch_worker.signals.log.connect(self.log)
        
        # 启动线程
        self.batch_worker.start()
        
        self.log(f"开始批量转录目录: {directory}")
    
    def cancel_batch_transcription(self):
        if self.batch_worker and self.batch_worker.isRunning():
            # 通知线程停止
            self.batch_worker.stop()
            
            # 等待线程结束
            self.batch_worker.wait()
            
            # 更新UI
            self.batch_status_label.setText("已取消")
            self.batch_start_button.setEnabled(True)
            self.scan_button.setEnabled(True)
            self.batch_cancel_button.setEnabled(False)
            
            self.log("批量转录已取消")
    
    def update_single_progress(self, value):
        self.single_progress_bar.setValue(value)
    
    def update_single_status(self, status):
        self.single_status_label.setText(status)
        self.statusBar().showMessage(status)
    
    def update_batch_progress(self, value):
        self.batch_progress_bar.setValue(value)
    
    def update_batch_status(self, status):
        self.batch_status_label.setText(status)
        self.statusBar().showMessage(status)
    
    def on_single_transcription_finished(self, text):
        # 更新预览
        self.preview_text.setText(text)
        
        # 恢复按钮状态
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
    
    def on_batch_transcription_finished(self, _):
        # 恢复按钮状态
        self.batch_start_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.batch_cancel_button.setEnabled(False)
    
    def on_error(self, error_message):
        QMessageBox.critical(self, "错误", error_message)
        
        # 恢复按钮状态
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.batch_start_button.setEnabled(True)
        self.scan_button.setEnabled(True)
        self.batch_cancel_button.setEnabled(False)
    
    def log(self, message):
        # 添加时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        
        # 添加到日志文本框
        self.log_text.append(log_entry)
        
        # 滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.log_text.setTextCursor(cursor)
    
    def clear_log(self):
        self.log_text.clear()

# 应用程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())