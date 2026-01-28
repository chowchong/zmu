import sys
import os
from pathlib import Path
from typing import Optional, List

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QRadioButton, QButtonGroup, QProgressBar, 
    QListWidget, QListWidgetItem, QMessageBox, QFileDialog, QCheckBox,
    QFrame, QScrollArea, QTabWidget, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QMimeData
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QFont, QDrag

from model_manager import ModelManager
from subtitle_engine import SubtitleEngine
from styles import (
    COLORS, FONT_FAMILY, 
    PRIMARY_BUTTON_STYLE, SECONDARY_BUTTON_STYLE, 
    DROP_ZONE_STYLE, RADIO_BUTTON_STYLE, 
    PROGRESS_BAR_STYLE, CARD_STYLE, 
    TAB_STYLE, TREE_VIEW_STYLE, LOG_STYLE
)


class ModelDownloadWorker(QThread):
    """后台模型下载线程，带进度更新"""
    progress = pyqtSignal(str, int, int, float)  # model_key, downloaded_mb, total_mb, speed_mbps
    finished = pyqtSignal(str, bool, str)  # model_key, success, message
    
    def __init__(self, model_manager: ModelManager, models_to_download: List[str]):
        super().__init__()
        self.model_manager = model_manager
        self.models_to_download = models_to_download
    
    def run(self):
        """下载指定的模型"""
        for model_key in self.models_to_download:
            try:
                # Create progress callback for this specific model
                def progress_callback(downloaded, total):
                    downloaded_mb = downloaded / (1024 * 1024)
                    total_mb = total / (1024 * 1024)
                    speed = 0  # Can calculate if needed
                    self.progress.emit(model_key, int(downloaded_mb), int(total_mb), speed)
                
                # Download the model
                success = self.model_manager._download_model_with_progress(
                    model_key, 
                    progress_callback
                )
                
                if success:
                    self.finished.emit(model_key, True, f"{ModelManager.MODELS[model_key]['name']} 下载成功")
                else:
                    self.finished.emit(model_key, False, "下载失败")
                    
            except Exception as e:
                self.finished.emit(model_key, False, f"下载出错: {str(e)}")



class TranscriptionWorker(QThread):
    """后台转录线程（支持批量 + 翻译 + 高级导出）"""
    progress_signal = pyqtSignal(str, int, int)  # message, current, total
    indeterminate_signal = pyqtSignal(str)  # message (for loading stages)
    file_finished_signal = pyqtSignal(bool, str, str)  # success, filename, message
    all_finished_signal = pyqtSignal(int, int)  # total, success_count

    def __init__(self, video_paths: List[str], model_key: str, language: Optional[str], 
                 auto_open_editor: bool, translate_target: Optional[str] = None,
                 export_mode: str = 'merge', track_order: List[str] = None):
        super().__init__()
        self.video_paths = video_paths
        self.model_key = model_key
        self.language = language
        self.auto_open_editor = auto_open_editor
        self.translate_target = translate_target
        self.export_mode = export_mode  # 'merge' or 'separate'
        self.track_order = track_order or ['translated', 'original']

    def run(self):
        success_count = 0
        total = len(self.video_paths)
        
        try:
            # Indeterminate progress for initialization
            self.indeterminate_signal.emit("正在初始化引擎...")
            engine = SubtitleEngine()
            
            # Indeterminate progress for model loading
            self.indeterminate_signal.emit(f"正在加载模型: {self.model_key}...")
            engine.load_model(self.model_key)
            
            for i, video_path in enumerate(self.video_paths, 1):
                filename = Path(video_path).name
                file_ext = Path(video_path).suffix.lower()
                
                try:
                    # Check if this is an SRT file (translation-only mode)
                    if file_ext == '.srt':
                        # Skip transcription, use SRT directly
                        self.progress_signal.emit(
                            f"检测到字幕文件 {filename} ({i}/{total})...", 
                            i-1, 
                            total
                        )
                        saved_path = video_path
                        
                        # Must have translation enabled for SRT-only mode to be useful
                        if not self.translate_target:
                            self.file_finished_signal.emit(
                                False, filename, "字幕文件需要启用翻译功能"
                            )
                            continue
                    else:
                        # Normal transcription workflow for video/audio
                        self.progress_signal.emit(
                            f"正在转录 {filename} ({i}/{total})...", 
                            i-1, 
                            total
                        )
                        
                        result = engine.transcribe(
                            video_path,
                            model_key=self.model_key,
                            language=self.language
                        )
                        
                        output_path = str(Path(video_path).with_suffix('.srt'))
                        # Pass video_path to enable frame validation/alignment
                        saved_path = engine.save_srt(result, output_path, video_path=video_path)
                    
                    # Translation Step
                    if self.translate_target:
                        # Indeterminate for translation
                        self.indeterminate_signal.emit(f"正在翻译 {filename}...")
                        # 1. Generate Translated SRT (Pure)
                        trans_path = engine.translate_srt(saved_path, self.translate_target)
                        
                        # 2. Handle Export logic
                        if self.export_mode == 'merge':
                            self.progress_signal.emit(
                                f"正在合并字幕...", 
                                i, 
                                total
                            )
                            # Generate merged filename
                            merged_path = str(Path(saved_path).with_suffix('.bilingual.srt'))
                            
                            engine.merge_subtitles(
                                original_path=saved_path,
                                translated_path=trans_path,
                                track_order=self.track_order,
                                output_path=merged_path
                            )
                            saved_path = merged_path
                            
                            # Optional: Clean up intermediate files if merged?
                            # For safety, let's keep them or maybe just keep the merged one.
                            # Let's keep all for now as user might want them.
                        else:
                            # Separate mode: just return the translated path as the 'main' result to show user
                            # effectively we have saved_path (original) and trans_path (translated)
                            saved_path = f"{saved_path}\n{trans_path}"
                    
                    success_count += 1
                    self.file_finished_signal.emit(True, filename, saved_path)
                    
                except Exception as e:
                    self.file_finished_signal.emit(False, filename, str(e))
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            
        self.all_finished_signal.emit(total, success_count)


class DragDropZone(QListWidget):
    """现代化拖拽区域"""
    files_dropped = pyqtSignal(list)  # List of file paths

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.setStyleSheet(DROP_ZONE_STYLE)
        self.setMinimumHeight(180)
        
        # Add placeholder when empty
        self.update_placeholder()

    def update_placeholder(self):
        if self.count() == 0:
            self.clear()
            placeholder = QListWidgetItem("📹 拖拽视频或字幕文件到这里\n或点击下方按钮选择文件")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            font = QFont()
            font.setPointSize(14)
            placeholder.setFont(font)
            self.addItem(placeholder)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            
            file_paths = []
            for url in event.mimeData().urls():
                file_paths.append(str(url.toLocalFile()))
            
            if file_paths:
                self.files_dropped.emit(file_paths)
        else:
            event.ignore()

    def add_files(self, file_paths):
        # Remove placeholder if exists
        if self.count() == 1 and self.item(0).flags() == Qt.ItemFlag.NoItemFlags:
            self.clear()
            
        for file_path in file_paths:
            # Check if already added
            existing = [self.item(i).data(Qt.ItemDataRole.UserRole) 
                       for i in range(self.count())]
            if file_path not in existing:
                item = QListWidgetItem(f"[VIDEO] {Path(file_path).name}")
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                item.setForeground(Qt.GlobalColor.green) # Terminal style
                self.addItem(item)
    
    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.update_placeholder()

    def get_all_files(self) -> List[str]:
        files = []
        for i in range(self.count()):
            item = self.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:  # Skip placeholder
                files.append(file_path)
        return files
            
    def is_video_file(self, path):
        suffix = Path(path).suffix.lower()
        return suffix in ['.mp4', '.mkv', '.avi', '.mov', '.m4a', '.wav', '.mp3', '.srt']

    def add_files(self, file_paths):
        # Remove placeholder if exists
        if self.count() == 1 and self.item(0).flags() == Qt.ItemFlag.NoItemFlags:
            self.clear()
            
        for file_path in file_paths:
            # Check if already added
            existing = [self.item(i).data(Qt.ItemDataRole.UserRole) 
                       for i in range(self.count())]
            if file_path not in existing:
                item = QListWidgetItem(f"🎬 {Path(file_path).name}")
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.addItem(item)

    def remove_selected(self):
        for item in self.selectedItems():
            self.takeItem(self.row(item))
        self.update_placeholder()

    def get_all_files(self) -> List[str]:
        files = []
        for i in range(self.count()):
            item = self.item(i)
            file_path = item.data(Qt.ItemDataRole.UserRole)
            if file_path:  # Skip placeholder
                files.append(file_path)
        return files


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subgen - 智能字幕生成器")
        self.resize(700, 750)
        
        # Data
        self.model_manager = ModelManager()
        self.worker: Optional[TranscriptionWorker] = None
        self.last_output_path: Optional[str] = None
        
        # Init UI
        # Init UI
        self.init_ui()
        self.check_models()

    def init_ui(self):
        # Set main window background
        self.setStyleSheet(f"QMainWindow {{ background-color: {COLORS['background']}; }}")
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # Main Layout (Horizontal Splitter)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(1)
        
        # === LEFT PANEL (Vertical Splitter) ===
        left_splitter = QSplitter(Qt.Orientation.Vertical)
        left_splitter.setHandleWidth(1)
        
        # 1. Media Browser (Top Left)
        media_widget = QWidget()
        media_layout = QVBoxLayout(media_widget)
        media_layout.setContentsMargins(0, 0, 0, 0)
        media_layout.setSpacing(0)
        
        # Header
        media_header = QLabel(" [ MEDIA BROWSER ] ")
        media_header.setStyleSheet(f"background: {COLORS['card']}; color: {COLORS['primary']}; font-weight: bold; border-bottom: 1px solid {COLORS['border']}; padding: 4px;")
        media_layout.addWidget(media_header)
        
        # File System Model
        from PyQt6.QtGui import QFileSystemModel
        from PyQt6.QtWidgets import QTreeView, QHeaderView
        self.fs_model = QFileSystemModel()
        self.fs_model.setRootPath(str(Path.home()))
        
        self.tree_view = QTreeView()
        self.tree_view.setModel(self.fs_model)
        self.tree_view.setRootIndex(self.fs_model.index(str(Path.home())))
        self.tree_view.setStyleSheet(TREE_VIEW_STYLE)
        self.tree_view.setColumnHidden(1, True) # Size
        self.tree_view.setColumnHidden(2, True) # Type
        self.tree_view.setColumnHidden(3, True) # Date
        self.tree_view.header().hide()
        self.tree_view.setDragEnabled(True)
        self.tree_view.setDragDropMode(QTreeView.DragDropMode.DragOnly)
        
        media_layout.addWidget(self.tree_view)
        left_splitter.addWidget(media_widget)
        
        # 2. Settings (Bottom Left)
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(0)
        
        settings_header = QLabel(" [ SETTINGS ] ")
        settings_header.setStyleSheet(f"background: {COLORS['card']}; color: {COLORS['primary']}; font-weight: bold; border-bottom: 1px solid {COLORS['border']}; border-top: 1px solid {COLORS['border']}; padding: 4px;")
        settings_layout.addWidget(settings_header)
        
        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(TAB_STYLE)
        
        # -- Tab 1 General --
        tab_general = QWidget()
        t1_layout = QVBoxLayout(tab_general)
        t1_layout.setContentsMargins(10, 10, 10, 10)
        t1_layout.setSpacing(10)
        
        # Models
        model_label = QLabel("MODEL SELECTION:")
        t1_layout.addWidget(model_label)
        
        self.model_group = QButtonGroup()
        models_to_show = [('whisper_medium', True), ('whisper_large', False), ('whisper_small', False), ('funasr', False)]
        status = self.model_manager.check_installation()
        
        for model_key, is_default in models_to_show:
            if model_key in self.model_manager.MODELS:
                info = self.model_manager.MODELS[model_key]
                installed = status.get(model_key, False)
                status_str = "[INSTALLED]" if installed else "[MISSING]"
                display_text = f"{info['name'].ljust(20)} {status_str}"
                
                radio = QRadioButton(display_text)
                radio.setStyleSheet(RADIO_BUTTON_STYLE)
                radio.model_key = model_key
                if is_default and installed: radio.setChecked(True)
                if not installed: radio.setEnabled(False)
                self.model_group.addButton(radio)
                t1_layout.addWidget(radio)
        
        # Language
        lang_label = QLabel("LANGUAGE:")
        lang_label.setStyleSheet("margin-top: 10px;")
        t1_layout.addWidget(lang_label)
        
        lang_layout = QHBoxLayout()
        self.lang_group = QButtonGroup()
        
        for name, code, checked in [("AUTO", None, True), ("ZH", 'zh', False), ("EN", 'en', False)]:
            r = QRadioButton(name)
            r.lang_code = code
            r.setStyleSheet(RADIO_BUTTON_STYLE)
            if checked: r.setChecked(True)
            self.lang_group.addButton(r)
            lang_layout.addWidget(r)
        
        lang_layout.addStretch()
        t1_layout.addLayout(lang_layout)
        
        # Translation
        trans_label = QLabel("TRANSLATION:")
        trans_label.setStyleSheet("margin-top: 10px;")
        t1_layout.addWidget(trans_label)
        
        trans_layout = QHBoxLayout()
        self.chk_translate = QCheckBox("ENABLE")
        self.chk_translate.setStyleSheet(f"color: {COLORS['text']}; font-family: {FONT_FAMILY};")
        
        self.group_trans_target = QButtonGroup()
        self.radio_trans_zh = QRadioButton("TO ZH")
        self.radio_trans_zh.lang_code = 'zh'; self.radio_trans_zh.setStyleSheet(RADIO_BUTTON_STYLE); self.radio_trans_zh.setChecked(True)
        self.radio_trans_en = QRadioButton("TO EN")
        self.radio_trans_en.lang_code = 'en'; self.radio_trans_en.setStyleSheet(RADIO_BUTTON_STYLE)
        
        self.group_trans_target.addButton(self.radio_trans_zh)
        self.group_trans_target.addButton(self.radio_trans_en)
        
        trans_layout.addWidget(self.chk_translate)
        trans_layout.addWidget(self.radio_trans_zh)
        trans_layout.addWidget(self.radio_trans_en)
        trans_layout.addStretch()
        t1_layout.addLayout(trans_layout)
        
        self.chk_translate.toggled.connect(lambda c: self.radio_trans_zh.setEnabled(c))
        self.chk_translate.toggled.connect(lambda c: self.radio_trans_en.setEnabled(c))
        self.radio_trans_zh.setEnabled(False); self.radio_trans_en.setEnabled(False)
        
        self.chk_auto_editor = QCheckBox("AUTO OPEN EDITOR")
        self.chk_auto_editor.setChecked(True)
        self.chk_auto_editor.setStyleSheet(f"margin-top: 5px; color: {COLORS['text']}; font-family: {FONT_FAMILY};")
        t1_layout.addWidget(self.chk_auto_editor)
        
        t1_layout.addStretch()
        
        # -- Tab 2 Advanced --
        tab_advanced = QWidget()
        t2_layout = QVBoxLayout(tab_advanced)
        t2_layout.setContentsMargins(10, 10, 10, 10)
        
        t2_layout.addWidget(QLabel("TRACK ORDER (DRAG TO REORDER):"))
        self.track_list = QListWidget()
        self.track_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self.track_list.setStyleSheet(DROP_ZONE_STYLE)
        
        it1 = QListWidgetItem("[ TRANSLATED TRACK ]"); it1.setData(Qt.ItemDataRole.UserRole, 'translated'); self.track_list.addItem(it1)
        it2 = QListWidgetItem("[ ORIGINAL TRACK ]"); it2.setData(Qt.ItemDataRole.UserRole, 'original'); self.track_list.addItem(it2)
        
        t2_layout.addWidget(self.track_list)
        
        t2_layout.addWidget(QLabel("EXPORT MODE:"))
        self.export_group = QButtonGroup()
        self.radio_exp_merge = QRadioButton("MERGE TO SINGLE FILE"); self.radio_exp_merge.mode='merge'; self.radio_exp_merge.setChecked(True); self.radio_exp_merge.setStyleSheet(RADIO_BUTTON_STYLE)
        self.radio_exp_separate = QRadioButton("SEPARATE FILES"); self.radio_exp_separate.mode='separate'; self.radio_exp_separate.setStyleSheet(RADIO_BUTTON_STYLE)
        self.export_group.addButton(self.radio_exp_merge); self.export_group.addButton(self.radio_exp_separate)
        t2_layout.addWidget(self.radio_exp_merge); t2_layout.addWidget(self.radio_exp_separate)
        
        # === SRT MERGE TOOL ===
        t2_layout.addWidget(QLabel(""))  # Spacer
        merge_label = QLabel("═══ MERGE SRT FILES ═══")
        merge_label.setStyleSheet(f"color: {COLORS['primary']}; font-weight: bold; margin-top: 10px;")
        t2_layout.addWidget(merge_label)
        
        # File 1 Selection
        merge_file1_layout = QHBoxLayout()
        self.btn_merge_file1 = QPushButton("[ SELECT FILE 1 ]")
        self.btn_merge_file1.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.btn_merge_file1.clicked.connect(self.select_merge_file1)
        merge_file1_layout.addWidget(self.btn_merge_file1)
        
        self.label_merge_file1 = QLabel("未选择")
        self.label_merge_file1.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        merge_file1_layout.addWidget(self.label_merge_file1, 1)
        t2_layout.addLayout(merge_file1_layout)
        
        # File 2 Selection
        merge_file2_layout = QHBoxLayout()
        self.btn_merge_file2 = QPushButton("[ SELECT FILE 2 ]")
        self.btn_merge_file2.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.btn_merge_file2.clicked.connect(self.select_merge_file2)
        merge_file2_layout.addWidget(self.btn_merge_file2)
        
        self.label_merge_file2 = QLabel("未选择")
        self.label_merge_file2.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        merge_file2_layout.addWidget(self.label_merge_file2, 1)
        t2_layout.addLayout(merge_file2_layout)
        
        # Merge Button
        self.btn_execute_merge = QPushButton("[ ⚡ MERGE NOW ]")
        self.btn_execute_merge.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.btn_execute_merge.clicked.connect(self.execute_srt_merge)
        t2_layout.addWidget(self.btn_execute_merge)
        
        # Store selected file paths
        self.merge_file1_path = None
        self.merge_file2_path = None
        
        t2_layout.addStretch()
        
        self.tabs.addTab(tab_general, "GENERAL")
        self.tabs.addTab(tab_advanced, "ADVANCED")
        settings_layout.addWidget(self.tabs)
        left_splitter.addWidget(settings_widget)
        
        # === RIGHT PANEL (Vertical Splitter) ===
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.setHandleWidth(1)
        
        # 3. Queue / Drop Zone (Top Right)
        queue_widget = QWidget()
        queue_layout = QVBoxLayout(queue_widget)
        queue_layout.setContentsMargins(0, 0, 0, 0)
        queue_layout.setSpacing(0)
        
        queue_header = QLabel(" [ JOB QUEUE ] ")
        queue_header = QLabel(" [ JOB QUEUE ] ")
        queue_header.setStyleSheet(f"background: {COLORS['card']}; color: {COLORS['primary']}; font-weight: bold; border-bottom: 1px solid {COLORS['border']}; padding: 4px;")
        queue_layout.addWidget(queue_header)
        
        # Init Drop Zone First
        self.drop_zone = DragDropZone()
        self.drop_zone.setStyleSheet(DROP_ZONE_STYLE)
        self.drop_zone.files_dropped.connect(self.on_files_dropped)
        
        # Toolbar
        toolbar = QWidget()
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(5, 5, 5, 5)
        
        self.btn_add = QPushButton("[ + ADD FILE ]")
        self.btn_add.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.btn_add.clicked.connect(self.browse_files)
        
        self.btn_remove = QPushButton("[ - REMOVE ]")
        self.btn_remove.setStyleSheet(SECONDARY_BUTTON_STYLE)
        self.btn_remove.clicked.connect(self.drop_zone.remove_selected)
        
        self.btn_start = QPushButton("[ >> START PROCESSING << ]")
        self.btn_start.setStyleSheet(PRIMARY_BUTTON_STYLE)
        self.btn_start.clicked.connect(self.start_transcription)
        
        tb_layout.addWidget(self.btn_add)
        tb_layout.addWidget(self.btn_remove)
        tb_layout.addStretch()
        tb_layout.addWidget(self.btn_start)
        
        queue_layout.addWidget(toolbar)
        queue_layout.addWidget(self.drop_zone)
        
        right_splitter.addWidget(queue_widget)
        
        # 4. Progress / Logs (Bottom Right)
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        log_layout.setContentsMargins(0, 0, 0, 0)
        log_layout.setSpacing(0)
        
        log_header = QLabel(" [ SYSTEM LOGS ] ")
        log_header.setStyleSheet(f"background: {COLORS['card']}; color: {COLORS['primary']}; font-weight: bold; border-bottom: 1px solid {COLORS['border']}; border-top: 1px solid {COLORS['border']}; padding: 4px;")
        log_layout.addWidget(log_header)
        
        from PyQt6.QtWidgets import QTextEdit
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet(LOG_STYLE)
        log_layout.addWidget(self.log_view)
        
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setStyleSheet(f"padding: 5px; color: {COLORS['text']}; background: {COLORS['card']}; border-top: 1px solid {COLORS['border']};")
        log_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(PROGRESS_BAR_STYLE)
        self.progress_bar.setFixedHeight(10)
        self.progress_bar.setTextVisible(False)
        log_layout.addWidget(self.progress_bar)
        
        right_splitter.addWidget(log_widget)
        
        # Add to Main Splitter
        main_splitter.addWidget(left_splitter)
        main_splitter.addWidget(right_splitter)
        
        # Set Sizes (approx 30% left, 70% right)
        main_splitter.setSizes([250, 550])
        left_splitter.setSizes([400, 300])
        right_splitter.setSizes([400, 200])
        
        main_layout.addWidget(main_splitter)


    def create_card(self) -> QWidget:
        """Create a styled card widget"""
        card = QWidget()
        card.setStyleSheet(CARD_STYLE + f"padding: 16px;")
        return card

    def check_models(self):
        """检查并自动安装必需模型"""
        status = self.model_manager.check_installation()
        required_models = [k for k, v in ModelManager.MODELS.items() if v['required']]
        missing = [k for k in required_models if not status.get(k, False)]
        
        if not missing:
            print("✅ 所有必需模型已检测到")
            return
        
        model_names = [ModelManager.MODELS[k]['name'] for k in missing]
        total_size = sum(ModelManager.MODELS[k]['size_mb'] for k in missing)
        models_text = "\n".join([f"• {name}" for name in model_names])
        
        msg = (f"检测到首次运行，需要下载以下必需模型:\n\n{models_text}\n\n"
               f"总大小约 {total_size} MB\n\n"
               "下载过程中会显示进度，请保持网络连接。\n\n是否现在下载？")
        
        reply = QMessageBox.question(
            self,
            "首次运行 - 需要下载模型",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self._download_models_with_progress(missing)
        else:
            QMessageBox.warning(self, "无法继续", "缺少必需模型，无法使用转录功能。\n请在准备好后重新启动应用。")
    
    def _download_models_with_progress(self, models_to_download):
        """在后台线程下载模型，显示实时进度"""
        from PyQt6.QtWidgets import QProgressDialog
        
        self.download_progress_dialog = QProgressDialog(
            "正在初始化下载...", None, 0, len(models_to_download), self
        )
        self.download_progress_dialog.setWindowTitle("下载 AI 模型")
        self.download_progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.download_progress_dialog.setMinimumDuration(0)
        self.download_progress_dialog.setMinimumWidth(400)
        self.download_progress_dialog.setAutoClose(False)
        self.download_progress_dialog.setAutoReset(False)
        self.download_progress_dialog.setValue(0)
        self.download_progress_dialog.show()
        QApplication.processEvents()
        
        class DownloadWorker(QThread):
            progress_update = pyqtSignal(int, str, str)
            download_complete = pyqtSignal(bool, str)
            
            def __init__(self, manager, models):
                super().__init__()
                self.manager = manager
                self.models = models
            
            def run(self):
                try:
                    for idx, model_key in enumerate(self.models):
                        model_info = ModelManager.MODELS[model_key]
                        model_name = model_info['name']
                        size_mb = model_info['size_mb']
                        
                        status_msg = f"正在下载 {model_name}... ({size_mb} MB)"
                        self.progress_update.emit(idx, model_name, status_msg)
                        
                        self.manager.download_model(model_key)
                        
                        done_msg = f"✅ {model_name} 下载完成"
                        self.progress_update.emit(idx + 1, model_name, done_msg)
                    
                    self.download_complete.emit(True, "所有模型下载完成！")
                    
                except Exception as e:
                    import traceback
                    err_trace = traceback.format_exc()
                    error_msg = f"下载失败：{str(e)}\n\n详细信息:\n{err_trace}"
                    self.download_complete.emit(False, error_msg)
        
        self.download_worker = DownloadWorker(self.model_manager, models_to_download)
        self.download_worker.progress_update.connect(self._on_download_progress)
        self.download_worker.download_complete.connect(self._on_download_complete)
        self.download_worker.start()
    
    def _on_download_progress(self, index, model_name, status):
        """下载进度回调"""
        self.download_progress_dialog.setValue(index)
        self.download_progress_dialog.setLabelText(status)
        QApplication.processEvents()
        print(f"[下载进度] {status}")
    
    def _on_download_complete(self, success, message):
        """下载完成回调"""
        self.download_progress_dialog.close()
        
        if success:
            QMessageBox.information(self, "下载完成", message + "\n\n现在可以开始使用 Subgen 了！")
        else:
            QMessageBox.critical(self, "下载失败", message + "\n\n请检查网络连接或查看日志了解详细错误信息。")


    def browse_files(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, 
            "选择视频或字幕文件", 
            "", 
            "Media Files (*.mp4 *.mkv *.avi *.mov *.m4a *.wav *.mp3 *.srt);;All Files (*)"
        )
        if file_paths:
            self.on_files_dropped(file_paths)

    def on_files_dropped(self, file_paths):
        self.drop_zone.add_files(file_paths)
        count = len(self.drop_zone.get_all_files())
        self.status_label.setText(f"已添加 {count} 个文件")
        self.btn_start.setText(f"开始生成字幕 ({count} 个文件)")

    def start_transcription(self):
        files = self.drop_zone.get_all_files()
        
        if not files:
            QMessageBox.warning(self, "提示", "请先添加视频文件")
            return
            
        # Get selected model
        selected_radio = self.model_group.checkedButton()
        if not selected_radio:
            QMessageBox.warning(self, "提示", "请选择一个模型")
            return
            
        model_key = selected_radio.model_key
        
        # Get selected language
        lang_radio = self.lang_group.checkedButton()
        language = lang_radio.lang_code if lang_radio else None
        
        # Get translation target
        translate_target = None
        if self.chk_translate.isChecked():
            trans_radio = self.group_trans_target.checkedButton()
            if trans_radio:
                translate_target = trans_radio.lang_code
        
        # Get Export Settings
        export_mode = 'merge'
        if self.radio_exp_separate.isChecked():
            export_mode = 'separate'
            
        # Get Track Order
        track_order = []
        for i in range(self.track_list.count()):
            item = self.track_list.item(i)
            track_order.append(item.data(Qt.ItemDataRole.UserRole))
        
        auto_open_editor = self.chk_auto_editor.isChecked()
        
        # Disable UI
        self.btn_start.setEnabled(False)
        self.drop_zone.setEnabled(False)
        self.btn_add.setEnabled(False)
        self.btn_remove.setEnabled(False)
        self.tabs.setEnabled(False)  # Disable tabs during processing
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(files))
        self.progress_bar.setValue(0)
        
        # Start Worker
        self.worker = TranscriptionWorker(
            files, model_key, language, auto_open_editor, translate_target,
            export_mode=export_mode, track_order=track_order
        )
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.indeterminate_signal.connect(self.update_indeterminate_progress)
        self.worker.file_finished_signal.connect(self.on_file_finished)
        self.worker.all_finished_signal.connect(self.on_all_finished)
        self.worker.start()

    def update_progress(self, message, current, total):
        """Handle determinate progress updates"""
        # Switch to determinate mode
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, total)
        
        self.status_label.setText(message)
        self.progress_bar.setValue(current)
        self.log_view.append(f"[PROGRESS] {message}")
    
    def update_indeterminate_progress(self, message):
        """Handle indeterminate progress updates (loading animation)"""
        # Switch to indeterminate mode (animated)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText(message)
        self.log_view.append(f"[LOADING] {message}")

    def on_file_finished(self, success, filename, message):
        if success:
            self.last_output_path = message
            log_msg = f"✅ {filename} -> {message}"
            print(log_msg)
            self.log_view.append(f"[SUCCESS] {log_msg}")
        else:
            err_msg = f"❌ {filename}: {message}"
            print(err_msg)
            self.log_view.append(f"[ERROR] {err_msg}")

    def on_all_finished(self, total, success_count):
        self.progress_bar.setVisible(False)
        self.btn_start.setEnabled(True)
        self.drop_zone.setEnabled(True)
        self.btn_add.setEnabled(True)
        self.btn_remove.setEnabled(True)
        self.status_label.setText("就绪")
        
        if success_count == total:
            msg = f"全部完成！成功生成 {success_count} 个字幕文件"
            
            if total == 1 and self.chk_auto_editor.isChecked() and self.last_output_path:
                # Auto-open editor for single file
                try:
                    from editor_gui import SubtitleEditor
                    self.editor = SubtitleEditor(self.last_output_path)
                    self.editor.show()
                except Exception as e:
                    QMessageBox.warning(self, "提示", f"无法打开编辑器: {e}")
            else:
                QMessageBox.information(self, "完成", msg)
        else:
            QMessageBox.warning(
                self, 
                "部分失败", 
                f"成功: {success_count}/{total}\n失败: {total - success_count}"
            )
            
        self.worker = None

    def select_merge_file1(self):
        """选择第一个SRT文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择第一个字幕文件", 
            "", 
            "SRT Files (*.srt);;All Files (*)"
        )
        if file_path:
            self.merge_file1_path = file_path
            self.label_merge_file1.setText(Path(file_path).name)
            self.label_merge_file1.setStyleSheet(f"color: {COLORS['progress']}; font-size: 11px;")
            self.log_view.append(f"[MERGE] 已选择文件1: {Path(file_path).name}")

    def select_merge_file2(self):
        """选择第二个SRT文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "选择第二个字幕文件", 
            "", 
            "SRT Files (*.srt);;All Files (*)"
        )
        if file_path:
            self.merge_file2_path = file_path
            self.label_merge_file2.setText(Path(file_path).name)
            self.label_merge_file2.setStyleSheet(f"color: {COLORS['progress']}; font-size: 11px;")
            self.log_view.append(f"[MERGE] 已选择文件2: {Path(file_path).name}")

    def execute_srt_merge(self):
        """执行SRT文件合并"""
        # Validate files selected
        if not self.merge_file1_path or not self.merge_file2_path:
            QMessageBox.warning(self, "提示", "请先选择两个SRT文件")
            return
        
        if not Path(self.merge_file1_path).exists():
            QMessageBox.warning(self, "错误", f"文件1不存在: {self.merge_file1_path}")
            return
            
        if not Path(self.merge_file2_path).exists():
            QMessageBox.warning(self, "错误", f"文件2不存在: {self.merge_file2_path}")
            return
        
        try:
            # Get track order from UI
            track_order = []
            for i in range(self.track_list.count()):
                item = self.track_list.item(i)
                track_key = item.data(Qt.ItemDataRole.UserRole)
                track_order.append(track_key)
            
            # Determine which file is which based on track order
            # Assume: 'original' -> file1, 'translated' -> file2
            # But we'll use the order directly
            file1_name = Path(self.merge_file1_path).stem
            file2_name = Path(self.merge_file2_path).stem
            
            # Generate output path (save next to first file)
            output_dir = Path(self.merge_file1_path).parent
            output_path = output_dir / f"{file1_name}+{file2_name}.merged.srt"
            
            self.log_view.append(f"[MERGE] 正在合并字幕文件...")
            self.log_view.append(f"[MERGE] 轨道顺序: {' -> '.join(track_order)}")
            
            # Use SubtitleEngine to merge
            from subtitle_engine import SubtitleEngine
            engine = SubtitleEngine()
            
            # Map track order: first track gets file1, second gets file2
            # This allows flexibility in how user wants to combine
            merged_path = engine.merge_subtitles(
                original_path=self.merge_file1_path,
                translated_path=self.merge_file2_path,
                track_order=track_order,
                output_path=str(output_path)
            )
            
            self.log_view.append(f"[SUCCESS] ✨ 合并完成: {Path(merged_path).name}")
            
            # Ask if user wants to open editor
            reply = QMessageBox.question(
                self,
                "合并成功",
                f"字幕已成功合并!\n\n保存位置:\n{merged_path}\n\n是否打开编辑器？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                from editor_gui import SubtitleEditor
                self.editor = SubtitleEditor(merged_path)
                self.editor.show()
                
        except Exception as e:
            import traceback
            error_msg = f"合并失败: {str(e)}\n\n{traceback.format_exc()}"
            self.log_view.append(f"[ERROR] {error_msg}")
            QMessageBox.critical(self, "合并失败", error_msg)


def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")  # Use Fusion style for better cross-platform consistency
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
