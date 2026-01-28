from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QHeaderView,
    QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt
import pysrt
from pathlib import Path

class SubtitleEditor(QMainWindow):
    def __init__(self, srt_path=None):
        super().__init__()
        self.setWindowTitle("Subgen - 字幕编辑器")
        self.resize(800, 600)
        self.srt_path = srt_path
        self.subs = None
        
        self.init_ui()
        if srt_path:
            self.load_srt(srt_path)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Toolbar
        toolbar = QHBoxLayout()
        
        btn_open = QPushButton("打开 SRT")
        btn_open.clicked.connect(self.open_file_dialog)
        toolbar.addWidget(btn_open)
        
        btn_save = QPushButton("保存")
        btn_save.clicked.connect(self.save_srt)
        toolbar.addWidget(btn_save)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["开始时间", "结束时间", "内容"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)

    def open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开字幕文件", "", "SRT Files (*.srt);;All Files (*)"
        )
        if file_path:
            self.load_srt(file_path)

    def load_srt(self, path):
        self.srt_path = path
        self.setWindowTitle(f"Subgen Editor - {Path(path).name}")
        
        try:
            self.subs = pysrt.open(path)
            self.table.setRowCount(len(self.subs))
            
            for row, sub in enumerate(self.subs):
                # Start
                item_start = QTableWidgetItem(str(sub.start))
                self.table.setItem(row, 0, item_start)
                
                # End
                item_end = QTableWidgetItem(str(sub.end))
                self.table.setItem(row, 1, item_end)
                
                # Text
                item_text = QTableWidgetItem(sub.text)
                self.table.setItem(row, 2, item_text)
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法读取文件: {e}")

    def save_srt(self):
        if not self.subs:
            return
            
        try:
            # Update internal structure from table
            for row in range(self.table.rowCount()):
                start_str = self.table.item(row, 0).text()
                end_str = self.table.item(row, 1).text()
                text = self.table.item(row, 2).text()
                
                # Simple update (pysrt parses strings automatically usually, but let's be careful)
                # pysrt SubRipItem expects objects or we parse string to time
                # Ideally we update the existing object properties
                
                self.subs[row].start = start_str
                self.subs[row].end = end_str
                self.subs[row].text = text
            
            # Save
            if self.srt_path:
                self.subs.save(self.srt_path, encoding='utf-8')
                QMessageBox.information(self, "成功", "字幕已保存！")
            else:
                self.save_as()
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")

    def save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "另存为", "", "SRT Files (*.srt)")
        if path:
            self.srt_path = path
            self.save_srt()
