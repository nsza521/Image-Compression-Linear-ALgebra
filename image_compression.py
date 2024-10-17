import sys
import os
import numpy as np
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                             QWidget, QFileDialog, QSlider, QLabel, QGroupBox, QFormLayout, 
                             QSpinBox, QProgressBar)
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt5.QtCore import Qt, QThread, pyqtSignal

def average_pooling(channel, pool_size=2):
    pooled_height = channel.shape[0] // pool_size
    pooled_width = channel.shape[1] // pool_size
    pooled_channel = np.zeros((pooled_height, pooled_width), dtype=np.float32)
    
    for i in range(pooled_height):
        for j in range(pooled_width):
            start_i, start_j = i * pool_size, j * pool_size
            block = channel[start_i:start_i + pool_size, start_j:start_j + pool_size]
            pooled_channel[i, j] = np.mean(block)
    
    return pooled_channel

def expand_pooled_array(pooled_array, original_shape):
    expanded = np.repeat(np.repeat(pooled_array, 2, axis=0), 2, axis=1)
    if expanded.shape[0] > original_shape[0]:
        expanded = expanded[:original_shape[0], :]
    if expanded.shape[1] > original_shape[1]:
        expanded = expanded[:, :original_shape[1]]
    return expanded

class CompressionThread(QThread):
    update_progress = pyqtSignal(int)
    compression_done = pyqtSignal(np.ndarray)

    def __init__(self, image, k_value):
        super().__init__()
        self.image = image
        self.k_value = k_value

    def run(self):
        image_ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(image_ycrcb)
        
        Cr_pooled = average_pooling(Cr)
        Cb_pooled = average_pooling(Cb)
        
        self.update_progress.emit(20)

        U_Y, S_Y, Vt_Y = np.linalg.svd(Y, full_matrices=False)
        k = min(self.k_value, len(S_Y))
        Y_compressed = np.dot(U_Y[:, :k], np.dot(np.diag(S_Y[:k]), Vt_Y[:k, :]))
        
        self.update_progress.emit(50)

        U_Cr, S_Cr, Vt_Cr = np.linalg.svd(Cr_pooled, full_matrices=False)
        U_Cb, S_Cb, Vt_Cb = np.linalg.svd(Cb_pooled, full_matrices=False)
        
        k_chroma = min(self.k_value // 2, len(S_Cr), len(S_Cb))  # Use fewer components for chroma
        
        Cr_compressed = np.dot(U_Cr[:, :k_chroma], np.dot(np.diag(S_Cr[:k_chroma]), Vt_Cr[:k_chroma, :]))
        Cb_compressed = np.dot(U_Cb[:, :k_chroma], np.dot(np.diag(S_Cb[:k_chroma]), Vt_Cb[:k_chroma, :]))
        
        self.update_progress.emit(80)

        Cr_expanded = expand_pooled_array(Cr_compressed, Cr.shape)
        Cb_expanded = expand_pooled_array(Cb_compressed, Cb.shape)

        Y_compressed = np.clip(Y_compressed, 0, 255).astype(np.uint8)
        Cr_expanded = np.clip(Cr_expanded, 0, 255).astype(np.uint8)
        Cb_expanded = np.clip(Cb_expanded, 0, 255).astype(np.uint8)

        compressed_ycrcb = cv2.merge([Y_compressed, Cr_expanded, Cb_expanded])
        compressed_image = cv2.cvtColor(compressed_ycrcb, cv2.COLOR_YCrCb2BGR)
        
        self.update_progress.emit(100)
        self.compression_done.emit(compressed_image)

class ImageCompressionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.compressed_image = None
        self.original_size = 0

    def initUI(self):
        self.setWindowTitle('Advanced Image Compression with SVD and Pooling')
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("""
            QMainWindow, QWidget {background-color: #2b2b2b; color: #ffffff;}
            QLabel {font-size: 12px; color: #ffffff;}
            QPushButton {
                background-color: #4CAF50;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {background-color: #45a049;}
            QPushButton:disabled {background-color: #555555; color: #888888;}
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #606060);
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #b4b4b4, stop:1 #8f8f8f);
                border: 1px solid #5c5c5c;
                width: 18px;
                margin: -2px 0;
                border-radius: 3px;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                padding: 10px;
                border: 1px solid #555555;
                border-radius: 6px;
                color: #ffffff;
            }
            QProgressBar {
                height: 20px;
                font-size: 12px;
                background-color: #3a3a3a;
                border-radius: 6px;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 20px;
            }
        """)

        self.load_button = QPushButton('Load Image', self)
        self.compress_button = QPushButton('Compress', self)
        self.download_button = QPushButton('Download Compressed Image', self)
        self.download_button.setEnabled(False)

        self.k_slider = QSlider(Qt.Horizontal)
        self.k_slider.setMinimum(1)
        self.k_slider.setMaximum(100)
        self.k_slider.setValue(50)
        self.k_label = QLabel('K value: 50')

        self.original_image_label = QLabel(self)
        self.original_image_label.setAlignment(Qt.AlignCenter)
        self.original_image_label.setStyleSheet("border: 1px solid #555555;")
        self.compressed_image_label = QLabel(self)
        self.compressed_image_label.setAlignment(Qt.AlignCenter)
        self.compressed_image_label.setStyleSheet("border: 1px solid #555555;")

        self.original_size_label = QLabel('Size: N/A')
        self.compressed_size_label = QLabel('Size: N/A')
        self.mse_label = QLabel('MSE: N/A')

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.hide()

        main_layout = QVBoxLayout()
        image_layout = QHBoxLayout()
        control_layout = QVBoxLayout()

        before_group = QGroupBox('Before Compression')
        before_layout = QVBoxLayout()
        before_layout.addWidget(self.original_image_label)
        before_layout.addWidget(self.original_size_label)
        before_group.setLayout(before_layout)

        after_group = QGroupBox('After Compression')
        after_layout = QVBoxLayout()
        after_layout.addWidget(self.compressed_image_label)
        after_layout.addWidget(self.compressed_size_label)
        after_layout.addWidget(self.mse_label)
        after_group.setLayout(after_layout)

        image_layout.addWidget(before_group)
        image_layout.addWidget(after_group)

        control_layout.addWidget(self.load_button)
        control_layout.addWidget(self.compress_button)
        control_layout.addWidget(self.download_button)
        control_layout.addWidget(self.k_slider)
        control_layout.addWidget(self.k_label)
        control_layout.addWidget(self.progress_bar)

        main_layout.addLayout(image_layout)
        main_layout.addLayout(control_layout)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.load_button.clicked.connect(self.load_image)
        self.compress_button.clicked.connect(self.compress_image)
        self.download_button.clicked.connect(self.download_image)
        self.k_slider.valueChanged.connect(self.update_k_value)

    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            self.image = cv2.imread(file_name)
            self.image = self.resize_to_even(self.image)  # ปรับขนาดภาพให้เป็นเลขคู่
            self.display_image(self.image, self.original_image_label)
            self.original_size = os.path.getsize(file_name)
            self.update_original_size_label()
            self.compressed_image_label.clear()
            self.compressed_size_label.setText("Size: N/A")
            self.mse_label.setText("MSE: N/A")
            self.download_button.setEnabled(False)

    def resize_to_even(self, image):
        height, width = image.shape[:2]
        new_height = height if height % 2 == 0 else height - 1
        new_width = width if width % 2 == 0 else width - 1
        if new_height != height or new_width != width:
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return image

    def compress_image(self):
        if self.image is not None:
            k_value = self.k_slider.value()
            self.progress_bar.show()
            self.compress_button.setEnabled(False)
            self.compression_thread = CompressionThread(self.image, k_value)
            self.compression_thread.update_progress.connect(self.update_progress_bar)
            self.compression_thread.compression_done.connect(self.compression_finished)
            self.compression_thread.start()

    def compression_finished(self, compressed_image):
        self.compressed_image = compressed_image
        self.display_image(self.compressed_image, self.compressed_image_label)
        self.update_compressed_size_label()
        self.calculate_and_display_mse()
        self.progress_bar.hide()
        self.compress_button.setEnabled(True)
        self.download_button.setEnabled(True)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def display_image(self, image, label):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_k_value(self, value):
        self.k_label.setText(f'K value: {value}')

    def update_original_size_label(self):
        if self.image is not None:
            height, width = self.image.shape[:2]
            size_kb = self.original_size / 1024
            self.original_size_label.setText(f"Size: {width}x{height}, {size_kb:.2f} KB")

    def update_compressed_size_label(self):
        if self.compressed_image is not None:
            height, width = self.compressed_image.shape[:2]
            
            compression_ratio = self.k_slider.value() / min(self.image.shape[:2])
            estimated_size = self.original_size * compression_ratio
            size_kb = estimated_size / 1024
            
            self.compressed_size_label.setText(f"Estimated Size: {width}x{height}, {size_kb:.2f} KB")

    def calculate_and_display_mse(self):
        if self.image is not None and self.compressed_image is not None:
            mse = self.calculate_mse(self.image, self.compressed_image)
            self.mse_label.setText(f'MSE: {mse:.2f}')

    def calculate_mse(self, original, compressed):
        original = original.astype(np.float32)
        compressed = compressed.astype(np.float32)
        diff = original - compressed
        return np.mean(diff ** 2)

    def download_image(self):
        if self.compressed_image is not None:
            file_name, _ = QFileDialog.getSaveFileName(self, "Save Compressed Image", "", "Images (*.png *.jpg *.bmp)")
            if file_name:
                cv2.imwrite(file_name, self.compressed_image)
                actual_size = os.path.getsize(file_name)
                size_kb = actual_size / 1024
                height, width = self.compressed_image.shape[:2]
                self.compressed_size_label.setText(f"Actual Size: {width}x{height}, {size_kb:.2f} KB")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageCompressionUI()
    ex.show()
    sys.exit(app.exec_())