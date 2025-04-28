import sys
import cv2
import torch
import numpy as np
import json
import csv
from datetime import datetime
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox, QFileDialog, QAbstractItemView, QVBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from facenet_pytorch import InceptionResnetV1
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDING_PATH = 'face_embeddings.json'

class FaceApp(QWidget):
    def __init__(self):
        super().__init__()
        loader = QUiLoader()
        self.ui = loader.load('face_app.ui', self)
        if not self.ui:
            print("UI文件加载失败，请检查face_app.ui路径和文件名！")
            sys.exit(1)
        
        # 设置窗口属性
        self.setWindowTitle("深度学习人脸识别系统")
        self.setGeometry(100, 100, 950, 600)  # 设置窗口位置和大小
        
        # 将UI布局添加到主窗口
        layout = QVBoxLayout()
        layout.addWidget(self.ui)
        self.setLayout(layout)
        
        # 初始化摄像头状态
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.camera_active = False
        
        self.mode = None  # 'register' or 'recognize'
        self.register_embeddings = []
        self.register_count = 5
        self.capture_flag = False
        self.result_records = []  # 存储识别结果

        # 模型
        self.yolo_model = YOLO('./yolo11_face.pt')
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        # 绑定按钮
        self.ui.registerBtn.clicked.connect(self.start_register)
        self.ui.recognizeBtn.clicked.connect(self.start_recognize)
        self.ui.exportBtn.clicked.connect(self.export_results)
        self.ui.startCameraBtn.clicked.connect(self.start_camera)
        self.ui.stopCameraBtn.clicked.connect(self.stop_camera)

        # 设置表头
        self.ui.resultTable.setHorizontalHeaderLabels(['时间', '姓名', 'ID'])
        self.ui.resultTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        
        # 初始状态
        self.ui.stopCameraBtn.setEnabled(False)
        self.ui.registerBtn.setEnabled(False)
        self.ui.recognizeBtn.setEnabled(False)

    def start_camera(self):
        if not self.camera_active:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                QMessageBox.warning(self, "警告", "无法打开摄像头！")
                return
            self.camera_active = True
            self.timer.start(30)
            self.ui.startCameraBtn.setEnabled(False)
            self.ui.stopCameraBtn.setEnabled(True)
            self.ui.registerBtn.setEnabled(True)
            self.ui.recognizeBtn.setEnabled(True)
            self.ui.statusLabel.setText("状态：摄像头已开启")

    def stop_camera(self):
        if self.camera_active:
            self.timer.stop()
            if self.camera:
                self.camera.release()
            self.camera_active = False
            self.ui.startCameraBtn.setEnabled(True)
            self.ui.stopCameraBtn.setEnabled(False)
            self.ui.registerBtn.setEnabled(False)
            self.ui.recognizeBtn.setEnabled(False)
            self.ui.statusLabel.setText("状态：摄像头已关闭")
            # 清空画面
            self.ui.cameraLabel.clear()
            self.ui.cameraLabel.setText("摄像头画面")

    def update_frame(self):
        if not self.camera_active or not self.camera:
            return
            
        ret, frame = self.camera.read()
        if not ret:
            return
            
        show_frame = frame.copy()
        if self.mode == 'register' and len(self.register_embeddings) < self.register_count:
            results = self.yolo_model(frame)
            if len(results[0].boxes) > 0:
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                face = frame[y1:y2, x1:x2]
                cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if self.capture_flag:
                    embedding = self.get_face_embedding(face)
                    self.register_embeddings.append(embedding)
                    self.capture_flag = False
                    self.ui.statusLabel.setText(f"状态：已采集 {len(self.register_embeddings)}/{self.register_count} 张")
                    if len(self.register_embeddings) == self.register_count:
                        self.save_registration()
        elif self.mode == 'recognize':
            results = self.yolo_model(frame)
            if len(results[0].boxes) > 0:
                x1, y1, x2, y2 = map(int, results[0].boxes[0].xyxy[0])
                face = frame[y1:y2, x1:x2]
                embedding = self.get_face_embedding(face)
                name, student_id, sim = self.recognize(embedding)
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if name:
                    cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(show_frame, f"{name} ({student_id})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.ui.statusLabel.setText(f"状态：识别为 {name} (ID: {student_id}) 相似度: {sim:.2f}")
                    if not self.result_records or self.result_records[-1][1] != name or self.result_records[-1][2] != student_id:
                        self.add_result(now, name, student_id)
                else:
                    cv2.rectangle(show_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(show_frame, "Unknown", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.ui.statusLabel.setText("状态：未识别到已注册人脸")
        
        # 显示画面
        rgb_image = cv2.cvtColor(show_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.ui.cameraLabel.setPixmap(QPixmap.fromImage(qt_image))

    def start_register(self):
        self.mode = 'register'
        self.register_embeddings = []
        self.capture_flag = False
        self.ui.statusLabel.setText("状态：请对准摄像头，按R键采集一张人脸")
        self.ui.cameraLabel.setFocus()
        self.ui.cameraLabel.keyPressEvent = self.keyPressEvent

    def keyPressEvent(self, event):
        if self.mode == 'register' and event.key() == Qt.Key_R:
            self.capture_flag = True

    def save_registration(self):
        name = self.ui.nameEdit.text().strip()
        student_id = self.ui.idEdit.text().strip()
        if not name or not student_id:
            QMessageBox.warning(self, "警告", "请填写姓名和ID")
            self.register_embeddings = []
            self.mode = None
            return
        avg_embedding = np.mean(self.register_embeddings, axis=0)
        try:
            with open(EMBEDDING_PATH, 'r') as f:
                database = json.load(f)
        except:
            database = {}
        database[student_id] = {'name': name, 'embedding': avg_embedding.tolist()}
        with open(EMBEDDING_PATH, 'w') as f:
            json.dump(database, f)
        self.ui.statusLabel.setText(f"状态：{name} (ID: {student_id}) 注册成功！")
        self.mode = None

    def start_recognize(self):
        self.mode = 'recognize'
        self.ui.statusLabel.setText("状态：识别中...")

    def get_face_embedding(self, face_img):
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (160, 160))
        face_tensor = torch.tensor(face_resized).permute(2, 0, 1).float() / 255.0
        face_tensor = (face_tensor - 0.5) / 0.5
        face_tensor = face_tensor.unsqueeze(0)
        with torch.no_grad():
            embedding = self.resnet(face_tensor).numpy()[0]
        return embedding

    def recognize(self, embedding):
        try:
            with open(EMBEDDING_PATH, 'r') as f:
                database = json.load(f)
            for k in database:
                database[k]['embedding'] = np.array(database[k]['embedding'])
        except:
            return None, None, 0
        best_match = None
        best_similarity = 0
        best_id = None
        for student_id, data in database.items():
            similarity = cosine_similarity([embedding], [data['embedding']])[0][0]
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = data['name']
                best_id = student_id
        if best_similarity >= 0.7:
            return best_match, best_id, best_similarity
        else:
            return None, None, best_similarity

    def add_result(self, time_str, name, student_id):
        row = self.ui.resultTable.rowCount()
        self.ui.resultTable.insertRow(row)
        self.ui.resultTable.setItem(row, 0, self.create_table_item(time_str))
        self.ui.resultTable.setItem(row, 1, self.create_table_item(name))
        self.ui.resultTable.setItem(row, 2, self.create_table_item(student_id))
        self.result_records.append([time_str, name, student_id])

    def create_table_item(self, text):
        from PySide6.QtWidgets import QTableWidgetItem
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        return item

    def export_results(self):
        if not self.result_records:
            QMessageBox.information(self, "提示", "没有识别记录可导出。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出为CSV", "识别记录.csv", "CSV Files (*.csv)")
        if path:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['时间', '姓名', 'ID'])
                writer.writerows(self.result_records)
            QMessageBox.information(self, "导出成功", f"已导出到 {path}")

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = FaceApp()
    win.show()
    sys.exit(app.exec())