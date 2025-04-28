# FaceRecognition-YOLO

基于YOLO11和FaceNet的实时人脸识别系统。

## 功能特点
- 实时人脸检测和识别
- 多样本人脸注册
- 导出识别记录
- 友好的图形界面
- 高精度识别

## 环境要求
- Python 3.8+
- 支持CUDA的GPU（推荐）
- 摄像头

## 安装步骤
1. 克隆仓库：
```bash
git clone https://github.com/yourusername/FaceRecognition-YOLO.git
cd FaceRecognition-YOLO
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法
1. 运行程序：
```bash
python main.py
```

2. 人脸注册：
   - 点击"开启摄像头"
   - 点击"注册"
   - 输入姓名和ID
   - 按'R'键5次采集人脸样本

3. 人脸识别：
   - 点击"开启摄像头"
   - 点击"识别"
   - 系统将自动识别人脸

4. 导出记录：
   - 点击"导出"将识别记录保存为CSV

## 界面展示
![界面](interface.png)

界面包含：
- 摄像头显示区域
- 注册表单
- 识别结果表格
- 控制按钮

## 项目结构
- `main.py`: 主程序文件，包含GUI实现和人脸识别逻辑
- `face_app.ui`: Qt Designer界面文件，定义应用程序界面
- `yolo11_face.pt`: 预训练的YOLO11模型权重文件，用于人脸检测
- `face_embeddings.json`: 存储已注册人脸特征向量的数据库文件
- `test_yolo.py`: YOLO人脸检测功能的测试脚本
- `requirements.txt`: Python包依赖文件
- `README.md`: 中文文档
- `README_EN.md`: 英文文档
- `LICENSE`: MIT许可证文件

## 许可证
本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。