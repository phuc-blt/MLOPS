import mlflow
import os
from ultralytics import YOLO

# Chỉ định thư mục lưu trữ MLflow trong thư mục mlruns
os.environ['MLFLOW_TRACKING_URI'] = 'file:///workspace/phucnt/MLOP/mlruns'

mlflow.autolog()

# Đặt các tham số huấn luyện
batch_size = 6
learning_rate = 0.001

# Log các tham số huấn luyện với MLflow
mlflow.log_param('batch_size', batch_size)
mlflow.log_param('learning_rate', learning_rate)

# Tải mô hình YOLO đã huấn luyện
model = YOLO("yolo11n.pt")

# Huấn luyện mô hình YOLO trên 2 GPU
results = model.train(
    data="/workspace/phucnt/MLOP/brain tumor detection.v2-mahitha.yolov11/data.yaml",
    epochs=3,
    imgsz=640,
    device=[0, 1],  # Sử dụng 2 GPU (0 và 1)
    batch=batch_size,
    lr0=learning_rate
)

# Lưu mô hình YOLO và cấu hình vào file
model.save("yolo_weights.h5")
model.save("yolo_config.cfg")

# Log các artifacts với MLflow
mlflow.log_artifact('yolo_weights.h5')
mlflow.log_artifact('yolo_config.cfg')

print("Training complete and results logged.")
