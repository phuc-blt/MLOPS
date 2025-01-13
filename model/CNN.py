import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Bật MLflow autologging
mlflow.pytorch.autolog()

# Định nghĩa mô hình CNN
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)  # Giả sử ảnh có kích thước 32x32
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 128 * 8 * 8)  # Flatten layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Chuẩn bị dữ liệu
data_transforms = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize ảnh xuống 32x32
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root='/workspace/phucnt/MLOP/data/train', transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = datasets.ImageFolder(root='/workspace/phucnt/MLOP/data/valid', transform=data_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Khởi tạo mô hình
model = CNNModel(num_classes=len(train_dataset.classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss function và optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 10
with mlflow.start_run() as run:
    # Ghi thông số hyperparameters
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("num_epochs", epochs)
    mlflow.log_param("model_architecture", "Simple CNN")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Tính loss trung bình trên train set
        avg_train_loss = running_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss}")

    # Đánh giá trên validation set
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Tính metrics validation
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    mlflow.log_metric("val_loss", avg_val_loss)
    mlflow.log_metric("val_accuracy", val_accuracy)
    print(f"Validation Loss: {avg_val_loss}, Accuracy: {val_accuracy * 100}%")

    # Lưu mô hình vào artifacts
    mlflow.pytorch.log_model(model, "model")

print("Training complete")
