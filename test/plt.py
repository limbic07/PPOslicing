import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# --- 1. 检查并设置设备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {device}")

# --- 2. 定义超参数 ---
# 使用大批次来最大化GPU利用率
BATCH_SIZE = 2048
EPOCHS = 5
LEARNING_RATE = 0.001

# --- 3. 准备数据集 ---
# 数据预处理和加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# --- 4. 定义一个简单的CNN模型 ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 计算展平后的尺寸: 28 -> 14 (after pool) -> 7 (after pool)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个类别

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # 展平
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# --- 5. 实例化模型、损失函数和优化器 ---
model = SimpleCNN().to(device)  # *** 关键：将模型移动到GPU ***
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 6. 训练循环（带计时） ---
print("\n开始训练...")
total_start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start_time = time.time()
    model.train()  # 设置为训练模式

    for i, (images, labels) in enumerate(train_loader):
        # *** 关键：将数据移动到GPU ***
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {loss.item():.4f}, 耗时: {epoch_duration:.2f}秒")

total_end_time = time.time()
total_duration = total_end_time - total_start_time
print(f"\n训练完成！总耗时: {total_duration:.2f}秒")

