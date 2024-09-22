import torch
import torch.nn as nn
from torch import optim
from UNet import UNet
from ImageSegmentationDataset import *

# 检查是否有可用的GPU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化模型、损失函数和优化器  
model = UNet(in_channels=3, out_channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()  # 使用带逻辑的二元交叉熵损失  
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型  
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # 前向传播  
        outputs = model(images)

        # 计算损失  
        loss = criterion(outputs, masks)

        # 反向传播和优化  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')

print('训练完成！')