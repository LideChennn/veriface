import torch
import torch.nn as nn
import torch.optim as optim
from model.resnet import FaceNetModel

# 定义训练设备cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

triplet_dataloader = None
facenet_model = FaceNetModel().to(device)
triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
optimizer = optim.Adam(facenet_model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):

    facenet_model.train()
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_dataloader):
        # 将输入数据转移到 GPU（如果可用）
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # 清空梯度
        optimizer.zero_grad()

        # 计算嵌入
        anchor_embedding = facenet_model(anchor)
        positive_embedding = facenet_model(positive)
        negative_embedding = facenet_model(negative)

        # 计算 Triplet 损失
        loss = triplet_loss(anchor_embedding, positive_embedding, negative_embedding)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        # 输出损失信息
        if batch_idx % log_interval == 0:
            print(f'Epoch: {epoch}/{num_epochs}, Batch: {batch_idx}/{len(triplet_dataloader)}, Loss: {loss.item()}')