import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
import numpy as np
from model.resnet import FaceNetModel
import torch.nn.functional as F
from util.dataset import LFWTripletDataset

# 定义训练设备cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lfw_root = "./data/lfw"
dataset = LFWTripletDataset(root_dir=lfw_root,
                            transform=transforms.Compose([
                                transforms.Resize((96, 96)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean= [0.5865, 0.4551, 0.3913], std=[0.2292, 0.2240, 0.2251])
                            ]))

# 80%为训练集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

# 模型
facenet = FaceNetModel().to(device)

# 损失函数
loss_function = nn.TripletMarginLoss(margin=1.0, p=2)

# 优化器
optimizer = optim.Adam(facenet.parameters(), lr=0.0001)
# 自动调节学习率
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
least_loss = float('inf')
epochs = 30
writer = SummaryWriter('./runs/logs')
# 从训练数据加载器中获取一个样本输入
sample_images, sample_labels = next(iter(train_dataloader))
# 将模型和样本输入添加到 TensorBoard
writer.add_graph(facenet, sample_images)

for epoch in range(1, epochs + 1):
    print("--------------第{}轮训练开始--------------".format(epoch))
    train_loss = 0.0

    facenet.train()
    for anchors, positives, negatives in train_dataloader:
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

        optimizer.zero_grad()

        anchors_embeddings = facenet(anchors)
        positives_embeddings = facenet(positives)
        negatives_embeddings = facenet(negatives)

        loss = loss_function(anchors_embeddings, positives_embeddings, negatives_embeddings)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(train_dataloader)
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, loss:{}".format(total_train_step, loss))

    writer.add_scalar('Loss/train', train_loss, epoch)

    # 积累机会，调用道step_size = 10 就会更新学习率
    scheduler.step()

    facenet.eval()
    running_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for anchors, positives, negatives in test_dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            anchors_embeddings = facenet(anchors)
            positives_embeddings = facenet(positives)
            negatives_embeddings = facenet(negatives)

            # 计算测试集损失
            val_loss = loss_function(anchors_embeddings, positives_embeddings, negatives_embeddings)
            running_val_loss += val_loss.item()
            num_val_batches += 1

        # 计算平均测试损失
        val_loss = running_val_loss / num_val_batches
        print("Epoch: {}, Validation Loss: {}".format(epoch, val_loss))
        writer.add_scalar('Loss/validation', val_loss, epoch)

writer.close()