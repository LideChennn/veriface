import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import random_split, DataLoader
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
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

# 70%为训练集
train_size = int(0.7 * len(dataset))
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
epochs = 20

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

    # 积累机会，调用道step_size = 10 就会更新学习率
    scheduler.step()

    facenet.eval()
    # 初始化距离和标签列表
    distances = []
    labels = []

    with torch.no_grad():
        for anchors, positives, negatives in test_dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            anchors_embeddings = facenet(anchors)
            positives_embeddings = facenet(positives)
            negatives_embeddings = facenet(negatives)

            positive_distances = F.pairwise_distance(anchors_embeddings, positives_embeddings, p=2.0)  # 计算欧式举例
            negative_distances = F.pairwise_distance(anchors_embeddings, negatives_embeddings, p=2.0)

            distances.extend(positive_distances.tolist())
            distances.extend(negative_distances.tolist())

            labels.extend([1] * positive_distances.size(0))  # 正例对标签为 1
            labels.extend([0] * negative_distances.size(0))  # 负例对标签为 0

            total_test_step += 1

    # 计算阈值
    threshold = np.percentile(distances, 50)  # 例如，使用距离的中位数作为阈值
    # 为距离分配类别
    predictions = [1 if d <= threshold else 0 for d in distances]
    # 计算精度、召回率和 F1 分数
    precision, recall, f1_score, _ = precision_recall_fscore_support(labels, predictions, average='binary')

    print("学习率：{}".format(scheduler.get_lr()))
    print("Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(precision, recall, f1_score))
    print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}")

    if train_loss < least_loss:
        print("找到更优的训练结果,保存模型")
        torch.save(facenet.state_dict(), "./weight/facenet_checkpoint{}.pth".format(epoch))
        least_loss = train_loss
