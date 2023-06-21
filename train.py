import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from model.resnet import FaceNetModel
from util.dataset import LFWTripletDataset

# 定义训练设备cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lfw_root = "./data/lfw"
dataset = LFWTripletDataset(root_dir=lfw_root,
                            transform=transforms.Compose([
                                transforms.Resize((112, 112)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ]))

# 70%为训练集
train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

facenet = FaceNetModel().to(device)

loss_function = nn.TripletMarginLoss(margin=1.0)

optimizer = optim.Adam(facenet.parameters(), lr=0.05)

total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数
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

    facenet.eval()
    test_loss = 0

    with torch.no_grad():
        for anchors, positives, negatives in test_dataloader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            anchors_embeddings = facenet(anchors)
            positives_embeddings = facenet(positives)
            negatives_embeddings = facenet(negatives)

            loss = loss_function(anchors_embeddings, positives_embeddings, negatives_embeddings)

            test_loss += loss.item() / len(test_dataloader)

            total_test_step += 1

    print(f"Epoch: {epoch}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")