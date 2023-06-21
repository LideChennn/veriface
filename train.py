import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.resnet import net


root_path = './data/cifar/images'
train_data = torchvision.datasets.CIFAR10(root_path,
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

test_data = torchvision.datasets.CIFAR10(root_path,
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)

# 获取数据集的长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用data_loader加载数据
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

loss_function = nn.CrossEntropyLoss()
loss_function = loss_function.cuda()

net.cuda()

# 优化器
learning_rate = 0.05  # 1e-2
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# 设置训练网络的参数
total_train_step = 0  # 记录训练的次数
total_test_step = 0  # 记录测试的次数

# 训练的轮数
epoch = 10

writer = SummaryWriter("../runs/logs")
for i in range(epoch):
    print("--------------第{}轮训练开始--------------".format(i + 1))

    # 训练步骤开始
    net.train()  # 设置网络是训练模式,对dropout等有作用
    for data in train_data_loader:
        imgs, targets = data

        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()

        outputs = net(imgs)
        loss = loss_function(outputs, targets)

        # 优化器调优
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数:{}, loss:{}".format(total_train_step, loss))  # loss.item()就是 tensor的item的值
            writer.add_scalar("train_loss", loss, total_train_step)

    # 测试步骤开始
    net.eval()  # 进入evaluation评估状态,有特殊的层一定要调用这个
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 没有梯度的影响
        for data in test_data_loader:
            imgs, targets = data

            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()

            outputs = net(imgs)
            loss = loss_function(outputs, targets)
            total_test_loss += loss.item()

            # 正确率
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print("整体测试集上的loss:{}".format(total_test_loss))  # loss.item()就是 tensor的item的值
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))  # 正确率

    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)

    total_test_step += 1

    torch.save(net, "./weight/resnet{}.pth".format(i))
    print('模型已保存')

writer.close()
