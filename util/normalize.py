import os
import numpy as np
from PIL import Image
from torchvision import transforms

# 定义数据集的根目录
lfw_root = "../data/lfw"

# 获取所有图像的路径
image_paths = []
for folder in os.listdir(lfw_root):
    for img_name in os.listdir(os.path.join(lfw_root, folder)):
        image_paths.append(os.path.join(lfw_root, folder, img_name))

# 初始化存储每个通道的像素值的列表
pixels_r = []
pixels_g = []
pixels_b = []

# 遍历数据集中的每个图像
for img_path in image_paths:
    img = Image.open(img_path)
    img_np = np.array(img) / 255.0  # 将像素值归一化到[0, 1]

    # 将像素值添加到相应通道的列表中
    pixels_r.extend(img_np[:, :, 0].flatten())
    pixels_g.extend(img_np[:, :, 1].flatten())
    pixels_b.extend(img_np[:, :, 2].flatten())

# 计算每个通道的均值和标准差
mean_r = np.mean(pixels_r)
mean_g = np.mean(pixels_g)
mean_b = np.mean(pixels_b)

std_r = np.std(pixels_r)
std_g = np.std(pixels_g)
std_b = np.std(pixels_b)

print("Mean: [{:.4f}, {:.4f}, {:.4f}]".format(mean_r, mean_g, mean_b))
print("Std: [{:.4f}, {:.4f}, {:.4f}]".format(std_r, std_g, std_b))