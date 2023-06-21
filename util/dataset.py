import os
import random
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class LFWTripletDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # 遍历 LFW 数据集，获取文件路径和人名
        self.image_paths = []
        self.person_names = []

        for person_name in os.listdir(root_dir):
            person_dir = os.path.join(root_dir, person_name)
            if os.path.isdir(person_dir):
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    self.image_paths.append(image_path)
                    self.person_names.append(person_name)

        # 空字典，键为name
        self.name_to_indices = {name: [] for name in set(self.person_names)}

        # {人名: 图片index,图片index...}
        for i, name in enumerate(self.person_names):
            self.name_to_indices[name].append(i)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # 从数据集中获取锚点、正例和负例
        anchor_path = self.image_paths[index]
        anchor_name = self.person_names[index]

        # 确保至少有两张图像可用于选择正例, 如果一个人少于两张图片，就选另外一个人当anchor
        while len(self.name_to_indices[anchor_name]) < 2:
            index = random.randint(0, len(self.image_paths) - 1)
            anchor_path, anchor_name = self.image_paths[index], self.person_names[index]

        # 随机选一个同人的不同照片，
        positive_index = random.choice([i for i in self.name_to_indices[anchor_name] if i != index])
        positive_path = self.image_paths[positive_index]

        # 选择不同一个人的照片
        negative_name = random.choice([name for name in set(self.person_names) if name != anchor_name])
        negative_path = self.image_paths[random.choice(self.name_to_indices[negative_name])]

        # 读取并转换图像
        anchor_image = cv2.imread(anchor_path)
        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)

        positive_image = cv2.imread(positive_path)
        positive_image = cv2.cvtColor(positive_image, cv2.COLOR_BGR2RGB)

        negative_image = cv2.imread(negative_path)
        negative_image = cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB)

        # 将图像转换为 PIL Image 对象
        anchor_image = Image.fromarray(anchor_image)
        positive_image = Image.fromarray(positive_image)
        negative_image = Image.fromarray(negative_image)

        if self.transform:
            anchor_image = self.transform(anchor_image)
            positive_image = self.transform(positive_image)
            negative_image = self.transform(negative_image)

        return anchor_image, positive_image, negative_image


image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 图片三通道进行归一化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

lfw_root = "../data/lfw"
dataset = LFWTripletDataset(root_dir=lfw_root, transform=image_transform)

triplet_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


