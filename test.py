import torch
import os
import numpy as np
from model.resnet import FaceNetModel
from PIL import Image
from torchvision import transforms

from util.object_detect import object_detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_path = "./weight/facenet_checkpoint16.pth"
facenet = FaceNetModel().to(device)
facenet.load_state_dict(torch.load(model_path))
facenet.eval()

# 计算阈值，或从训练过程中确定
threshold = 1.0


# 对一张图像预处理
def preprocess(image_path):
    # 目标检测，检测到人脸
    img = object_detect(img_dir=image_path)[0]
    pil_image = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])
    return transform(pil_image).unsqueeze(0)


# 计算两个向量之间的欧氏距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# 从文件夹中加载人脸库
face_database_path = "./data/face/database"
face_database = {}
for person_name in os.listdir(face_database_path):
    person_folder = os.path.join(face_database_path, person_name)
    if person_name not in face_database:
        face_database[person_name] = []
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # 处理路径得到图片张量
        image_tensor = preprocess(image_path).to(device)
        with torch.no_grad():
            embedding = facenet(image_tensor).cpu().numpy()
        face_database[person_name].append(embedding)

# 读取待识别的图像
test_image_path = "./data/face/test/cxk"
test_image_tensor = preprocess(test_image_path).to(device)

# 提取待识别图像的特征向量
with torch.no_grad():
    test_embedding = facenet(test_image_tensor).cpu().numpy()

# 计算待识别图像与人脸库中所有图像的欧氏距离
distances = {}
for person_name, embeddings in face_database.items():
    min_distance = float('inf')
    for embedding in embeddings:
        distance = euclidean_distance(test_embedding, embedding)
        min_distance = min(min_distance, distance)
    distances[person_name] = min_distance

# 根据阈值判断待识别图像与人脸库中哪个人脸最接近
closest_name, closest_distance = min(distances.items(), key=lambda x: x[1])
if closest_distance <= threshold:
    print("识别结果: {}，距离: {:.4f}".format(closest_name, closest_distance))
else:
    print("未识别到匹配的人脸")

for person_name, distance in distances.items():
    print(f"与 {person_name} 的最小距离: {distance:.4f}")