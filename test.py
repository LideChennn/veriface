import cv2
import torch
import os
import numpy as np
from model.resnet import FaceNetModel
from PIL import Image
from torchvision import transforms

from util.object_detect import object_detect

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_path = "./weight/facenet_checkpoint14_0.0007.pth"
facenet = FaceNetModel().to(device)
facenet.load_state_dict(torch.load(model_path))
facenet.eval()

# 计算阈值，或从训练过程中确定
threshold = 1.0


# 对一张图像进行预处理，获取适用于模型的图像张量
def preprocess(image_path):
    imgs = object_detect(img_dir=image_path)
    if len(imgs) == 0:
        img = cv2.imread(image_path)
    else:
        img = imgs[0]

    pil_image = Image.fromarray(img)

    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
    ])
    return transform(pil_image).unsqueeze(0)


# 计算两个向量之间的欧氏距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


# 从文件夹中加载人脸库, face_database字典{ person_name :  list(每个图片的embedding) }
def load_face_database(face_database_path):
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
    return face_database


# 计算待识别图像与人脸库中所有图像的最小欧氏距离 返回值distances 字典 {person_name : min_distance}
def compute_distances(test_embedding, face_database):
    distances = {}

    for person_name, embeddings in face_database.items():
        min_distance = float('inf')

        # 找最小的欧氏距离
        for embedding in embeddings:
            distance = euclidean_distance(test_embedding, embedding)
            min_distance = min(min_distance, distance)

        distances[person_name] = min_distance

    return distances


# 计算待识别图像与人脸库中所有图像的平均欧氏距离 返回值distances 字典 {person_name : min_distance}
# def compute_distances(test_embedding, face_database):
#     distances = {}
#
#     for person_name, embeddings in face_database.items():
#         avg_distance = 0
#
#         # 计算平均欧氏距离
#         for embedding in embeddings:
#             distance = euclidean_distance(test_embedding, embedding)
#             avg_distance += distance
#
#         avg_distance /= len(embeddings)
#         distances[person_name] = avg_distance
#
#     return distances


# 根据阈值判断待识别图像与人脸库中哪个人脸最接近
def recognize_face(test_image_path, face_database, threshold):
    test_image_tensor = preprocess(test_image_path).to(device)

    # 获取文件夹名称，就是某个人
    label = os.path.basename(os.path.dirname(test_image_path))

    # 提取待识别图像的特征向量
    with torch.no_grad():
        test_embedding = facenet(test_image_tensor).cpu().numpy()

    distances = compute_distances(test_embedding, face_database)

    closest_name, closest_distance = min(distances.items(), key=lambda x: x[1])
    if closest_distance <= threshold:
        print("识别结果: {}，{}，距离: {:.4f}".format(closest_name, closest_name == label, closest_distance))
    else:
        print("未识别到匹配的人脸")

    for person_name, distance in distances.items():
        print(f"与 {person_name} 的最小距离: {distance:.4f}")

    # 最像的人等于文件名
    return closest_name == label


def recognize_dir(input_dir, face_database):
    hit = 0
    img_sum = 0
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)

        # 如果预测成功
        if recognize_face(img_path, face_database, threshold):
            hit += 1
        img_sum += 1

    print("一共{}张图片,命中{}个,命中率: {}\n".format(img_sum, hit, hit / img_sum))


if __name__ == "__main__":
    face_database_path = "./data/face/database"
    face_database = load_face_database(face_database_path)

    recognize_dir("./data/face/test/cxk", face_database)

    recognize_dir("./data/face/test/trump", face_database)
    recognize_dir("./data/face/test/dingzhen", face_database)
