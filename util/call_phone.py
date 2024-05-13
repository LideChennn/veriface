import os
import numpy as np
import cv2
import cv2 as cv
import torch
from PIL import Image
from torchvision.transforms import transforms


def preprocess(img):
    pil_image = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(pil_image).unsqueeze(0)


def delete_or_create_folder(folder_path):
    if os.path.exists(folder_path):
        # 文件夹存在，删除文件夹和文件夹下的所有文件
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for file in files:
                file_path = os.path.join(root, file)
                os.remove(file_path)
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                os.rmdir(dir_path)
        os.rmdir(folder_path)
        print(f"文件夹 '{folder_path}' 已删除。")
    # 文件夹不存在，创建文件夹
    os.makedirs(folder_path)
    print(f"文件夹 '{folder_path}' 已创建。")


def rotated_img(img, angle=90):
    # 获取图像的尺寸
    height, width = img.shape[:2]
    # 计算旋转的中心点q
    center = (width // 2, height // 2)
    # 获取旋转矩阵
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    # 应用旋转矩阵
    rotated_frame = cv.warpAffine(img, rotation_matrix, (width, height))
    return rotated_frame


# 人脸录入
def capture_faces(people_name, file_prefix='cropped_face_'):
    # 如果存在，就重新保存文件
    delete_or_create_folder(f"./data/realTime/database/{people_name}")

    # 读取视频信息。
    cap = cv.VideoCapture("http://admin:admin@10.134.43.118:8081/")  # @前为账号密码，@后为ip地址
    face_xml = cv.CascadeClassifier("./data/haarcascades/haarcascade_frontalface_default.xml")  # 导入XML文件

    frame_count = 0  # 添加一个帧计数器

    # 对每一帧处理
    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧图片
        if not ret:
            break

        frame_count += 1
        # 旋转图片
        rotated_frame = rotated_img(img)

        gray = cv.cvtColor(rotated_frame, cv.COLOR_BGR2GRAY)  # 转换为灰度图
        face = face_xml.detectMultiScale(gray, 1.1, 5)  # 检测人脸，并返回人脸位置信息
        cropped_face = None

        for (x, y, w, h) in face:
            cv.rectangle(rotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_face = rotated_frame[y:y + h, x:x + w]
            if frame_count % 10 == 0 and cropped_face is not None:  # 每10帧保存一张图片
                cv.imwrite(f"./data/realTime/database/{people_name}/{file_prefix}{frame_count}.jpg", cropped_face)
                print(f"Saved cropped_face_{frame_count}.jpg")

        cv.imshow("1", rotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# 人脸识别
def recognize_faces(facenet, face_database, threshold, device):
    # 读取视频信息。
    cap = cv.VideoCapture("http://admin:admin@10.134.43.118:8081/")  # @前为账号密码，@后为ip地址
    face_xml = cv.CascadeClassifier("./data/haarcascades/haarcascade_frontalface_default.xml")  # 导入XML文件

    frame_count = 0  # 添加一个帧计数器

    while cap.isOpened():
        ret, img = cap.read()  # 读取一帧图片
        if not ret:
            break

        frame_count += 1
        # 旋转图片
        rotated_frame = rotated_img(img)

        gray = cv.cvtColor(rotated_frame, cv.COLOR_BGR2GRAY)  # 转换为灰度图
        face = face_xml.detectMultiScale(gray, 1.1, 5)  # 检测人脸，并返回人脸位置信息

        for (x, y, w, h) in face:
            cv.rectangle(rotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cropped_face = rotated_frame[y:y + h, x:x + w]

            if cropped_face is not None:
                closest_name, closest_distance = \
                    recognize_face(cropped_face, face_database, threshold, facenet, device)
                # 在识别框旁边标注文本
                text = f"{closest_name}: {closest_distance:.4f}"
                cv.putText(rotated_frame, text, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv.imshow("1", rotated_frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# 根据阈值判断待识别图像与人脸库中哪个人脸最接近
def recognize_face(test_image, face_database, threshold, facenet, device):
    test_image_tensor = preprocess(test_image).to(device)

    # 提取待识别图像的特征向量
    with torch.no_grad():
        test_embedding = facenet(test_image_tensor).cpu().numpy()

    distances = compute_distances(test_embedding, face_database)

    # (cxk, 1), (trump, 2)
    closest_name, closest_distance = min(distances.items(), key=lambda x: x[1])

    if closest_distance <= threshold:
        print("识别结果: {},距离: {:.4f}".format(closest_name, closest_distance))
    else:
        print("未识别到匹配的人脸")
    return closest_name, closest_distance


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


# 计算两个向量之间的欧氏距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)
