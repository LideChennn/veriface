import os
import cv2
import tkinter as tk
from tkinter import messagebox
import torch
from PIL import Image
from torchvision.transforms import transforms
from model.resnet import FaceNetModel
from util.call_phone import capture_faces, recognize_faces

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的模型
model_path = "./weight/facenet_checkpoint11_0.00033.pth"
facenet = FaceNetModel().to(device)
facenet.load_state_dict(torch.load(model_path))

facenet.eval()
# 计算阈值，或从训练过程中确定
threshold = 1.0


def preprocess(image_path):
    img = cv2.imread(image_path)
    pil_image = Image.fromarray(img)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(pil_image).unsqueeze(0)


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


face_database = load_face_database('./data/realTime/database')


# 将 capture_faces 函数定义放在此处
def start_capture():
    name = name_entry.get()
    if not name:
        messagebox.showerror("Error", "名字不为空.")
        return
    file_prefix = f"{name}_"
    capture_faces(name, file_prefix)

    # 更新 face_database
    face_database_path = './data/realTime/database'
    person_folder = os.path.join(face_database_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # 处理路径得到图片张量
        image_tensor = preprocess(image_path).to(device)
        with torch.no_grad():
            embedding = facenet(image_tensor).cpu().numpy()

        if name not in face_database:
            face_database[name] = []

        face_database[name].append(embedding)


def update_face_database(name):
    # 更新 face_database
    face_database_path = './data/realTime/database'
    person_folder = os.path.join(face_database_path, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        # 处理路径得到图片张量
        image_tensor = preprocess(image_path).to(device)
        with torch.no_grad():
            embedding = facenet(image_tensor).cpu().numpy()

        if name not in face_database:
            face_database[name] = []

        face_database[name].append(embedding)


def start_recognition():
    if not face_database:
        messagebox.showerror("Error", "人脸数据库为空.")
        return

    recognize_faces(facenet, face_database, threshold, device)


if __name__ == '__main__':
    # 创建主窗口
    root = tk.Tk()
    root.title("人脸识别")

    # 创建标签、文本框和按钮
    name_label = tk.Label(root, text="输入名字:")
    name_entry = tk.Entry(root)
    capture_button = tk.Button(root, text="录入人脸", command=start_capture)
    recognize_button = tk.Button(root, text="人脸识别", command=start_recognition)

    # 将组件添加到主窗口
    name_label.grid(row=0, column=0, padx=10, pady=10)
    name_entry.grid(row=0, column=1, padx=10, pady=10)
    capture_button.grid(row=1, column=0, padx=10, pady=10)
    recognize_button.grid(row=1, column=1, padx=10, pady=0)
    # 运行 Tkinter 事件循环
    root.mainloop()
