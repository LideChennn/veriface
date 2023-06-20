import cv2
import os


def object_detect(
        classifier_file='./data/haarcascades/haarcascade_frontalface_default.xml',
        img_dir='./data/images',
        scale_factor=1.1,
        min_neighbors=5
):
    face_cascade = cv2.CascadeClassifier(classifier_file)  # 分类器
    imgs = cv2_imread_imgs(img_dir)  # 使用cv2读取一个目录或者一张图片

    faces = []  # faces 人脸矩形坐标的列表。每个矩形由 4 个整数值表示：(x, y, w, h),其中 (x, y) 是左上角的坐标，w 和 h 分别是矩形的宽度和高度。
    cropped_faces = []  # 裁剪后的脸部图片

    # 对每一张获取灰度图片，再用灰度图片进行人脸检测
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转化为灰度图像，不影响目标检测
        face = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)  # 人脸检测
        faces.append(face)

    for img, face in zip(imgs, faces):  # 使用 zip() 函数同时遍历 imgs 和 faces 列表
        for (x, y, w, h) in face:  # 一张图片可能有多个人脸
            cropped_face = img[y:y + h, x:x + w]  # 切片，取矩形区域
            cropped_faces.append(cropped_face)
    return cropped_faces


# 使用cv2读取一个目录或者一张图片
def cv2_imread_imgs(path):
    images = []
    if not os.path.exists(path):
        print("Error: Path does not exist.")
        return images
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    elif os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path)
                if img is not None:
                    images.append(img)
    return images


def resize_faces(cropped_faces, size=(160, 160)):
    resized_faces = []

    for face in cropped_faces:
        resized_face = cv2.resize(face, size, interpolation=cv2.INTER_AREA)
        resized_faces.append(resized_face)

    return resized_faces
