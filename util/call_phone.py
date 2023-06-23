import cv2 as cv

# 读取视频信息。
cap = cv.VideoCapture("http://admin:admin@10.134.43.118:8081/")  # @前为账号密码，@后为ip地址
face_xml = cv.CascadeClassifier("../data/haarcascades/haarcascade_frontalface_default.xml")  # 导入XML文件


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


while cap.isOpened():
    ret, img = cap.read()  # 读取一帧图片
    if not ret:
        break

    # 旋转图片
    rotated_frame = rotated_img(img)

    # 在图像上添加文字
    font = cv.FONT_HERSHEY_SIMPLEX  # 定义字体类型
    text = "Please adjust your head position."  # 要显示的文本
    position = (50, 50)  # 文本显示的位置（左下角坐标）
    font_scale = 1  # 字体缩放比例
    font_color = (0, 255, 0)  # 字体颜色（BGR）
    line_thickness = 2  # 文字线条粗细

    cv.putText(rotated_frame, text, position, font, font_scale, font_color, line_thickness)

    gray = cv.cvtColor(rotated_frame, cv.COLOR_BGR2GRAY)  # 转换为灰度图
    face = face_xml.detectMultiScale(gray, 1.1, 5)  # 检测人脸，并返回人脸位置信息

    for (x, y, w, h) in face:
        cv.rectangle(rotated_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv.imshow("1", rotated_frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
