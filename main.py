import cv2
import util.object_detect as bd

imgs = bd.resize_faces(bd.object_detect(img_dir='./data/images'))

for i in range(len(imgs)):
    cv2.imshow("{}".format(i), imgs[i])

cv2.waitKey(0)
cv2.destroyAllWindows()
