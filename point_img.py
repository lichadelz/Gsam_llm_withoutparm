import cv2
import numpy as np
#图片路径
img = cv2.imread('/home/lichalab/GSam_llm/data_mask/metal_seg/image_0.png')
a =[]
b = []
def on_EVENT_LBUTTONDOWN(event, x, y,flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "(%d,%d,%d)" % (img[y][x][0], img[y][x][1], img[y][x][2])
        a.append(x)
        b.append(y)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 255), thickness=1)
        cv2.imshow("image", img)
 
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)