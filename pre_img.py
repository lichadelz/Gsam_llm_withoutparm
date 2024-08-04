import cv2
import numpy as np

# 鼠标回调函数
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # 获取点击位置的像素值
        pixel_value = param[y, x]
        print(f"Clicked at pixel: ({x}, {y}) with color value: {pixel_value}")

# 读取图片
img_path = 'data_seg/seg_image.jpg'  # 替换为你的图片路径
image = cv2.imread(img_path)
cv2.namedWindow('image')

# 设置鼠标回调函数
cv2.setMouseCallback('image', mouse_click, image)

while True:
    # 显示图片
    cv2.imshow('image', image)
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()