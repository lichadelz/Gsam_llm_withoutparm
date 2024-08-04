import cv2
import numpy as np

# 读取图像
image = cv2.imread('/home/lichalab/GSam_llm/data_mask/image_1.png')

# 将图像转换为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊来减少图像噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny边缘检测算法
edges = cv2.Canny(blurred, 50, 150)

# 查找轮廓
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
contour_image = np.zeros_like(image)
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 3)

# 显示原始图像和轮廓图像
cv2.imshow('Original Image', image)
cv2.imshow('Contours', contour_image)

# 等待按键后关闭所有窗口
cv2.waitKey(0)
cv2.destroyAllWindows()