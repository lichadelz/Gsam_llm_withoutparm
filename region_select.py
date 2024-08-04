import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import math
import further_seg
PI=3.1415926
def point_select(mask):
    # 假设 mask 是一个 640x640 的二值矩阵
    #采样比例
    area=area_region(mask,1)
    p=0.02
    radius=np.sqrt(area/PI)
    ratio_radius=0.8
    print(radius)
    region=mask.copy()
    n=2
    for i in range(n):
        sample_points=sample_points_in_region(region, p,i+1)
        top_points=top_points_(sample_points,mask.copy(),radius)
        top_points_only = [point[0] for point in top_points]
        region=region_iter(top_points_only,region,i+2)
        radius=radius*ratio_radius
    final_point = max(top_points, key=lambda x: x[1])[0]


    img_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    img_rgb[mask == 1] = [255, 255, 255]  # 将 mask 中的 1 设置为白色，以便在 RGB 图像中可见
    for point in sample_points:
        cv2.circle(img_rgb, (point[1], point[0]), 1, (0, 0, 255), -1)  # 注意坐标顺序
    for point in top_points_only:
        cv2.circle(img_rgb, (point[1], point[0]), 1, (0, 255, 0), -1)  # 注意坐标顺序
    cv2.circle(img_rgb, (final_point[1], final_point[0]), 1, (255, 0, 0), -1)  # 注意坐标顺序

    cv2.imshow('Masked Image', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def area_region(region, i):
    # 计算region中等于i的元素的数量
    area= 0
    for row in region:  # 遍历每一行
        for element in row:  # 遍历行中的每一个元素
            if element == i:
                area += 1
    return area
def region_iter(top_points_only,region,i):
    min_x=1e6
    min_y=1e6
    max_x=-1
    max_y=-1
    for point in top_points_only:
        x, y = point
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    # 计算长宽，并按比例扩大
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    expand_width = int(width * 0.1)
    expand_height = int(height * 0.1)

    # 更新边界值
    min_x = max(0, min_x - expand_width)
    max_x = min(region.shape[0] - 1, max_x + expand_width)
    min_y = max(0, min_y - expand_height)
    max_y = min(region.shape[1] - 1, max_y + expand_height)

    # 在 mask 上建立矩形区域
    region[min_x:max_x+1, min_y:max_y+1][region[min_x:max_x+1, min_y:max_y+1] != 0] = i  # 将矩形区域内的值设为2以区分

    return region

def top_points_(sample_points,mask,radius):
    points_with_counts=[]
    for point in sample_points:
        contain_points=detect(mask,point,radius)
        points_with_counts.append((point,contain_points))
    sorted_points = sorted(points_with_counts, key=lambda x: x[1], reverse=True)

    # 计算前%的索引
    top_index = int(len(sorted_points) * 0.2)
    #  取前%
    top_points = sorted_points[:top_index]

    return top_points
def sample_points_in_region(region, p,i):
    area= area_region(region, i)
    num_samples=round(area*p)
    if num_samples>80:
        num_samples=80
    # 找到所有前景像素的位置
    indices = np.where(region == i)
    # 将这些位置组合成一个二维数组
    points_array = np.column_stack(indices)
    
    # 随机选择num_samples个点
    if len(points_array) >= num_samples:
        sampled_indices = np.random.choice(len(points_array), num_samples, replace=False)
        sampled_points = points_array[sampled_indices]
    else:
        # 如果前景像素的数量少于要求的样本数，则返回所有前景像素的位置
        sampled_points = points_array
    
    # 将结果转换为列表
    points = [tuple(point) for point in sampled_points]
    
    return points
def detect(mask,point,radius):
    # 假设 region 是一个二维列表，其中每个元素可以是任意数值
    # point 是一个包含两个元素的列表或元组，表示中心点的坐标 [x, y]
    # radius 是一个数字，表示正方形的半边长
    x_center, y_center = point
    contain_points = 0
    
    for y in range(max(0, int(y_center - radius)), min(mask.shape[1], int(y_center + radius + 1))):
        for x in range(max(0, int(x_center - radius)), min(mask.shape[0], int(x_center + radius + 1))):
            # if (x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2:
            if mask[x][y] != 0:
                contain_points += 1
                
    return contain_points

if __name__ == "__main__":
    image_name="image_0.png"
    mask_path='/home/lichalab/GSam_llm/data_mask/'
    pr_mask=further_seg.suction_inference(mask_path,image_name)
    print(pr_mask)
    point_select(pr_mask)