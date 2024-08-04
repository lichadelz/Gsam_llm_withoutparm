import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import ssl
DEVICE = 'cuda'
ssl._create_default_https_context = ssl._create_unverified_context
best_model = torch.load('./best_model.pth')
new_size=(640, 640)
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
def metal_inference(mask_path,image_name):
    image_path=mask_path+image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(image.shape)
    image = cv2.resize(image, new_size)
    resized_image = np.transpose(image, (2, 0, 1)).astype(np.float32)/255
    print(resized_image.shape)
    x_tensor = torch.from_numpy(resized_image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())
    print("pr_mask.shape=",pr_mask.shape)
    masked_image = np.where(np.repeat(np.expand_dims(pr_mask, axis=-1), 3, axis=-1), image, 0)

    # 将 pr_mask 转换为 uint8 类型，前景为 255，背景为 0
    pr_mask_uint8 = (pr_mask * 255).astype(np.uint8)

    # 寻找轮廓
    contours, _ = cv2.findContours(pr_mask_uint8, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 绘制轮廓
    cv2.drawContours(masked_image, contours, -1, (0,0,255), 3)
    # # 显示或保存结果图像
    # cv2.imshow('Masked Image', masked_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # 或者保存图像
    metal_seg_path=mask_path+'metal_seg/'+image_name
    cv2.imwrite(metal_seg_path, masked_image)
    # visualize(
    #     image=image, 
    #     predicted_mask=pr_mask
    # )
    
    return pr_mask,contours
def metal_seg(metal_indices):
    metal_contours=[]
    mask=[]
    for i in metal_indices:
        image_name='image_'+str(i)+'.png'
        mask_path='/home/lichalab/GSam_llm/data_mask/'
        contours,pr_mask=metal_inference(mask_path,image_name)
        metal_contours.append(contours)
        mask.append(pr_mask)
    return metal_contours,mask
def suction_inference(mask_path,image_name):
    image_path=mask_path+image_name
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, new_size)
   # 调整图片尺寸为640x640
    image = cv2.resize(image, (640, 640))
    
    # 将非黑色像素转换为1，黑色像素保持为0
    pr_mask = (image != [0, 0, 0]).any(axis=-1).astype(np.uint8)
    
    # 打印转换后的矩阵形状
    print(pr_mask.shape)
    
    return pr_mask


if __name__ == "__main__":
    image_name="image_0.png"
    mask_path='/home/lichalab/GSam_llm/data_mask/'
    suction_inference(mask_path,image_name)
