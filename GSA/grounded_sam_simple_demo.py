import cv2
import numpy as np
import supervision as sv
import os
import sys
sys.path.append('/home/lichalab/GSam_llm/GSA')
sys.path.append('/home/lichalab/GSam_llm/GSA/GroundingDINO')
sys.path.append('/home/lichalab/GSam_llm/GSA/segment_anything')
import torch
import torchvision
from PIL import Image
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GSA/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./GSA/groundingdino_swinb_cogcoor.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./GSA/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

def ground_sam(num_jpg,obj_list):
    # Predict classes and hyper-param for GroundingDINO
    SAVE_PATH="/home/lichalab/GSam_llm/data_rbg/"
    SOURCE_IMAGE_PATH = SAVE_PATH+num_jpg
    CLASSES =  obj_list
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.3
    NMS_THRESHOLD = 0.8

    # load image
    image = cv2.imread(SOURCE_IMAGE_PATH)

    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=CLASSES,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    # print("labels=",labels)

    annotated_frame = box_annotator.annotate(scene=image.copy(), detections=detections, labels=labels)


    # save the annotated grounding dino image
    cv2.imwrite("/home/lichalab/GSam_llm/data_seg/detect_image.jpg", annotated_frame)

    # NMS post process
    print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        NMS_THRESHOLD
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    print(f"After NMS: {len(detections.xyxy)} boxes")


    # convert detections to masks
    detections.mask = segment(
        sam_predictor=sam_predictor,
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        xyxy=detections.xyxy
    )

    # annotate image with detections
    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()
    labels = [
        f"{CLASSES[class_id]} {confidence:0.2f}" 
        for _, _, confidence, class_id, _, _ 
        in detections]
    print("labels=",labels)
    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

    # save the annotated grounded-sam image
    print(detections.mask.shape)
    # 将布尔值转换为整数，True变为255，False变为0
    mask_uint8 = np.where(detections.mask, 255, 0).astype(np.uint8)
    image_gray = Image.fromarray(mask_uint8[0])  
    contour_image = np.zeros_like(image_gray)
    contour_image = cv2.cvtColor(contour_image, cv2.COLOR_GRAY2BGR)
    contours_all = []
    # 由于mask是三维的，我们可能需要循环处理每一层
    for i in range(detections.mask.shape[0]):
        # 将当前层转换为图像
        # image_gray = Image.fromarray(mask_uint8[i])
        mask = mask_uint8[i]
        masked_image = np.where(mask[..., None].astype(bool), image, 0)

        blurred = cv2.GaussianBlur(mask_uint8[i], (5, 5), 0)
        # 使用Canny边缘检测算法
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 由于 contours 是一个元组，我们首先将其转换为列表
        contours_all.append(contours)
        # 绘制轮廓
        random_color = get_random_color()
        cv2.drawContours(contour_image, contours, -1, random_color, 3)
        cv2.imwrite(f'/home/lichalab/GSam_llm/data_mask/image_{i}.png', masked_image)
        
    # 保存或显示图像
    cv2.imwrite('/home/lichalab/GSam_llm/data_mask/contour_image.png', contour_image)
    cv2.imwrite("/home/lichalab/GSam_llm/data_seg/seg_image.jpg", annotated_image)
    return contours_all
def get_random_color():
    return (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)
if __name__ == "__main__":
    ground_sam()