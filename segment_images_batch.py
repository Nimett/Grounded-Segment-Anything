from argparse import ArgumentParser
from pathlib import Path
import glob
import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "./groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)

BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


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


def segment_images_batch(image_folder, output_folder, seg_classes, extension) -> None:
    image_paths = glob.glob(f"{image_folder}/*.{extension}")

    for image_path in image_paths:
        img_name = Path(image_path).stem
        img = cv2.imread(image_path)

        detections = grounding_dino_model.predict_with_classes(
            image=img,
            classes=seg_classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        labels = [
            f"{seg_classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections, labels=labels)

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
            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        labels = [
            f"{seg_classes[class_id]} {confidence:0.2f}" 
            for _, _, confidence, class_id, _, _ 
            in detections]
        annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
        annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save annotated grounded-sam image
        cv2.imwrite(f"{output_folder}/{img_name}.jpg", annotated_image)
        np.save(f"{output_folder}/{img_name}.mask", detections.mask)
        np.save(f"{output_folder}/{img_name}.label", labels)



if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--image_folder", type=str)
    args.add_argument("--output_folder", type=str)
    args.add_argument("--image_extension", type=str, default="png")
    args.add_argument("--seg_classes", type=str, default="High-standing platforms, Ground, Humans", help="Comma seperated object list")
    args = args.parse_args()
    seg_classes = args.seg_classes.strip().split(",")
    segment_images_batch(args.image_folder, args.output_folder, seg_classes, args.image_extension)