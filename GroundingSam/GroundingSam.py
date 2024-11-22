import supervision as sv
import os
import torch
from typing import List
import cv2
from tqdm.notebook import tqdm
from .groundingdino.util.inference import Model
import json
import xml.etree.ElementTree as ET


import numpy as np
from .segment_anything import sam_model_registry, SamPredictor

# Global Variables
GROUNDING_DINO_CONFIG_PATH = "/content/module_prompt/GroundingSam/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/content/module_prompt/GroundingSam/weights/groundingdino_swint_ogc.pth"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert(os.path.isfile(GROUNDING_DINO_CONFIG_PATH)), "GroundingDINO config file not found!"
assert(os.path.isfile(GROUNDING_DINO_CHECKPOINT_PATH)), "GroundingDINO checkpoint file not found!"
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/content/module_prompt/GroundingSam/weights/sam_vit_h_4b8939.pth"
assert(os.path.isfile(SAM_CHECKPOINT_PATH)), "SAM checkpoint file not found!"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)

# Function to enhance class names and get
def enhance_class_name(class_names: List[str]) -> List[str]:
  return [
      f"all {class_name}s"
      for class_name
      in class_names
  ]

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


class GroundingSam:
  def __init__(self, classes, images_dir = "./data/", annotations_dir = "./annotations/", images_extensions = ['jpg', 'jpeg', 'png']):
    self.classes = classes
    self.images_dir = images_dir
    self.annotations_dir = annotations_dir
    self.images_extensions = images_extensions

    self.image_paths = sv.list_files_with_extensions(
      directory=self.images_dir,
      extensions=self.images_extensions)

    self.detections = None
    self.images = {}
    self.annotations = {}

  def get_detections(self, BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
    for image_path in tqdm(self.image_paths):
      image_name = image_path.name
      image_path = str(image_path)
      image = cv2.imread(image_path)

      self.detections = grounding_dino_model.predict_with_classes(
          image=image,
          classes=enhance_class_name(class_names=self.classes),
          box_threshold=BOX_TRESHOLD,
          text_threshold=TEXT_TRESHOLD)
      
      self.detections = self.detections[self.detections.class_id != None] 
      self.images[image_name] = image
      self.annotations[image_name] = self.detections
    return self.detections


  def get_detection_data(self):
        # Create a structured dictionary based on `self.annotations`
        detection_data = {}
        
        for image_name, detections in self.annotations.items():
            labels = [self.classes[class_id] for class_id in detections.class_id]
            bounding_boxes = detections.xyxy.tolist()  # Convert bounding boxes to a list format
            
            detection_data[image_name] = {
                "labels": labels,
                "bounding_boxes": bounding_boxes
            }
        
        return detection_data

        
  def annotate_images(self):
    plot_images = []
    plot_titles = []

    box_annotator = sv.BoxAnnotator()

    for image_name, self.detections in self.annotations.items():
      image = self.images[image_name]
      plot_images.append(image)
      plot_titles.append(image_name)

      labels = [
          f"{self.classes[class_id]} {confidence:0.2f}"
          for _, _, confidence, class_id, _
          in self.detections]
      annotated_image = box_annotator.annotate(scene=image.copy(), detections=self.detections, labels=labels)
      plot_images.append(annotated_image)
      title = " ".join(set([
          self.classes[class_id]
          for class_id
          in self.detections.class_id
      ]))
      plot_titles.append(title)

    sv.plot_images_grid(
        images=plot_images,
        titles=plot_titles,
        grid_size=(len(self.annotations), 2),
        size=(2 * 4, len(self.annotations) * 4)
    )
    
    
  def get_masks(self,BOX_TRESHOLD=0.35, TEXT_TRESHOLD=0.25, class_enhancer=enhance_class_name):
    images = {}
    annotations = {}

    for image_path in tqdm(self.image_paths):
        image_name = image_path.name
        image_path = str(image_path)
        image = cv2.imread(image_path)

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=self.classes),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        detections = detections[detections.class_id != None]
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        images[image_name] = image
        annotations[image_name] = detections
    

    plot_images = []
    plot_titles = []

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    for image_name, detections in annotations.items():
      image = images[image_name]
      plot_images.append(image)
      plot_titles.append(image_name)

      labels = [
          f"{self.classes[class_id]} {confidence:0.2f}" 
          for _, _, confidence, class_id, _ 
          in detections]
      annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
      annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
      plot_images.append(annotated_image)
      title = " ".join(set([
          self.classes[class_id]
          for class_id
          in detections.class_id
      ]))
      plot_titles.append(title)

    sv.plot_images_grid(
        images=plot_images,
        titles=plot_titles,
        grid_size=(len(annotations), 2),
        size=(2 * 4, len(annotations) * 4)
        )
    
    
  def detect_all_objects(self, prompts: list, BOX_THRESHOLD=0.35, TEXT_THRESHOLD=0.25):
      """
      Détecte tous les objets dans les images en utilisant une liste de classes données.
      
      :param prompts: Une liste de classes à utiliser pour la détection (par exemple, ["guitar", "piano", "person"]).
      :param BOX_THRESHOLD: Seuil pour la détection des boîtes.
      :param TEXT_THRESHOLD: Seuil pour la détection des objets par texte.
      :return: Un dictionnaire contenant les objets détectés pour chaque image.
      """
      images = {}
      annotations = {}
      detected_objects = {}  # Dictionnaire pour stocker les objets détectés
  
      for image_path in tqdm(self.image_paths):
          image_name = image_path.name
          image_path = str(image_path)
          image = cv2.imread(image_path)
  
          # Utiliser les prompts pour la détection
          detections = grounding_dino_model.predict_with_classes(
              image=image,
              classes=prompts,  # Utiliser la liste de classes comme prompt
              box_threshold=BOX_THRESHOLD,
              text_threshold=TEXT_THRESHOLD
          )
          
          # Filtrer les détections valides
          detections = detections[detections.class_id != None]
          detections.mask = segment(
              sam_predictor=sam_predictor,
              image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
              xyxy=detections.xyxy
          )
          
          images[image_name] = image
          annotations[image_name] = detections
          
          # Ajouter les objets détectés à la liste
          detected_objects[image_name] = [
              {
                  "label": self.classes[class_id],
                  "confidence": confidence,
                  "bounding_box": bounding_box
              }
              for _, _, confidence, class_id, bounding_box in detections
          ]
  
      # Annoter et afficher les images détectées
      
  
      return detected_objects  
  


  def annotate_images_with_prompt(self, images: dict, annotations: dict):
        plot_images = []
        plot_titles = []

        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()

        for image_name, detections in annotations.items():
            image = images[image_name]
            plot_images.append(image)
            plot_titles.append(image_name)

            labels = [
                f"{self.classes[class_id]} {confidence:0.2f}" 
                for _, _, confidence, class_id, _ 
                in detections
            ]
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)
            plot_images.append(annotated_image)
            title = " ".join(set([
                self.classes[class_id]
                for class_id
                in detections.class_id
            ]))
            plot_titles.append(title)

        sv.plot_images_grid(
            images=plot_images,
            titles=plot_titles,
            grid_size=(len(annotations), 2),
            size=(2 * 4, len(annotations) * 4)
        )

  def save_new_annotations(self, output_path, approximation_percentage=0.75):
        # Initialize the COCO data structure
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": []
        }
    
        # Add new classes to categories
        coco_data["categories"] = [
            {"id": int(idx), "name": class_name, "supercategory": "none"}
            for idx, class_name in enumerate(self.new_classes)
        ]
    
        annotation_id = 0
    
        # Iterate over images and annotations
        for image_id, (image_name, detections) in enumerate(self.annotations.items()):
            image = self.images[image_name]
            height, width, _ = image.shape
    
            # Add image metadata
            coco_data["images"].append({
                "id": int(image_id),
                "file_name": image_name,
                "width": int(width),
                "height": int(height)
            })
    
            # Iterate over detections and filter to only include new classes
            for bbox, mask, class_id, confidence in zip(
                detections.xyxy, detections.mask, detections.class_id, detections.confidence
            ):
                if class_id >= len(self.new_classes):  # Skip classes not in new_classes
                    continue
    
                # Convert bounding box to COCO format [x, y, width, height]
                x_min, y_min, x_max, y_max = bbox
                coco_bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
    
                # Add annotation
                coco_data["annotations"].append({
                    "id": int(annotation_id),
                    "image_id": int(image_id),
                    "category_id": int(class_id),
                    "bbox": coco_bbox,
                    "area": float((x_max - x_min) * (y_max - y_min)),
                    "iscrowd": 0,
                    "segmentation": self._get_segmentation_from_mask(mask, approximation_percentage)
                })
                annotation_id += 1
    
        # Save the COCO annotations to a JSON file
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=4)
    
        print(f"Annotations with new classes saved to {output_path}")
