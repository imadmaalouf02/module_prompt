from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
from groundingdino import GroundingDINOProcessor, GroundingDINOModel  # Assuming you're using Grounding DINO
from segment_anything import SamProcessor, SamModel  # Assuming you're using SAM

class ObjectDetector:
    def __init__(self):
        # Load the Grounding DINO for zero-shot object detection
        self.grounding_dino_model = GroundingDINOModel.from_pretrained("ShilongLiu/GroundingDINO")
        self.grounding_dino_processor = GroundingDINOProcessor.from_pretrained("ShilongLiu/GroundingDINO")

        # Load SAM (Segment Anything) model for segmentation
        self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge")
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grounding_dino_model.to(self.device)
        self.sam_model.to(self.device)

    def detect_objects(self, image_path, prompts):
        """
        Detect objects in an image based on the provided zero-shot prompts using Grounding DINO.
        """
        detected_objects = {}
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Process each prompt
        for prompt in prompts:
            # Grounding DINO: Detect objects based on the textual prompt
            dino_inputs = self.grounding_dino_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            dino_outputs = self.grounding_dino_model(**dino_inputs)

            # Post-process the outputs to get bounding boxes and labels
            boxes, labels = self.grounding_dino_processor.post_process(dino_outputs)

            # SAM: Use SAM to further segment the detected objects (optional, depending on the application)
            sam_inputs = self.sam_processor(images=image, boxes=boxes, return_tensors="pt").to(self.device)
            sam_outputs = self.sam_model(**sam_inputs)

            # Store the detected labels and boxes
            detected_objects[prompt] = {"labels": labels, "boxes": boxes.cpu().numpy()}

        return detected_objects

    def extract_classes_from_prompts(self, prompts_dict):
        """
        Given a dictionary of prompts and corresponding images, extract detected object class names using Grounding DINO.
        """
        extracted_classes = {}

        for image_name, prompts_list in prompts_dict.items():
            image_classes = []
            for prompt in prompts_list:
                # Detect objects for each prompt using zero-shot learning
                detected_objects = self.detect_objects(image_name, [prompt])
                labels = detected_objects[prompt]["labels"]

                # Convert label IDs to class names
                class_names = [self.grounding_dino_processor.convert_label_to_text(label) for label in labels]
                image_classes.append(", ".join(class_names))

            extracted_classes[image_name] = image_classes

        return extracted_classes
