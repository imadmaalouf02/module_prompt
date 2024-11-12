from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

class ObjectDetector:
    def __init__(self):
        # Load pre-trained DETR model and processor
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_classes_from_prompt(self, prompt):
        """
        Given a prompt (text), this function will extract object class names from the text.
        """
        # Placeholder for object names instead of class IDs
        class_labels = {
            "dog": "dog",
            "cat": "cat",
            "person": "person",
        }
        
        detected_classes = []
        
        # Check for object names in the prompt and return corresponding class names
        for obj, label in class_labels.items():
            if obj in prompt.lower():  # This checks for the presence of specific objects in the prompt
                detected_classes.append(label)
        
        # Join the detected classes into a string separated by commas
        return ", ".join(detected_classes)

    def extract_classes_from_prompts(self, prompts_dict):
        """
        Given a dictionary of prompts, extract the detected object class names for each prompt.
        """
        extracted_classes = {}
        
        for image_name, prompts_list in prompts_dict.items():
            image_classes = []
            for prompt in prompts_list:
                # Extract detected classes for each prompt
                detected_classes = self.extract_classes_from_prompt(prompt)
                image_classes.append(detected_classes)
            
            extracted_classes[image_name] = image_classes

        return extracted_classes
