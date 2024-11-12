import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
from groundingdino.util.inference import load_model, predict
from PIL import Image

class ObjectDetector:
    def __init__(self):
        # Load pre-trained Grounding DINO model for zero-shot object detection
        self.model = load_model("GroundingDINO_SwinT_OGC.pth")  # Assuming model is locally available or can be loaded
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def extract_classes_from_prompt(self, prompt, image_path):
        """
        Detect objects in an image using Grounding DINO based on a text prompt.
        :param prompt: Text prompt to guide zero-shot object detection
        :param image_path: Path to the image file
        :return: List of detected objects as class names (comma-separated)
        """
        # Load the image
        image = Image.open(image_path)

        # Run prediction using Grounding DINO
        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,  # This is the textual prompt for zero-shot detection
            box_threshold=0.3,  # Threshold for box detection
            text_threshold=0.25  # Threshold for text detection
        )

        # Extract detected class phrases (object names) from the results
        detected_classes = [phrase for phrase in phrases]
        
        # Join the detected classes into a string separated by commas
        return ", ".join(detected_classes)

    def extract_classes_from_prompts(self, prompts_dict):
        """
        Extract object classes from images based on prompts using Grounding DINO for zero-shot detection.
        :param prompts_dict: Dictionary where keys are image paths and values are lists of text prompts
        :return: Dictionary of image paths and detected object classes (comma-separated) for each prompt
        """
        extracted_classes = {}
        
        for image_name, prompts_list in prompts_dict.items():
            image_classes = []
            for prompt in prompts_list:
                # Extract detected classes for each prompt
                detected_classes = self.extract_classes_from_prompt(prompt, image_name)
                image_classes.append(detected_classes)
            
            extracted_classes[image_name] = image_classes

        return extracted_classes
