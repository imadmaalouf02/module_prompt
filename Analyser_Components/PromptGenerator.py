from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
import os

class PromptGenerator:
    def __init__(self, image_directory="./data/", images_extensions=['jpg', 'jpeg', 'png'], 
                 model_name="nlpconnect/vit-gpt2-image-captioning", device=None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else "cpu")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.image_directory = image_directory
        self.images_extensions = images_extensions
        self.prompts_dict = {}

    def _load_images(self):
        images = []
        image_names = []
        
        for image_name in os.listdir(self.image_directory):
            if image_name.split('.')[-1].lower() in self.images_extensions:
                image_path = os.path.join(self.image_directory, image_name)
                try:
                    i_image = Image.open(image_path)
                    if i_image.mode != "RGB":
                        i_image = i_image.convert(mode="RGB")
                    images.append(i_image)
                    image_names.append(image_name)
                except Exception as e:
                    print(f"Error opening image {image_name}: {e}")
                    continue
                    
        return images, image_names

    def generate_prompts(self, max_length=15, num_prompts=20):
        images, image_names = self._load_images()

        if not images:
            print("No valid images found!")
            return

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values

        for idx, image_name in enumerate(image_names):
            image_pixel_values = pixel_values[idx].unsqueeze(0).to(self.device)
            prompts = []

            for _ in range(num_prompts):
                output_ids = self.model.generate(
                    image_pixel_values,
                    max_length=max_length,
                    num_beams=1,  # Disable beam search
                    do_sample=True,  # Enable sampling
                    top_k=50,  # Top-k sampling
                    temperature=1.0  # Sampling temperature
                )
                pred = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
                prompts.append(pred)

            self.prompts_dict[image_name] = prompts
            print(f"Generated {len(prompts)} prompts for {image_name}")

        return self.prompts_dict
