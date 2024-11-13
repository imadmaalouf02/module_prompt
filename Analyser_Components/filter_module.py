import torch
from groundingdino.models import GroundingDINO
from groundingdino.utils import get_tokenizer
from groundingdino.utils.dino_utils import get_obj_from_config
from PIL import Image
import torchvision.transforms as T

class ObjectDetectorDino:
    def __init__(self, config_path, checkpoint_path):
        # Spécifier les chemins du modèle et de la configuration
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        # Charger la configuration et le modèle
        self.model = self.load_model()

        # Charger le tokenizer
        self.tokenizer = get_tokenizer()

    def load_model(self):
        # Charger la configuration du modèle
        config = get_obj_from_config(self.config_path)
        
        # Créer une instance du modèle GroundingDINO
        model = GroundingDINO(config)
        
        # Charger les poids du modèle pré-entraîné
        checkpoint = torch.load(self.checkpoint_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(checkpoint['model'])
        
        # Déplacer le modèle sur GPU ou CPU
        model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        return model

    def detect_objects(self, image_path, prompt):
        # Ouvrir l'image et prétraiter
        image = Image.open(image_path).convert("RGB")
        transform = T.Compose([T.Resize((800, 800)), T.ToTensor()])
        image_tensor = transform(image).unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

        # Tokenizer le prompt
        inputs = self.tokenizer(prompt)

        # Passer l'image et le prompt dans le modèle
        with torch.no_grad():
            outputs = self.model(image_tensor, inputs)

        # Extraire les objets détectés
        detected_objects = outputs['labels']  # Vous pouvez ajuster cette ligne en fonction de la sortie du modèle
        return detected_objects

    def extract_classes_from_prompt(self, prompt, image_path):
        detected_objects = self.detect_objects(image_path, prompt)
        return detected_objects

    def extract_classes_from_prompts(self, prompts_dict):
        detected_classes = {}
        for image_path, prompts in prompts_dict.items():
            detected_objects = []
            for prompt in prompts:
                detected_objects.extend(self.extract_classes_from_prompt(prompt, image_path))
            detected_classes[image_path] = detected_objects
        return detected_classes
