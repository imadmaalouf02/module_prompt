import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Assurez-vous de télécharger les ressources nécessaires
nltk.download('punkt')
nltk.download('stopwords')

class TextExtractor:
    def __init__(self):
        # Charger les mots vides en anglais (ou une autre langue si nécessaire)
        self.stop_words = set(stopwords.words('english'))  # Changez 'english' par 'french' pour le français
        self.punctuation = set(string.punctuation)

    def extract_keywords(self, text: str):
        """
        Extrait les mots d'un texte en éliminant les mots vides et la ponctuation.
        
        :param text: Le texte à traiter.
        :return: Une liste de mots extraits.
        """
        # Tokenisation du texte
        words = word_tokenize(text)

        # Filtrer les mots vides et la ponctuation
        keywords = [
            word.lower() for word in words 
            if word.lower() not in self.stop_words and word not in self.punctuation
        ]

        return keywords

    def extract_image_and_object_names(self, data):
        """
        Extrait les noms des images et les noms des objets de chaque image.

        :param data: Liste contenant un dictionnaire avec des images et leurs objets détectés.
        :return: Liste de tuples contenant le nom de l'image et les noms des objets.
        """
        result = []
        
        # Parcourir la liste principale
        for item in data:
            for image_name, objects in item.items():
                # Extraire les noms des objets
                object_names = [obj['label'] for obj in objects]
                # Ajouter le nom de l'image et les noms des objets au résultat
                result.append((image_name, object_names))
        
        return result

    def extract_object_names(self, data):
        """
        Extrait uniquement les noms des objets détectés, en ignorant les noms des images et en supprimant les doublons.

        :param data: Liste contenant un dictionnaire avec des images et leurs objets détectés.
        :return: Liste de noms d'objets sans doublons.
        """
        object_names = []
        
        # Parcourir la liste principale
        for item in data:
            # item est un dictionnaire
            for objects in item.values():  # Parcourir les objets de chaque image
                # Extraire les noms des objets et les ajouter à la liste
                object_names.extend([obj['label'] for obj in objects])
        
        # Supprimer les doublons en convertissant la liste en ensemble, puis en convertissant à nouveau en liste
        object_names = list(set(object_names))
        
        return object_names