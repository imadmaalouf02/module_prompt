import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Assurez-vous de télécharger les ressources nécessaires
import nltk
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')


nltk.download('punkt')
nltk.download('stopwords')

class TextExtractor:
    def __init__(self):
        # Charger les mots vides en français (ou une autre langue si nécessaire)
        self.stop_words = set(stopwords.words('english'))  # Changez 'french' par 'english' pour l'anglais
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
