import language_tool_python
import random
import csv
from PIL import Image
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords, cmudict
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#from object_detector import * 

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')

class prompt_analyzer:
    def __init__(self, prompts_dict):
        self.prompts_dict = prompts_dict  
        #self.object_detector = ObjectDetector()

    def process_prompts(self, readability=False, complexity=True):
        results = []
        for image_name, prompts_list in self.prompts_dict.items():
            self.prompts_list = prompts_list
            self.prompts_similarity(remove_similar=False)  
            
            if complexity:
                scores = self.prompt_complexity()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: -x[1])
            elif readability:
                scores = self.prompt_readability()
                sorted_prompts = sorted(zip(self.prompts_list, scores), key=lambda x: x[1])

            # Handle fewer than three prompts
            top_prompts = [prompt[0] for prompt in sorted_prompts[:3]]  
            while len(top_prompts) < 3:
                top_prompts.append("N/A")  # Fill in missing prompts with "N/A"

            results.append({
                'image_name': image_name,
                'best_prompt1': top_prompts[0],
                'best_prompt2': top_prompts[1],
                'best_prompt3': top_prompts[2]
            })

        self.write_to_csv(results)

    def write_to_csv(self, results):
        with open('prompt_results.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['image_name','best_prompt1', 'best_prompt2', 'best_prompt3'])
            writer.writeheader()
            for result in results:
                writer.writerow(result)

    def prompt_processing(self):
        stop_words = set(stopwords.words('english'))
        punct_table = str.maketrans('', '', string.punctuation)
        prompts_unpunctuated = []
        prompts_filtered = []
        prompts_length = []
        unique_words_list = []

        for prompt in self.prompts_list:
            prompt_unpunctuated = prompt.translate(punct_table)
            words = word_tokenize(prompt_unpunctuated)
            prompt_filtered = [word for word in words if word.lower() not in stop_words]
            prompt_length = len(prompt_filtered)
            unique_words = set(prompt_filtered)

            prompts_unpunctuated.append(prompt_unpunctuated)
            prompts_filtered.append(prompt_filtered)
            prompts_length.append(prompt_length)
            unique_words_list.append(unique_words)

        return prompts_unpunctuated, prompts_filtered, prompts_length, unique_words_list

    def prompts_similarity(self, remove_similar=False, threshold=0.7):
        prompts_unpunctuated, _, _, _ = self.prompt_processing()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(prompts_unpunctuated)
        similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

        if not remove_similar:
            return similarity_matrix
        else:
            similar_prompts = set()
            for i in range(similarity_matrix.shape[0]):
                for j in range(i + 1, similarity_matrix.shape[1]):
                    if similarity_matrix[i][j] > threshold:
                        similar_prompts.update([i, j])

            similar_prompts = list(similar_prompts)
            m = len(similar_prompts)
            
            # Vérification du nombre de prompts similaires
            if m <= 1:
                print("Not enough similar prompts found, returning original prompts.")
                return self.prompts_list  # Retourne les prompts d'origine s'il n'y en a pas assez

            prompts_to_remove = [self.prompts_list[i] for i in random.sample(similar_prompts, m - 1)]
            self.prompts_list = [prompt for prompt in self.prompts_list if prompt not in prompts_to_remove]
            return self.prompts_list



    def prompt_complexity(self):
        complexity_scores = []
        _, prompts_filtered, _, _ = self.prompt_processing()

        for filtered_prompt in prompts_filtered:
            prompt_length = len(filtered_prompt)
            unique_words_number = set(filtered_prompt)
            vocabulary_richness = len(unique_words_number) / prompt_length if prompt_length > 0 else 0
            complexity_score = prompt_length * vocabulary_richness
            complexity_scores.append(complexity_score)

        return sorted(complexity_scores)

    def prompt_readability(self):
        flesch_scores = []

        for prompt in self.prompts_list:
            flesch_score = self.readability(prompt)
            flesch_scores.append(flesch_score)

        return sorted(flesch_scores)

    def readability(self, prompt):
        sentences = sent_tokenize(prompt)
        words = word_tokenize(prompt)
        num_sentences = len(sentences)
        num_words = len(words)
        d = cmudict.dict()

        def count_syllables(word):
            pronunciation_list = d.get(word.lower())
            if not pronunciation_list:
                return 0
            pronunciation = pronunciation_list[0]
            return sum(1 for s in pronunciation if s[-1].isdigit())

        num_syllables = sum(count_syllables(word) for word in words)
        flesch_score = round(206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words), 2)
        return flesch_score


# def filter_prompts_by_objects(self, required_objects):
#     filtered_prompts = {}
    
#     for image_name, prompts_list in self.prompts_dict.items():
#         detected_objects = self.object_detector.detect_objects(prompts_list)

#         # Filtrer les prompts en fonction des objets détectés
#         filtered_prompts_list = []
#         for prompt, objects in detected_objects.items():
#             if any(obj in required_objects for obj in objects["labels"]):
#                 filtered_prompts_list.append(prompt)

#         filtered_prompts[image_name] = filtered_prompts_list

#     return filtered_prompts