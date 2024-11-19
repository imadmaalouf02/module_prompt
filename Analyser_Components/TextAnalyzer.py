class TextAnalyzer:
    def __init__(self, text, keywords):
        self.text = text.lower()  # Convertir le texte en minuscules
        self.keywords = [keyword.lower() for keyword in keywords]  # Convertir les mots-clés en minuscules
        self.negation_words = ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor', 
                               "don't", "doesn't", "can't", "won't", "isn't", "aren't", "wasn't", "weren't", "without"]
        self.negation_words = [word.lower() for word in self.negation_words]  # Convertir les mots de négation en minuscules
    
    def analyze(self):
        results = {}
        words = self.text.split()
        last_index = -1  # Pour suivre la position du dernier mot-clé analysé
        
        for keyword in self.keywords:
            if keyword in words:
                index = words.index(keyword, last_index + 1)  # Commencer la recherche après le dernier mot-clé
                negation_found = False
                
                # Vérifier les mots de négation avant le mot-clé
                for i in range(last_index + 1, index):
                    if words[i] in self.negation_words:
                        negation_found = True
                        break
                
                results[keyword] = 'negation' if negation_found else 'no comment'
                last_index = index  # Mettre à jour l'index du dernier mot-clé analysé
            else:
                results[keyword] = 'not found'
        
        return results