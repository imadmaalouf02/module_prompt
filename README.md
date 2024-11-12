<img src="./Assets/promptAnalyzer.jpg" width="200px"></img>

## Table of Contents

> [Overview](#overview)

> [Prompt Gemini Generator](#features)

> [Features](#features)

- [Prompt pre-processing](#prompt-processing)
- [Similarity Reduction](#Similarity-Reduction)
- [Complexity Analysis](#Complexity-Analysis)
- [Readability Analysis](#Readability-Analysis)
> [Prompts processing](#Prompts-processing)

> [Prompt Generator Exemple](#Prompt-Generator-Exemple)

> [Prompt Analyzer Exemple](#Prompt-Analyzer-Exemple)


## Overview

The prompt_analyzer is designed to analyze sets of prompts associated with images and generated using gemini pro vision model. After properly processing the prompts, removing similarities based on user set threshold, the prompt_analyzer evaluates them based on complexity and readability to identify the most effective prompts. It leverages various Python libraries including NLTK for natural language processing, scikit-learn for feature extraction and cosine similarity, and others for specific linguistic tasks.

 

## Prompt Gemini Generator
The prompt_generator class is designed to automate the creation of textual prompts for images using Gemini pro Vision API. 

The prompt_generator class automates the process of generating text from images, offering a bridge between visual content and textual descriptions through advanced machine learning techniques.
> Parameters:
- model: Identifier or configuration for a generative model.
- key: API key for accessing the generative model's service.
- images_dir: Directory path where image files are stored.
- images_extensions: List of image file extensions to consider.
> Operations:
- Configures the generative model with the provided API key (gai.configure).
- Initializes the model instance (gai.GenerativeModel(model)).
- Retrieves and stores paths to images within the specified directory that match the given extensions using sv.list_files_with_extensions.
- Initializes a dictionary (prompts_dict) to store the generated prompts indexed by image name.

> Prompt Generation (generate_prompts):

>> Parameter:
- number_of_prompts: Specifies how many prompts to generate per image.
>> Operations:
- Iterates over each image file retrieved during initialization.
- For each image, it opens the image file and generates the specified number of prompts using the configured model.
- Each prompt's text is added to prompts_dict under the corresponding image name.
>> Output:
- Returns prompts_dict, a dictionary where each key is an image name and the value is a list of generated prompts for that image.
>> Key Functionalities
- Image Handling: Opens image files and prepares them for prompt generation.
- Prompt Generation: Leverages a deep learning model to generate creative or descriptive text based on the image content.
- Data Management: Efficiently manages and catalogs prompts for multiple images, facilitating easy retrieval and usage.



## Features
- Prompt processing : Removes stop words and puntuation to help ensure similarity comparison;
- Similarity Reduction: Removes highly similar prompts to ensure diversity using cosine similarity.
- Complexity Analysis: Evaluates the complexity of prompts based on the length and vocabulary richness.
- Readability Analysis: Computes readability scores using the Flesch Reading Ease formula.


### Prompt processing
```python
def prompt_processing(self)
```
The prompt_processing performs preprocessing on a list of text prompts to prepare them for further analysis.

> Core Functionality:

- Remove Punctuation: Each prompt is stripped of punctuation using a translation table, which simplifies the text and removes unnecessary characters.
- Tokenization: The unpunctuated prompt is then split into individual words (tokens) using NLTK’s word_tokenize.
- Remove Stop Words: Common words (like "and", "the", etc.) that do not add much value in text analysis (known as stop words) are filtered out from the tokens.
- Track Lengths and Unique Words: The method calculates the length of each filtered prompt (number of meaningful words) and identifies the unique words used in each prompt.
> Outputs:
- prompts_unpunctuated: List of prompts with punctuation removed.
prompts_filtered: List of prompts after removing stop words.
- prompts_length: List containing the length of each filtered prompt.
- unique_words_list: List of sets, each containing unique words from each prompt.

### Similarity Reduction

```python
def prompts_similarity(self, remove_similar=False, threshold=0.7):
```
The prompts_similarity method evaluates the similarity between text prompts and optionally removes highly similar ones based on a specified threshold (set by default as 70% similarity, meaning that for 10 prompts with similarity rate higher than 70%, only one will remain).

> Functionality:
- Preprocessing: It first processes the list of prompts to remove punctuation, using the prompt_processing method.
- Vectorization: Converts the cleaned prompts into a TF-IDF matrix, which numerically represents the importance of words within the prompts.
- Similarity Calculation: Computes pairwise cosine similarities between all prompts, resulting in a similarity matrix.
> Parameters:
- remove_similar (boolean): If set to True, the method will remove prompts that are similar above a certain threshold.
- threshold (float): The similarity threshold for determining whether two prompts are considered similar.
> Outputs:
- If remove_similar is False, the method returns the similarity matrix.
- If remove_similar is True, it modifies the list of prompts by removing similar ones: Identifies pairs of prompts that exceed the similarity threshold. Removes prompts to reduce redundancy, keeping one prompt from each similar pair, and returns the pruned list of prompts.
> Use Case:

This method is useful for reducing redundancy in datasets where prompts may be too similar, which can be essential for training models where diversity of input data enhances learning efficacy.

### Complexity Analysis

```python
def prompt_complexity(self):
```

The prompt_complexity method calculates the complexity of text prompts based on their length and vocabulary richness. 

> Functionality:
- Preprocessing: It starts by calling prompt_processing to get a list of prompts that have been filtered of punctuation and stop words.
- Complexity Calculation:
- Prompt Length: Measures the number of words in each filtered prompt.
- Unique Words: Counts the number of unique words in each prompt.
- Vocabulary Richness: Calculates the ratio of unique words to total prompt length, which indicates the diversity of vocabulary used.
- Complexity Score: Multiplies the prompt length by the vocabulary richness to get a score representing the prompt's complexity.
- Compilation of Scores: Stores and then returns a list of these complexity scores for each prompt, sorted from least to most complex.
> Output:

Returns a sorted list of complexity scores, where each score quantifies the lexical richness and length of a prompt, serving as an indicator of its complexity.
> Use Case:

This method is valuable for analyzing and ranking prompts based on their linguistic complexity, which can be important for applications where the level of language complexity is critical, such as educational content creation or text-based AI training scenarios.


### Readability Analysis

```python
def readability(self):
```

The readability method calculates the readability of a text prompt using the Flesch Reading Ease formula, a widely recognized method to evaluate the ease of understanding of a text.

> Functionality:
- Sentence and Word Tokenization: The method first tokenizes the input prompt into sentences and words using NLTK's sent_tokenize and word_tokenize.
- Syllable Counting: Retrieves the CMU Pronouncing Dictionary (cmudict) to count syllables. For each word, it extracts the pronunciation and counts the number of syllable markers (digits in the pronunciation).
- Flesch Score Calculation: Computes the total number of sentences, words, and syllables in the prompt. Applies the Flesch Reading Ease formula: 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words). Rounds the result to two decimal places.

> Output:

Returns the Flesch Reading Ease score for the prompt, where higher scores indicate texts that are easier to read, and lower scores indicate texts that are more difficult.

> Use Case:

This method is particularly useful for ensuring text content is appropriate for the intended audience's reading level, such as in educational materials, marketing content, or publishing, where readability can impact engagement and comprehension.


## Prompt processing
The process_prompts method in performs several operations to analyze and rank text prompts based on either their complexity or readability (user's choice), and then records the top results. Here's a brief overview:

> Functionality:
- Initialization: It initializes an empty list to store results.
- Loop Through Prompts: Iterates through each set of prompts associated with an image:
- Similarity Removal: First removes similar prompts to ensure diversity using the prompts_similarity method.
Scoring and Sorting:
- If complexity is True, it calculates complexity scores using prompt_complexity and sorts prompts from most to least complex.
- If readability is True, it calculates readability scores using prompt_readability and sorts prompts from easiest to hardest to read.
- Selection of Top Prompts: Extracts the top three prompts. If there are fewer than three, fills the remaining slots with "N/A".
- Recording Results: Compiles results into a dictionary for each image, including the image name and the top three prompts.
- Output to CSV: Calls write_to_csv to write the results to a CSV file.
> Parameters:
- readability (bool): Determines if prompts should be analyzed and sorted by readability.
- complexity (bool): Determines if prompts should be analyzed and sorted by complexity.
> Output:
- Writes a CSV file named prompt_results.csv with headers image_name, best_prompt1, best_prompt2, and best_prompt3, documenting the top three prompts for each image.
> Use Case:
- This method is useful for preparing prompt datasets where the best prompts need to be identified and cataloged based on specific criteria like readability or complexity. It's particularly valuable in scenarios where prompt quality impacts user engagement or educational outcomes, ensuring that the most suitable prompts are used for further applications or studies.



## Prompt Generator Exemple
Here is a simple example to demonstrate how to use the prompt_generator class:
```python
API_Key=input("Enter your API Key")
prompts=prompt_generator('gemini-pro-vision',API_Key)
prompts_dict=prompts.generate_prompts(number_of_prompts=10)
```

## Prompt Analyzer Exemple
Here is a simple example to demonstrate how to use the prompt_analyzer class:
```python
prompts_dict = {
    'image1.jpg': ['An early morning', 'Sunrise at the beach', 'Dawn breaks over the ocean']
}
analyzer = prompt_analyzer(prompts_dict)
analyzer.process_prompts(complexity=True)
```

