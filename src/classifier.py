import json
import ast
import pymorphy2
import string

import torch

class NewsClassifier: 
    def __init__(self): 
        with open("src/resources/dictionary.json", "r", encoding="utf-8") as f:
            self. dictionary = json.load(f)
        with open("src/resources/markers.txt", "r", encoding="utf-8") as f:
            self.markers = [w.replace("\n", "") for w in f.readlines()[:100]]
        with open(r'src/resources/stopwords_ua_list.txt', "r", encoding="utf-8") as f: 
            self.stopwords = ast.literal_eval(f.read())
        self.morph = pymorphy2.MorphAnalyzer(lang='uk')
        self.embedding_dim = 150
        self.LSTMClassifier = torch.jit.load("src/lstm.pt")

    def classify_news(self, text: str, title: str) -> str: 
        """classifies news"""
        x = (title + " " + text) if title else text
        x = torch.LongTensor(self.tokenize_x([x]))
        label = self.LSTMClassifier(x)
        label = (label >= 0.5).squeeze().long()
        label = "Real" if int(label) == 0 else "Fake"
        return label

    def tokenize_x(self, x: list[str]) -> list[list[float]]: 
        """turns text into numbers"""
        tokenized = []
        for text in x: 
            words = self.preprocess_text(text)
            for i, word in enumerate(words): 
                if word not in self.dictionary or word in self.markers or word in self.stopwords or not self.is_not_punctuation(word): 
                    words[i] = 0
                else:
                    words[i] = self.dictionary[word]
            tokenized.append(words)
        return tokenized
    
    def preprocess_text(self, text: str) -> list[str]: 
        """Removing stopwords + lemmatization"""
        text = str(text)
        words = [self.lemmatize_word(word) for word in text.split() if word.lower() not in self.stopwords]
        padded = words[:self.embedding_dim] + [0] * (self.embedding_dim - len(words))
        return padded
    
    def lemmatize_word(self, word) -> str:
        """Lemmatization"""
        return self.morph.parse(word)[0].normal_form
    
    def is_not_punctuation(self, word) -> bool:
        """Checks if string is punctuation"""
        return all(char not in string.punctuation for char in word)