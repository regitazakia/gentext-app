"""
Bigram Model Implementation
This module contains the logic for processing bigrams and can be used with an API.
"""

import re
import json
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional, Any
import pickle
import os


class BigramModel:
    """
    A class for building and using bigram language models.
    """
    
    def __init__(self):
        """Initialize the bigram model."""
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.vocabulary = set()
        self.vocab_size = 0
        self.total_bigrams = 0
        self.smoothing_alpha = 1.0  # Laplace smoothing parameter
        
    def preprocess_text(self, text: str, lowercase: bool = True) -> List[str]:
        """
        Preprocess text by tokenizing and optionally lowercasing.
        
        Args:
            text (str): Input text to preprocess
            lowercase (bool): Whether to convert to lowercase
            
        Returns:
            List[str]: List of tokens
        """
        if lowercase:
            text = text.lower()
        
        # Basic tokenization - split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text)
        
        # Add sentence boundary markers
        tokens = ['<START>'] + tokens + ['<END>']
        
        return tokens
    
    def train(self, texts: List[str], lowercase: bool = True) -> None:
        """
        Train the bigram model on a collection of texts.
        
        Args:
            texts (List[str]): List of training texts
            lowercase (bool): Whether to convert texts to lowercase
        """
        print(f"Training bigram model on {len(texts)} texts...")
        
        # Reset counters
        self.bigrams = defaultdict(Counter)
        self.unigrams = Counter()
        self.vocabulary = set()
        
        # Process each text
        for i, text in enumerate(texts):
            if i % 1000 == 0:
                print(f"Processing text {i+1}/{len(texts)}")
                
            tokens = self.preprocess_text(text, lowercase)
            
            # Count unigrams
            for token in tokens:
                self.unigrams[token] += 1
                self.vocabulary.add(token)
            
            # Count bigrams
            for j in range(len(tokens) - 1):
                w1, w2 = tokens[j], tokens[j + 1]
                self.bigrams[w1][w2] += 1
                self.total_bigrams += 1
        
        self.vocab_size = len(self.vocabulary)
        print(f"Training complete. Vocabulary size: {self.vocab_size}")
        print(f"Total bigrams: {self.total_bigrams}")
    
    def get_bigram_probability(self, w1: str, w2: str, smoothed: bool = True) -> float:
        """
        Calculate the probability of a bigram P(w2|w1).
        
        Args:
            w1 (str): First word
            w2 (str): Second word
            smoothed (bool): Whether to apply Laplace smoothing
            
        Returns:
            float: Bigram probability
        """
        if smoothed:
            # Laplace smoothing: P(w2|w1) = (count(w1, w2) + α) / (count(w1) + α * V)
            numerator = self.bigrams[w1][w2] + self.smoothing_alpha
            denominator = self.unigrams[w1] + self.smoothing_alpha * self.vocab_size
        else:
            # No smoothing
            if self.unigrams[w1] == 0:
                return 0.0
            numerator = self.bigrams[w1][w2]
            denominator = self.unigrams[w1]
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def calculate_perplexity(self, test_texts: List[str]) -> float:
        """
        Calculate perplexity on test texts.
        
        Args:
            test_texts (List[str]): List of test texts
            
        Returns:
            float: Perplexity score
        """
        total_log_prob = 0.0
        total_words = 0
        
        for text in test_texts:
            tokens = self.preprocess_text(text)
            
            for i in range(len(tokens) - 1):
                w1, w2 = tokens[i], tokens[i + 1]
                prob = self.get_bigram_probability(w1, w2)
                
                if prob > 0:
                    total_log_prob += -1 * (prob ** 0.5)  # Negative log probability
                    total_words += 1
        
        if total_words == 0:
            return float('inf')
        
        avg_log_prob = total_log_prob / total_words
        perplexity = 2 ** avg_log_prob
        
        return perplexity
    
    def generate_text(self, start_word: str = '<START>', max_length: int = 50, 
                     temperature: float = 1.0) -> str:
        """
        Generate text using the bigram model.
        
        Args:
            start_word (str): Starting word for generation
            max_length (int): Maximum number of words to generate
            temperature (float): Temperature for sampling (higher = more random)
            
        Returns:
            str: Generated text
        """
        import random
        import math
        
        words = [start_word]
        current_word = start_word
        
        for _ in range(max_length):
            if current_word == '<END>':
                break
                
            # Get possible next words and their probabilities
            next_words = list(self.bigrams[current_word].keys())
            if not next_words:
                break
            
            # Calculate probabilities with temperature
            probs = []
            for next_word in next_words:
                prob = self.get_bigram_probability(current_word, next_word)
                # Apply temperature
                prob = prob ** (1.0 / temperature) if temperature > 0 else prob
                probs.append(prob)
            
            # Normalize probabilities
            total_prob = sum(probs)
            if total_prob == 0:
                break
            probs = [p / total_prob for p in probs]
            
            # Sample next word
            current_word = random.choices(next_words, weights=probs)[0]
            words.append(current_word)
        
        # Remove boundary markers and join
        generated_words = [w for w in words if w not in ['<START>', '<END>']]
        return ' '.join(generated_words)
    
    def get_most_likely_next_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get the most likely next words for a given word.
        
        Args:
            word (str): Input word
            top_k (int): Number of top words to return
            
        Returns:
            List[Tuple[str, float]]: List of (word, probability) tuples
        """
        if word not in self.bigrams:
            return []
        
        next_words = []
        for next_word, count in self.bigrams[word].items():
            prob = self.get_bigram_probability(word, next_word)
            next_words.append((next_word, prob))
        
        # Sort by probability and return top_k
        next_words.sort(key=lambda x: x[1], reverse=True)
        return next_words[:top_k]
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        model_data = {
            'bigrams': dict(self.bigrams),
            'unigrams': dict(self.unigrams),
            'vocabulary': list(self.vocabulary),
            'vocab_size': self.vocab_size,
            'total_bigrams': self.total_bigrams,
            'smoothing_alpha': self.smoothing_alpha
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Convert back to defaultdict and Counter
        self.bigrams = defaultdict(Counter)
        for w1, next_words in model_data['bigrams'].items():
            self.bigrams[w1] = Counter(next_words)
        
        self.unigrams = Counter(model_data['unigrams'])
        self.vocabulary = set(model_data['vocabulary'])
        self.vocab_size = model_data['vocab_size']
        self.total_bigrams = model_data['total_bigrams']
        self.smoothing_alpha = model_data['smoothing_alpha']
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.
        
        Returns:
            Dict[str, Any]: Model information
        """
        return {
            'vocab_size': self.vocab_size,
            'total_bigrams': self.total_bigrams,
            'num_unique_bigrams': sum(len(next_words) for next_words in self.bigrams.values()),
            'smoothing_alpha': self.smoothing_alpha,
            'top_unigrams': self.unigrams.most_common(10)
        }


# API-friendly functions
def create_and_train_model(texts: List[str], lowercase: bool = True) -> BigramModel:
    """
    Create and train a new bigram model.
    
    Args:
        texts (List[str]): Training texts
        lowercase (bool): Whether to lowercase the text
        
    Returns:
        BigramModel: Trained bigram model
    """
    model = BigramModel()
    model.train(texts, lowercase)
    return model


def load_trained_model(filepath: str) -> BigramModel:
    """
    Load a pre-trained bigram model.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        BigramModel: Loaded bigram model
    """
    model = BigramModel()
    model.load_model(filepath)
    return model


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "A quick brown dog jumps over the fox.",
        "The lazy dog sleeps under the tree.",
        "Brown foxes are quick and smart.",
        "Dogs and foxes are animals."
    ]
    
    # Create and train model
    print("Creating and training bigram model...")
    model = create_and_train_model(sample_texts)
    
    # Print model info
    print("\nModel Information:")
    info = model.get_model_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Test bigram probability
    print(f"\nP(brown|quick) = {model.get_bigram_probability('quick', 'brown'):.4f}")
    print(f"P(fox|brown) = {model.get_bigram_probability('brown', 'fox'):.4f}")
    
    # Get most likely next words
    print(f"\nMost likely words after 'the':")
    for word, prob in model.get_most_likely_next_words('the', top_k=5):
        print(f"  {word}: {prob:.4f}")
    
    # Generate some text
    print(f"\nGenerated text: {model.generate_text('the', max_length=10)}")
    
    # Save model
    model.save_model('bigram_model.pkl')
    
    # Test loading
    print("\nTesting model loading...")
    loaded_model = load_trained_model('bigram_model.pkl')
    print(f"Loaded model vocabulary size: {loaded_model.vocab_size}")