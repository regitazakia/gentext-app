from collections import defaultdict, Counter
import numpy as np
import random
import re


class BigramModel:
    """
    A bigram language model for text generation.
    
    This model analyzes text to compute bigram probabilities and can generate
    new text based on these probabilities.
    """
    
    def __init__(self, corpus=None, frequency_threshold=5):
        """
        Initialize the bigram model.
        
        Args:
            corpus (list): List of text strings to train on
            frequency_threshold (int): Minimum frequency for word inclusion
        """
        self.frequency_threshold = frequency_threshold
        self.vocab = []
        self.bigram_probs = defaultdict(dict)
        self.unigram_counts = Counter()
        
        if corpus:
            self.train(corpus)
    
    def simple_tokenizer(self, text, frequency_threshold=None):
        """
        Simple tokenizer that splits text into words.
        
        Args:
            text (str): Input text to tokenize
            frequency_threshold (int): Minimum frequency for word inclusion
            
        Returns:
            list: List of tokens
        """
        if frequency_threshold is None:
            frequency_threshold = self.frequency_threshold
            
        # Convert to lowercase and extract words using regex
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        if not frequency_threshold:
            return tokens
        
        # Count word frequencies
        word_counts = Counter(tokens)
        
        # Filter tokens based on frequency threshold
        filtered_tokens = [
            token for token in tokens 
            if word_counts[token] >= frequency_threshold
        ]
        
        return filtered_tokens
    
    def analyze_bigrams(self, text, frequency_threshold=None):
        """
        Analyze text to compute bigram probabilities.
        
        Args:
            text (str): Input text to analyze
            frequency_threshold (int): Minimum frequency for word inclusion
            
        Returns:
            tuple: (vocabulary, bigram_probabilities)
        """
        words = self.simple_tokenizer(text, frequency_threshold)
        bigrams = list(zip(words[:-1], words[1:]))  # Create bigrams
        
        # Count bigram and unigram frequencies
        bigram_counts = Counter(bigrams)
        unigram_counts = Counter(words)
        
        # Compute bigram probabilities
        bigram_probs = defaultdict(dict)
        for (word1, word2), count in bigram_counts.items():
            bigram_probs[word1][word2] = count / unigram_counts[word1]
        
        return list(unigram_counts.keys()), bigram_probs
    
    def train(self, corpus):
        """
        Train the bigram model on a corpus of texts.
        
        Args:
            corpus (list): List of text strings to train on
        """
        # Combine all texts in the corpus
        combined_text = " ".join(corpus)
        
        # Analyze bigrams from the combined text
        self.vocab, self.bigram_probs = self.analyze_bigrams(combined_text)
        
        # Store unigram counts for reference
        words = self.simple_tokenizer(combined_text)
        self.unigram_counts = Counter(words)
    
    def generate_text(self, start_word, num_words=20):
        """
        Generate text based on bigram probabilities.
        
        Args:
            start_word (str): Starting word for text generation
            num_words (int): Number of words to generate
            
        Returns:
            str: Generated text
        """
        current_word = start_word.lower()
        generated_words = [current_word]
        
        for _ in range(num_words - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:  # If no bigrams for the current word, stop generating
                break
            
            # Choose the next word based on probabilities
            next_word = random.choices(
                list(next_words.keys()),
                weights=next_words.values()
            )[0]
            
            generated_words.append(next_word)
            current_word = next_word  # Move to the next word
        
        return " ".join(generated_words)
    
    def print_bigram_probs_matrix(self):
        """
        Print bigram probabilities in a matrix format.
        """
        print(f"{'':15}", end="")
        for word in self.vocab:
            print(f"{word:15}", end="")
        print("\n" + "-" * (15 * (len(self.vocab) + 1)))
        
        # Print each row with probabilities
        for word1 in self.vocab:
            print(f"{word1:15}", end="")
            for word2 in self.vocab:
                prob = self.bigram_probs.get(word1, {}).get(word2, 0)
                print(f"{prob:15.2f}", end="")
            print()
    
    def get_vocabulary(self):
        """
        Get the vocabulary of the model.
        
        Returns:
            list: List of words in vocabulary
        """
        return self.vocab
    
    def get_bigram_probability(self, word1, word2):
        """
        Get the bigram probability P(word2|word1).
        
        Args:
            word1 (str): First word
            word2 (str): Second word
            
        Returns:
            float: Bigram probability
        """
        return self.bigram_probs.get(word1.lower(), {}).get(word2.lower(), 0.0)
    
    def get_next_word_probabilities(self, word):
        """
        Get all possible next words and their probabilities for a given word.
        
        Args:
            word (str): Current word
            
        Returns:
            dict: Dictionary of next words and their probabilities
        """
        return self.bigram_probs.get(word.lower(), {})


# Example usage and testing
if __name__ == "__main__":
    # Example corpus
    sample_corpus = [
        "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
        "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
        "this is another example sentence",
        "we are generating text based on bigram probabilities",
        "bigram models are simple but effective"
    ]
    
    # Create and train the model
    model = BigramModel(sample_corpus)
    
    # Generate some text
    print("Generated text:", model.generate_text("the", 10))
    print("Vocabulary size:", len(model.get_vocabulary()))
    print("Bigram probability P(count|the):", model.get_bigram_probability("the", "count"))