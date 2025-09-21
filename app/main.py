from typing import Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bigram_model import BigramModel
import spacy
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load spaCy model for embeddings
try:
    nlp = spacy.load("en_core_web_lg")
    print("✅ spaCy large model loaded successfully")
except IOError:
    print("❌ spaCy large model not found. Trying small model...")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ spaCy small model loaded successfully")
    except IOError:
        print("❌ No spaCy model found")
        nlp = None

# Download corpus from Project Gutenberg
print("📚 Downloading The Count of Monte Cristo from Project Gutenberg...")
try:
    book_url = "https://www.gutenberg.org/cache/epub/1184/pg1184.txt"
    response = requests.get(book_url)
    book_text = response.text
    
    # Remove Gutenberg header and footer
    start_marker = "*** START OF THIS PROJECT GUTENBERG EBOOK"
    end_marker = "*** END OF THIS PROJECT GUTENBERG EBOOK"
    
    start_idx = book_text.find(start_marker)
    end_idx = book_text.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        book_text = book_text[start_idx + len(start_marker) : end_idx]
        corpus = [book_text]  # Use the full book as corpus
        print(f"✅ Successfully downloaded book ({len(book_text)} characters)")
    else:
        raise Exception("Could not find book markers")
        
except Exception as e:
    print(f"❌ Failed to download book: {e}")
    print("📝 Using fallback small corpus...")
    # Fallback to small corpus if download fails
    corpus = [
        "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
        "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
        "this is another example sentence",
        "we are generating text based on bigram probabilities",
        "bigram models are simple but effective"
    ]

# Initialize bigram model
print("🔄 Training bigram model...")
bigram_model = BigramModel(corpus, frequency_threshold=1)  # Higher threshold for large corpus
print("✅ Bigram model training complete")

# Request/Response models
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class WordEmbeddingRequest(BaseModel):
    word: str

class WordSimilarityRequest(BaseModel):
    word1: str
    word2: str

class SentenceEmbeddingRequest(BaseModel):
    sentence: str

class SentenceSimilarityRequest(BaseModel):
    sentence1: str
    sentence2: str

# API Endpoints
@app.get("/")
def read_root():
    spacy_status = "Available" if nlp else "Not available"
    return {
        "message": "Text Generation and Word Embeddings API",
        "spacy_status": spacy_status,
        "corpus_source": "The Count of Monte Cristo by Alexandre Dumas (Project Gutenberg)",
        "endpoints": {
            "generate": "POST /generate - Generate text using bigram model trained on The Count of Monte Cristo",
            "word_embedding": "POST /word-embedding - Get word embedding vector for a single word",
            "word_similarity": "POST /word-similarity - Calculate semantic similarity between two words", 
            "sentence_embedding": "POST /sentence-embedding - Get sentence embedding vector (averaged word embeddings)",
            "sentence_similarity": "POST /sentence-similarity - Calculate semantic similarity between two sentences"
        }
    }

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """
    Generate text using bigram language model.
    
    This endpoint uses a bigram model trained on "The Count of Monte Cristo" 
    by Alexandre Dumas to generate text starting from a given word.
    The model predicts the next word based on bigram probabilities learned 
    from the training corpus.
    
    Parameters:
    - start_word: The word to start text generation from
    - length: Number of words to generate
    
    Returns:
    - generated_text: The generated text sequence
    """
    try:
        generated_text = bigram_model.generate_text(request.start_word, request.length)
        return {
            "generated_text": generated_text,
            "start_word": request.start_word,
            "length": request.length,
            "model": "Bigram Language Model",
            "training_corpus": "The Count of Monte Cristo by Alexandre Dumas"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")

@app.post("/word-embedding")
def get_word_embedding(request: WordEmbeddingRequest):
    """Get word embedding for a single word"""
    if not nlp:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        word_doc = nlp(request.word)
        embedding = word_doc.vector.tolist()
        
        return {
            "word": request.word,
            "embedding": embedding,
            "embedding_size": len(embedding),
            "first_10_values": embedding[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/word-similarity")
def get_word_similarity(request: WordSimilarityRequest):
    """Calculate similarity between two words"""
    if not nlp:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        word1_doc = nlp(request.word1)
        word2_doc = nlp(request.word2)
        similarity = word1_doc.similarity(word2_doc)
        
        return {
            "word1": request.word1,
            "word2": request.word2,
            "similarity": float(similarity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentence-embedding")
def get_sentence_embedding(request: SentenceEmbeddingRequest):
    """Get sentence embedding (averaged word embeddings)"""
    if not nlp:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        sentence_doc = nlp(request.sentence)
        embedding = sentence_doc.vector.tolist()
        
        return {
            "sentence": request.sentence,
            "embedding": embedding,
            "embedding_size": len(embedding),
            "first_10_values": embedding[:10]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentence-similarity")
def get_sentence_similarity(request: SentenceSimilarityRequest):
    """Calculate similarity between two sentences"""
    if not nlp:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        sentence1_doc = nlp(request.sentence1)
        sentence2_doc = nlp(request.sentence2)
        similarity = sentence1_doc.similarity(sentence2_doc)
        
        return {
            "sentence1": request.sentence1,
            "sentence2": request.sentence2,
            "similarity": float(similarity)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))