from typing import Union, List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from bigram_model import BigramModel
import spacy
import numpy as np

# Initialize FastAPI app
app = FastAPI(
    title="Bigram Text Generation and Word Embeddings API",
    description="API for bigram-based text generation and spaCy word embeddings",
    version="1.0.0"
)

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully")
except IOError:
    print("❌ spaCy model not found. Will be downloaded during Docker build.")
    nlp = None

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "This is another example sentence about natural language processing.",
    "We are generating text based on bigram probabilities and word embeddings.",
    "Bigram models are simple but effective for text generation tasks.",
    "Natural language processing combines linguistics and machine learning.",
    "Machine learning models can generate coherent and meaningful text.",
    "The quick brown fox jumps over the lazy dog in the forest.",
    "Python is a great programming language for data science and NLP.",
    "Word embeddings capture semantic relationships between words.",
    "Deep learning has revolutionized natural language understanding."
]

# Initialize and train the bigram model
print("🔄 Training bigram model...")
bigram_model = BigramModel()
bigram_model.train(corpus)
print("✅ Bigram model training complete")


# Pydantic models for API requests/responses
class TextGenerationRequest(BaseModel):
    start_word: str
    length: int = 10


class WordEmbeddingRequest(BaseModel):
    word: str
    include_similarity: bool = False
    top_similar: int = 5


class EmbeddingResponse(BaseModel):
    word: str
    embedding: List[float]
    dimensions: int
    has_vector: bool
    similarity_words: Optional[List[Dict[str, float]]] = None


class SimilarityRequest(BaseModel):
    word1: str
    word2: str


class SimilarityResponse(BaseModel):
    word1: str
    word2: str
    similarity: float


class WordAnalysisRequest(BaseModel):
    text: str
    include_embeddings: bool = False


# Embedding utilities
class SpacyEmbeddings:
    """spaCy-based word embeddings utility class"""
    
    def __init__(self, nlp_model):
        self.nlp = nlp_model
        if self.nlp is None:
            raise ValueError("spaCy model not loaded")
    
    def get_word_embedding(self, word: str) -> Dict[str, Any]:
        """Get embedding for a single word"""
        doc = self.nlp(word.lower())
        token = doc[0] if len(doc) > 0 else None
        
        if token and token.has_vector:
            return {
                "word": word,
                "embedding": token.vector.tolist(),
                "dimensions": len(token.vector),
                "has_vector": True
            }
        else:
            return {
                "word": word,
                "embedding": [],
                "dimensions": 0,
                "has_vector": False
            }
    
    def get_similarity(self, word1: str, word2: str) -> float:
        """Calculate similarity between two words"""
        doc1 = self.nlp(word1.lower())
        doc2 = self.nlp(word2.lower())
        
        if len(doc1) > 0 and len(doc2) > 0:
            token1, token2 = doc1[0], doc2[0]
            if token1.has_vector and token2.has_vector:
                return float(token1.similarity(token2))
        
        return 0.0
    
    def find_similar_words(self, word: str, vocabulary: List[str], top_k: int = 5) -> List[Dict[str, float]]:
        """Find most similar words from vocabulary"""
        target_doc = self.nlp(word.lower())
        if len(target_doc) == 0 or not target_doc[0].has_vector:
            return []
        
        target_token = target_doc[0]
        similarities = []
        
        for vocab_word in vocabulary:
            if vocab_word.lower() != word.lower():
                doc = self.nlp(vocab_word.lower())
                if len(doc) > 0 and doc[0].has_vector:
                    similarity = float(target_token.similarity(doc[0]))
                    similarities.append({"word": vocab_word, "similarity": similarity})
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:top_k]
    
    def analyze_text(self, text: str, include_embeddings: bool = False) -> Dict[str, Any]:
        """Analyze text and extract word information"""
        doc = self.nlp(text)
        
        results = {
            "text": text,
            "tokens": [],
            "entities": []
        }
        
        for token in doc:
            token_info = {
                "text": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "tag": token.tag_,
                "is_alpha": token.is_alpha,
                "is_stop": token.is_stop,
                "has_vector": token.has_vector
            }
            
            if include_embeddings and token.has_vector:
                token_info["embedding"] = token.vector.tolist()
                token_info["vector_norm"] = float(token.vector_norm)
            
            results["tokens"].append(token_info)
        
        # Named entities
        for ent in doc.ents:
            results["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_)
            })
        
        return results


# Initialize spaCy embeddings
if nlp:
    embeddings = SpacyEmbeddings(nlp)
else:
    embeddings = None


# API Endpoints
@app.get("/")
def read_root():
    spacy_status = "✅ Available" if embeddings else "❌ Not available"
    return {
        "message": "Bigram Text Generation and spaCy Word Embeddings API",
        "spacy_status": spacy_status,
        "endpoints": {
            "generate": "POST /generate - Generate text using bigram model",
            "embedding": "POST /embedding - Get word embedding using spaCy",
            "similarity": "POST /similarity - Calculate similarity between two words",
            "similar_words": "GET /similar/{word} - Find similar words",
            "analyze": "POST /analyze - Analyze text with spaCy",
            "vocabulary": "GET /vocabulary - Get bigram model vocabulary",
            "health": "GET /health - Health check"
        }
    }


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    """Generate text using the bigram model"""
    try:
        generated_text = bigram_model.generate_text(
            request.start_word,
            request.length
        )
        return {
            "generated_text": generated_text,
            "start_word": request.start_word,
            "length": request.length,
            "method": "bigram_model"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Text generation failed: {str(e)}")


@app.post("/embedding", response_model=EmbeddingResponse)
def get_word_embedding(request: WordEmbeddingRequest):
    """Get spaCy word embedding for a word"""
    if not embeddings:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        # Get basic embedding
        embedding_data = embeddings.get_word_embedding(request.word)
        
        # Get similar words if requested
        similar_words = None
        if request.include_similarity and embedding_data["has_vector"]:
            vocab_list = list(bigram_model.vocabulary)
            similar_words = embeddings.find_similar_words(
                request.word, 
                vocab_list, 
                request.top_similar
            )
        
        return EmbeddingResponse(
            word=embedding_data["word"],
            embedding=embedding_data["embedding"],
            dimensions=embedding_data["dimensions"],
            has_vector=embedding_data["has_vector"],
            similarity_words=similar_words
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction failed: {str(e)}")


@app.get("/embedding/{word}")
def get_word_embedding_simple(word: str):
    """Get spaCy word embedding (simple GET endpoint)"""
    if not embeddings:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        return embeddings.get_word_embedding(word)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity", response_model=SimilarityResponse)
def calculate_similarity(request: SimilarityRequest):
    """Calculate semantic similarity between two words"""
    if not embeddings:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        similarity = embeddings.get_similarity(request.word1, request.word2)
        return SimilarityResponse(
            word1=request.word1,
            word2=request.word2,
            similarity=similarity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar/{word}")
def get_similar_words(word: str, top_k: int = 5):
    """Find words similar to the input word from the vocabulary"""
    if not embeddings:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        vocab_list = list(bigram_model.vocabulary)
        similar_words = embeddings.find_similar_words(word, vocab_list, top_k)
        
        return {
            "word": word,
            "similar_words": similar_words,
            "vocabulary_size": len(vocab_list),
            "found_similar": len(similar_words)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze_text(request: WordAnalysisRequest):
    """Analyze text using spaCy NLP pipeline"""
    if not embeddings:
        raise HTTPException(status_code=503, detail="spaCy model not available")
    
    try:
        analysis = embeddings.analyze_text(request.text, request.include_embeddings)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/vocabulary")
def get_vocabulary():
    """Get bigram model vocabulary"""
    return {
        "vocabulary": sorted(list(bigram_model.vocabulary)),
        "size": len(bigram_model.vocabulary),
        "source": "bigram_model"
    }


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "bigram_model": "loaded",
        "spacy_model": "loaded" if embeddings else "not_loaded",
        "vocabulary_size": len(bigram_model.vocabulary) if bigram_model else 0
    }