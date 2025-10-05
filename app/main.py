from typing import Union, List
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import spacy
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torchvision import transforms
from PIL import Image
import io
import os

# Import your existing bigram model
from bigram_model import BigramModel

# Import the CNN model from Part 1
from cnn_model import CNN

app = FastAPI()

# ==================== EXISTING SPACY AND BIGRAM MODEL CODE ====================
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

# ==================== NEW CNN MODEL CODE ====================
# Load CNN model for image classification
print("🔄 Loading CNN model for CIFAR10 classification...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cnn_model = CNN().to(device)
    
    # Get the directory where main.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'cnn_cifar10.pth')
    
    print(f"Looking for model at: {model_path}")
    print(f"File exists: {os.path.exists(model_path)}")
    
    cnn_model.load_state_dict(torch.load(model_path, map_location=device))
    cnn_model.eval()
    print(f"✅ CNN model loaded successfully on {device}")
    cnn_model_available = True
except FileNotFoundError:
    print("❌ CNN model file 'cnn_cifar10.pth' not found")
    cnn_model_available = False
except Exception as e:
    print(f"❌ Failed to load CNN model: {e}")
    cnn_model_available = False

# CIFAR10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Transform for CNN inference
cnn_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==================== REQUEST/RESPONSE MODELS ====================
# Existing models
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

# ==================== API ENDPOINTS ====================
@app.get("/")
def read_root():
    spacy_status = "Available" if nlp else "Not available"
    cnn_status = "Available" if cnn_model_available else "Not available"
    
    return {
        "message": "Text Generation, Word Embeddings, and Image Classification API",
        "spacy_status": spacy_status,
        "cnn_model_status": cnn_status,
        "corpus_source": "The Count of Monte Cristo by Alexandre Dumas (Project Gutenberg)",
        "endpoints": {
            "generate": "POST /generate - Generate text using bigram model trained on The Count of Monte Cristo",
            "word_embedding": "POST /word-embedding - Get word embedding vector for a single word",
            "word_similarity": "POST /word-similarity - Calculate semantic similarity between two words", 
            "sentence_embedding": "POST /sentence-embedding - Get sentence embedding vector (averaged word embeddings)",
            "sentence_similarity": "POST /sentence-similarity - Calculate semantic similarity between two sentences",
            "classify_image": "POST /classify-image - Classify an image using CNN trained on CIFAR10",
            "cnn_model_info": "GET /cnn-model-info - Get information about the CNN model"
        }
    }

# ==================== EXISTING BIGRAM AND EMBEDDING ENDPOINTS ====================
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

# ==================== NEW CNN IMAGE CLASSIFICATION ENDPOINTS ====================
@app.post("/classify-image")
async def classify_image(file: UploadFile = File(...)):
    """
    Classify an image using CNN model trained on CIFAR10.
    
    This endpoint uses a Convolutional Neural Network trained on the CIFAR10 dataset
    to classify images into one of 10 categories: airplane, automobile, bird, cat, 
    deer, dog, frog, horse, ship, or truck.
    
    Parameters:
    - file: Image file to classify (JPG, PNG, etc.)
    
    Returns:
    - predicted_class: The predicted class name
    - confidence: Confidence score for the prediction
    - class_id: Numeric ID of the predicted class
    - top_3_predictions: Top 3 most likely classes with their confidence scores
    """
    if not cnn_model_available:
        raise HTTPException(
            status_code=503, 
            detail="CNN model not available. Please ensure 'cnn_cifar10.pth' exists."
        )
    
    try:
        # Read image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Transform image
        image_tensor = cnn_transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = cnn_model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Get top 3 predictions
            top3_prob, top3_indices = torch.topk(probabilities, 3)
            top3_predictions = [
                {
                    'class': CIFAR10_CLASSES[idx.item()],
                    'confidence': float(prob.item())
                }
                for prob, idx in zip(top3_prob[0], top3_indices[0])
            ]
        
        return {
            "predicted_class": CIFAR10_CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "class_id": int(predicted.item()),
            "top_3_predictions": top3_predictions,
            "model": "CNN (Part 1 Architecture)",
            "dataset": "CIFAR10"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image classification failed: {str(e)}")

@app.get("/cnn-model-info")
def get_cnn_model_info():
    """Get information about the CNN model"""
    if not cnn_model_available:
        raise HTTPException(
            status_code=503, 
            detail="CNN model not available"
        )
    
    try:
        total_params = sum(p.numel() for p in cnn_model.parameters())
        trainable_params = sum(p.numel() for p in cnn_model.parameters() if p.requires_grad)
        
        return {
            "model_name": "CNN for CIFAR10 Classification",
            "architecture": "Part 1 CNN Architecture",
            "input_size": "64x64x3 (RGB images)",
            "num_classes": 10,
            "classes": CIFAR10_CLASSES,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(device),
            "model_file": "cnn_cifar10.pth",
            "architecture_details": {
                "conv1": "16 filters, 3x3 kernel, stride 1, padding 1",
                "conv2": "32 filters, 3x3 kernel, stride 1, padding 1",
                "pooling": "MaxPool2D with 2x2 kernel, stride 2",
                "fc1": "100 units",
                "fc2": "10 units (output)"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))