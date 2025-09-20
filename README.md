# Bigram Text Generation and Word Embeddings API

A FastAPI application that combines bigram-based text generation with spaCy word embeddings for natural language processing tasks.

## Features

- **Bigram Text Generation**: Generate text using trained bigram language models
- **spaCy Word Embeddings**: Extract semantic word vectors using pre-trained spaCy models
- **Word Similarity**: Calculate semantic similarity between words
- **Text Analysis**: Comprehensive NLP analysis including POS tagging and named entity recognition
- **Docker Support**: Fully containerized application
- **Interactive API Documentation**: Swagger UI available at `/docs`

## Quick Start

### Using Docker (Recommended)

```bash
# Build the Docker image
docker build -t gentext-app .

# Run the container
docker run -p 8000:8000 gentext-app
```

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the application
uvicorn main:app --reload
```

## API Endpoints

### Text Generation
- `POST /generate` - Generate text using bigram model

### Word Embeddings
- `POST /embedding` - Get word embedding using spaCy
- `GET /embedding/{word}` - Simple word embedding lookup

### Similarity Analysis
- `POST /similarity` - Calculate similarity between two words
- `GET /similar/{word}` - Find similar words from vocabulary

### Text Analysis
- `POST /analyze` - Analyze text with spaCy

### Utility Endpoints
- `GET /` - API information
- `GET /vocabulary` - Get bigram model vocabulary
- `GET /health` - Health check

## Access the API

- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## Example Usage

### Generate Text
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"start_word": "The", "length": 10}'
```

### Get Word Embedding
```bash
curl -X POST "http://localhost:8000/embedding" \
     -H "Content-Type: application/json" \
     -d '{"word": "language", "include_similarity": true}'
```

### Calculate Word Similarity
```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{"word1": "cat", "word2": "dog"}'
```

### Find Similar Words
```bash
curl "http://localhost:8000/similar/language?top_k=5"
```

### Analyze Text
```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{"text": "Natural language processing is fascinating", "include_embeddings": false}'
```

## Project Structure

```
gentext-app/
├── main.py              # FastAPI application with spaCy embeddings
├── bigram_model.py      # Bigram language model implementation
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
└── README.md           # This file
```

## Dependencies

- **FastAPI**: Modern web framework for building APIs
- **spaCy**: Industrial-strength NLP library
- **uvicorn**: ASGI server for FastAPI
- **numpy**: Numerical computing
- **pydantic**: Data validation using Python type annotations

## Model Information

- **Bigram Model**: Trained on a sample corpus for text generation
- **spaCy Model**: Uses `en_core_web_sm` for English word embeddings
- **Vocabulary**: Derived from training corpus (typically 40-50 unique tokens)

## API Response Examples

### Text Generation Response
```json
{
  "generated_text": "The quick brown fox jumps over",
  "start_word": "The",
  "length": 6,
  "method": "bigram_model"
}
```

### Word Embedding Response
```json
{
  "word": "language",
  "embedding": [0.1, -0.2, 0.5, ...],
  "dimensions": 96,
  "has_vector": true,
  "similarity_words": [
    {"word": "natural", "similarity": 0.8},
    {"word": "processing", "similarity": 0.7}
  ]
}
```

### Similarity Response
```json
{
  "word1": "cat",
  "word2": "dog",
  "similarity": 0.8016853
}
```

## Development

### Adding New Features

1. **Extend the API**: Add new endpoints in `main.py`
2. **Enhance the Model**: Modify `bigram_model.py` for new functionality
3. **Update Dependencies**: Add new packages to `requirements.txt`
4. **Rebuild**: `docker build -t gentext-app .`

### Testing

Test all endpoints using the interactive documentation at `/docs` or with curl commands shown above.

## Docker Details

- **Base Image**: python:3.9-slim
- **Port**: 8000
- **Health Check**: Available at `/health`
- **User**: Runs as non-root user for security
- **spaCy Model**: Automatically downloads `en_core_web_sm` during build

## Troubleshooting

### Common Issues

1. **spaCy model not found**: Ensure Docker build completed successfully
2. **Port already in use**: Stop other services on port 8000 or use `-p 8001:8000`
3. **Memory issues**: spaCy models require adequate RAM (recommend 2GB+)

### Health Check

Visit `/health` endpoint to verify both bigram model and spaCy model are loaded:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "bigram_model": "loaded",
  "spacy_model": "loaded",
  "vocabulary_size": 42
}
```

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request