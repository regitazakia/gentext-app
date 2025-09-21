# Gentext-App: Bigram Text Generation and Word Embeddings API

A FastAPI application that combines bigram-based text generation with spaCy word embeddings for natural language processing tasks. This project demonstrates the implementation of both traditional n-gram models and modern word embedding techniques in a production-ready API.

## 🚀 Features

- **Bigram Text Generation**: Generate coherent text using trained bigram language models
- **spaCy Word Embeddings**: Extract semantic word vectors using pre-trained spaCy models
- **Word Similarity Analysis**: Calculate semantic similarity between words
- **Text Analysis**: Comprehensive NLP analysis including POS tagging and named entity recognition
- **Docker Support**: Fully containerized application with multi-stage builds
- **Interactive API Documentation**: Auto-generated Swagger UI and ReDoc documentation
- **Health Monitoring**: Built-in health checks and monitoring endpoints
- **Modern Python Stack**: Built with FastAPI, Pydantic, and async support

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **spaCy**: Industrial-strength natural language processing
- **UV**: Fast Python package installer and resolver
- **Docker**: Containerization with multi-stage builds
- **Pydantic**: Data validation using Python type annotations
- **NumPy**: Numerical computing for vector operations

## 📋 Prerequisites

- Docker and Docker Desktop
- Python 3.9+ (for local development)
- Git

## 🚀 Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/regitazakia/gentext-app.git
   cd gentext-app
   ```

2. **Build and run with Docker**
   ```bash
   docker build -t gentext-app .
   docker run -p 8000:80 gentext-app
   ```

3. **Access the application**
   - **API**: http://localhost:8000
   - **Interactive Documentation**: http://localhost:8000/docs
   - **Alternative Docs**: http://localhost:8000/redoc
   - **Health Check**: http://localhost:8000/health

### Local Development

1. **Install dependencies using UV**
   ```bash
   uv sync
   uv run python -m spacy download en_core_web_sm
   ```

2. **Run the application**
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📚 API Documentation

### Text Generation

Generate text using bigram probability models:

```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "start_word": "The",
       "length": 15
     }'
```

**Response:**
```json
{
  "generated_text": "The quick brown fox jumps over the lazy dog",
  "start_word": "The",
  "length": 15,
  "method": "bigram_model"
}
```

### Word Embeddings

Extract semantic word vectors using spaCy:

```bash
curl -X POST "http://localhost:8000/embedding" \
     -H "Content-Type: application/json" \
     -d '{
       "word": "language",
       "include_similarity": true,
       "top_similar": 5
     }'
```

**Response:**
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

### Word Similarity

Calculate semantic similarity between words:

```bash
curl -X POST "http://localhost:8000/similarity" \
     -H "Content-Type: application/json" \
     -d '{
       "word1": "cat",
       "word2": "dog"
     }'
```

### Text Analysis

Perform comprehensive NLP analysis:

```bash
curl -X POST "http://localhost:8000/analyze" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Natural language processing is fascinating",
       "include_embeddings": false
     }'
```

## 🏗️ Project Structure

```
gentext-app/
├── app/
│   ├── main.py              # FastAPI application with spaCy embeddings
│   └── bigram_model.py      # Bigram language model implementation
├── requirements.txt         # Python dependencies (legacy)
├── pyproject.toml          # UV project configuration
├── uv.lock                 # UV lock file
├── Dockerfile              # Multi-stage Docker build
├── .dockerignore          # Docker ignore patterns
├── .gitignore             # Git ignore patterns
└── README.md              # Project documentation
```

## 🔧 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information and available endpoints |
| POST | `/generate` | Generate text using bigram model |
| POST | `/embedding` | Get word embedding with similarity analysis |
| GET | `/embedding/{word}` | Simple word embedding lookup |
| POST | `/similarity` | Calculate similarity between two words |
| GET | `/similar/{word}` | Find similar words from vocabulary |
| POST | `/analyze` | Comprehensive text analysis with spaCy |
| GET | `/vocabulary` | Get bigram model vocabulary |
| GET | `/health` | Health check and system status |

## 🐳 Docker Configuration

The application uses a multi-stage Docker build optimized for production:

- **Base Image**: `python:3.9-slim`
- **Package Manager**: UV for fast dependency resolution
- **Port**: Exposed on port 80 internally
- **Health Checks**: Built-in health monitoring
- **Security**: Runs as non-root user
- **Optimization**: Multi-stage build for smaller image size

### Build Arguments

```bash
# Build with custom port
docker build --build-arg PORT=8080 -t gentext-app .

# Run with port mapping
docker run -p 8000:80 gentext-app
```

## 🧪 Testing the API

### Health Check
```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "bigram_model": "loaded",
  "spacy_model": "loaded",
  "vocabulary_size": 42
}
```

### Interactive Testing

Visit http://localhost:8000/docs for the interactive Swagger UI where you can:
- Test all endpoints directly from the browser
- View request/response schemas
- See example payloads
- Download OpenAPI specifications

## 🔄 Development Workflow

### Adding New Features

1. **Extend the API**: Add new endpoints in `app/main.py`
2. **Enhance Models**: Modify `app/bigram_model.py` for new functionality
3. **Update Dependencies**: Add packages using `uv add package-name`
4. **Test Changes**: Use the interactive docs or curl commands
5. **Rebuild Container**: `docker build -t gentext-app .`

### Code Quality

The project follows modern Python development practices:
- Type hints throughout the codebase
- Pydantic models for request/response validation
- Comprehensive error handling
- Async/await support where applicable

## 📊 Model Information

### Bigram Model
- **Training Data**: Sample corpus of literary and technical texts
- **Vocabulary Size**: ~40-50 unique tokens
- **Smoothing**: Laplace smoothing for unseen bigrams
- **Generation**: Temperature-controlled sampling

### spaCy Model
- **Model**: `en_core_web_sm` (English small model)
- **Vector Dimensions**: 96-dimensional word vectors
- **Capabilities**: POS tagging, NER, similarity calculations
- **Language**: English

## 🚨 Troubleshooting

### Common Issues

1. **spaCy model not found**
   - Ensure Docker build completed successfully
   - Check that `en_core_web_sm` is properly downloaded

2. **Port conflicts**
   - Use different port mapping: `docker run -p 8001:80 gentext-app`
   - Stop conflicting services: `docker stop $(docker ps -q)`

3. **Memory issues**
   - spaCy models require adequate RAM (recommend 2GB+)
   - Use Docker Desktop with sufficient memory allocation

4. **Build failures**
   - Ensure UV is properly installed in the container
   - Check that all files are present in the build context

### Debugging

Enable verbose logging by setting environment variables:

```bash
docker run -p 8000:80 -e LOG_LEVEL=debug gentext-app
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit your changes: `git commit -am 'Add new feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy**: For providing excellent NLP capabilities
- **FastAPI**: For the modern, fast web framework
- **UV**: For fast and reliable Python package management
- **Docker**: For containerization support

## 📞 Contact

- **Repository**: [https://github.com/regitazakia/gentext-app](https://github.com/regitazakia/gentext-app)
- **Issues**: [GitHub Issues](https://github.com/regitazakia/gentext-app/issues)

---
