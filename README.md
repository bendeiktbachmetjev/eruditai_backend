# Erudit AI Backend

Backend API for generating flashcards using Google Gemini. This service processes highlighted text from books and generates educational flashcards.

## Features

- Generate flashcards from highlighted text using Google Gemini 1.5 Flash
- RESTful API with FastAPI
- CORS enabled for cross-origin requests
- Health check endpoints

## Setup

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

3. Run the server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Environment Variables

- `GOOGLE_API_KEY` (required): Your Google Gemini API key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))
- `PORT` (optional): Server port (defaults to 8000)

## API Endpoints

### Health Check
- `GET /` - Root endpoint
- `GET /health` - Health check

### Generate Flashcards
- `POST /api/generate-flashcards`

**Request Body:**
```json
{
  "highlightedText": "Text from the book",
  "bookTitle": "Book Title",
  "count": 3
}
```

**Response:**
```json
[
  {
    "question": "What is...?",
    "answer": "The answer is..."
  },
  ...
]
```

## Deployment on Railway

1. Push this code to GitHub
2. Connect your GitHub repository to Railway
3. Add environment variable `GOOGLE_API_KEY` in Railway dashboard (get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey))
4. Railway will automatically detect the Python project and deploy it
5. The `Procfile` tells Railway how to run the application

## API Documentation

Once the server is running, you can access:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

