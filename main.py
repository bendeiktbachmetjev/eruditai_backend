"""
FastAPI Backend for Erudit AI Flashcard Generation
Uses Google Gemini API to generate flashcards from highlighted text
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os
import json
import re
import logging
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Erudit AI Backend",
    description="Backend API for generating flashcards using Google Gemini",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Google Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.warning("GOOGLE_API_KEY not set. Please set it in environment variables.")
else:
    genai.configure(api_key=google_api_key)

# Request/Response Models
class GenerateFlashcardsRequest(BaseModel):
    highlightedText: str = Field(..., description="The highlighted text from the book")
    bookTitle: str = Field(..., description="Title of the book")
    count: int = Field(default=3, ge=1, le=10, description="Number of flashcards to generate (1-10)")

class FlashcardResponse(BaseModel):
    question: str = Field(..., description="The question for the flashcard")
    answer: str = Field(..., description="The answer for the flashcard")

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "Erudit AI Backend is running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Main endpoint for generating flashcards
@app.post("/api/generate-flashcards", response_model=List[FlashcardResponse])
async def generate_flashcards(request: GenerateFlashcardsRequest):
    """
    Generate flashcards from highlighted text using Google Gemini
    """
    if not google_api_key:
        raise HTTPException(
            status_code=500,
            detail="Google API key not configured. Please set GOOGLE_API_KEY environment variable."
        )
    
    try:
        logger.info(f"Generating {request.count} flashcards for book: {request.bookTitle}")
        
        # Load prompt template from file
        prompt_template_path = os.path.join(os.path.dirname(__file__), "prompt_template.txt")
        try:
            with open(prompt_template_path, "r", encoding="utf-8") as f:
                prompt_template = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found at {prompt_template_path}")
            raise HTTPException(
                status_code=500,
                detail="Prompt template file not found"
            )
        
        # Format prompt with request data
        prompt = prompt_template.format(
            highlightedText=request.highlightedText
        )

        # Prepare full prompt with system instruction
        full_prompt = f"""You are a flashcard creation master. You create high-quality flashcards that help people memorize information. You MUST return ONLY valid JSON arrays, no markdown formatting, no code blocks, no explanations.

{prompt}"""

        # Call Google Gemini API
        # Using Gemini 2.5 models (older models are deprecated)
        # Try different model names (cheapest first)
        model_names = ['gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-2.0-flash-exp']
        response = None
        last_error = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                logger.info(f"Attempting to use Gemini model: {model_name}")
                # Try to generate content to verify model works
                response = model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=2000,
                    )
                )
                logger.info(f"Successfully using Gemini model: {model_name}")
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Model {model_name} failed: {str(e)}")
                continue
        
        if response is None:
            raise HTTPException(
                status_code=500,
                detail=f"No available Gemini model found. Tried: {', '.join(model_names)}. Last error: {str(last_error)}"
            )
        
        # Extract response content
        content = response.text.strip()
        
        # Clean up response (remove markdown code blocks if present)
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Fix trailing commas in JSON (common issue with AI responses)
        # Remove trailing commas before closing brackets/braces
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        # Parse JSON response
        try:
            flashcards_data = json.loads(content)
            
            # Handle case where response might be wrapped in an object
            if isinstance(flashcards_data, dict):
                # Try to find array in common keys
                for key in ["flashcards", "cards", "items", "data"]:
                    if key in flashcards_data and isinstance(flashcards_data[key], list):
                        flashcards_data = flashcards_data[key]
                        break
                # If still a dict, try to extract values
                if isinstance(flashcards_data, dict):
                    # Check if it's a single flashcard object
                    if "question" in flashcards_data and "answer" in flashcards_data:
                        flashcards_data = [flashcards_data]
                    else:
                        raise ValueError("Unexpected JSON structure")
            
            # Validate and convert to response format
            flashcards = []
            for item in flashcards_data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    flashcards.append(FlashcardResponse(
                        question=str(item["question"]).strip(),
                        answer=str(item["answer"]).strip()
                    ))
                else:
                    logger.warning(f"Invalid flashcard format: {item}")
            
            # Ensure we have at least one flashcard
            if len(flashcards) == 0:
                raise ValueError("No valid flashcards generated")
            
            # Ensure we have the requested number of flashcards
            if len(flashcards) < request.count:
                logger.warning(f"Generated {len(flashcards)} flashcards, but {request.count} were requested")
            
            # Return only the requested count
            return flashcards[:request.count]
            
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {content[:500]}")  # Log first 500 chars
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse AI response. Please try again."
            )
            
    except Exception as e:
        logger.error(f"Error generating flashcards: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error generating flashcards: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

