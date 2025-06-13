from fastapi import FastAPI, HTTPException
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
import json

from pydantic import BaseModel
from pinecone import Pinecone
from pinecone_plugins.assistant.models.chat import Message
from typing import Optional
from dotenv import load_dotenv
import uvicorn
import os


load_dotenv() 
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('pinecone_api_key'))
assistant = pc.assistant.Assistant(assistant_name="tds-virtual-assistant")

# Request model
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

# Response model (matches expected format)
class QueryResponse(BaseModel):
    answer: str
    links: list[dict[str, str]]

@app.post("/api/", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    try:
        # Prepare message for Pinecone assistant
        message = {
            "role": "user",
            "content": request.question
        }
        
        if request.image:
            message["image"] = {
                "data": request.image
            }
        
        # Get response from Pinecone assistant
        resp = assistant.chat(messages=[message])
        
        # Assume response is already in the correct format due to system prompt
        response_data = resp["message"]["content"]
        
        # If response is a string, try to parse it as JSON (in case Pinecone returns JSON as string)
        import json
        if isinstance(response_data, str):
            response_data = json.loads(response_data)
        
        return QueryResponse(**response_data)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/api/test")
async def test(): 
    return {"response": "Test Done"}

@app.get("/")
async def test(): 
    return {"response": "Who are u?"}