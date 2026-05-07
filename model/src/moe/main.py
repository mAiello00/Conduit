"""
    LLM Microservice
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import model, tokenizer
from model import call_llm

class Request(BaseModel):
    request_id: str
    input: str

class Reponse(BaseModel):
    request_id: str
    output: str
    routed_to: str

# Create FastAPI app instance
app = FastAPI()

# Define POST Endpoint
@app.post("/generate", response_model = Reponse, status_code = 201)
async def generate_response(request: Request):
    generated = call_llm(request.input, tokenizer, model)
    print(f"{generated}\n")
    reponse = Reponse(request_id = request.request_id, output = generated, routed_to = "1")
    return reponse

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 8082)