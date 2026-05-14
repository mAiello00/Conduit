"""
    LLM Microservice
"""

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from model import model, tokenizer
from model import call_llm
import os
import logging
from logging.config import dictConfig
import sys
import json
from datetime import datetime

# Environment variables
PORT = int(os.environ.get("PORT")) # 8082 for testing
LLM_ID = str(os.environ.get("LLM_ID"))

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage()
        }

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_record)

# define logging configuration
log_config = {
    "version": 1,
    "diable_existing_loggers": False,
        "formatters": {
        "json": {
            "()": JSONFormatter
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "json",
            "stream": "exit://sys.stdout",
        },
        "rotating_file": {  # This is the handler name
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "fastapi.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 5,
        },
    },
    "file": {
        "class": "logging.FileHandler",
        "level": "INFO",
        "formatter": "json",
        "filename": "llm-microservice.log",
        "mode": "a",
    },
    "loggers": {
        "app": {"handlers": ["console", "rotating_file"], "level": "DEBUG", "propagate": False},
    },
    "root": {"handlers": ["console"], "level": "DEBUG"},
}

# Message formats
class Request(BaseModel):
    request_id: str
    input: str

class Reponse(BaseModel):
    request_id: str
    output: str
    routed_to: str

# Apply logger configuraton
dictConfig(log_config)

# Create logger instance
logger = logging.getLogger("app")

# Create FastAPI app instance
app = FastAPI()

# POST Endpoint
@app.post("/generate", response_model = Reponse, status_code = 201)
async def generate_response(request: Request):
    generated = call_llm(request.input, tokenizer, model)
    print(f"{generated}\n")
    reponse = Reponse(request_id = request.request_id, output = generated, routed_to = LLM_ID)
    return reponse

if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = PORT)