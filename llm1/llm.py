from fastapi import FastAPI
from pydantic import BaseModel
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch

app = FastAPI()

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("model")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
hf_llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)

class Query(BaseModel):
    prompt: str

@app.post("/complete")
async def complete(query: Query):
    response = hf_llm.complete(query.prompt)
    return {"response": response}
