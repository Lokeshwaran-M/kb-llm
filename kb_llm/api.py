from typing import Any, Dict, Optional


from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for the frontend
origins = [
    "http://localhost:5173",  # React frontend

]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



from kb_llm.llm import Model
from kb_llm.md_path import phi_3
from kb_llm.prompt import Prompt

from kb_llm.types import PromptModel
llm = Model()
llm.load(phi_3)

pmt = Prompt(phi_3["src"])


@app.get("/")
def get():
    res = llm.info()
    return res


@app.get("/get_info")
def get_db():
    res = llm.info()
    return res


@app.post("/generate")
def generate(data: PromptModel):
    
    prompt = pmt.process(data.prompt,data.context)
    res = llm.generate(prompt,max_tokens=data.max_tokens)
    return res


@app.post("/set_instruct/{instruct}")
def set_instruct(instruct: str):
    pmt.instruction = instruct


