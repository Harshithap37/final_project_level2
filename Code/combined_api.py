# combined_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import llama3_api    as chatmod   
import proof_api     as proofmod  

app = FastAPI(title="AxiomAI — Combined API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

#mount chatapi and proofapi
app.mount("/chatapi",  chatmod.app)   
app.mount("/proofapi", proofmod.app) 

app.mount("/", chatmod.app) 