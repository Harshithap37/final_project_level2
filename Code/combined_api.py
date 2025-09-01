# combined_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import llama3_api    as chatmod   # exposes chatmod.app
import proof_api     as proofmod  # exposes proofmod.app

app = FastAPI(title="AxiomAI — Combined API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if you want
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the two apps under clean prefixes
app.mount("/chatapi",  chatmod.app)   # /chatapi/health, /chatapi/chat, /chatapi/upload, ...
app.mount("/proofapi", proofmod.app)  # /proofapi/health, /proofapi/prove