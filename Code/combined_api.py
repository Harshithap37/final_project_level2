# combined_app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import llama3_api       as chatmod
import proof_api        as proofmod

apps = FastAPI(title="AxiomAI - Combined API")
apps.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

#mount chatapi and proofapi
apps.mount("/chatapi", chatmod.app)
apps.mount("/proofapi", proofmod.app)

apps.mount("/", chatmod.app)