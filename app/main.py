import os
from fastapi import FastAPI
import openai
from .routes.biomechanics_feedback_router import router
from fastapi.middleware.cors import CORSMiddleware


app_principal = FastAPI(title="Biomechanics Feedback API", version="0.1.0")

app_principal.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app_principal.include_router(router)
