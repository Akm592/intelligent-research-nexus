# services/kg_service/app/main.py
from fastapi import FastAPI
import logging

logger = logging.getLogger("IRN_Core").getChild("KGService")
app = FastAPI(title="Knowledge Graph Service (Placeholder)")

@app.get("/health")
async def health_check():
    logger.info("KG Service health check")
    return {"status": "ok", "message": "KG Service is running (placeholder)"}

