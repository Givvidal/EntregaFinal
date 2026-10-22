"""Comprueba que el Redis local para cola y estados esté disponible."""
import asyncio
import os
from dotenv import load_dotenv
from redis.asyncio import Redis
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from app.worker import REDIS_URL

async def main():
    load_dotenv()
    faltantes = [nombre for nombre in ("GOOGLE_API_KEY", "PINECONE_API_KEY") if not os.getenv(nombre)]
    if faltantes:
        raise SystemExit("Faltan variables en .env: " + ", ".join(faltantes))
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
        print(f"Redis disponible: {REDIS_URL}")
    finally:
        await client.aclose()

if __name__ == "__main__":
    asyncio.run(main())
