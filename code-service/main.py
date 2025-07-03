import random, string
import aioredis
from fastapi import FastAPI, HTTPException, Body

app = FastAPI(title="Leadinc Code Service")
redis = aioredis.from_url("redis://localhost:6379", decode_responses=True)

def generate_code():
    return ''.join(random.choices(string.digits, k=6))

@app.post("/api/generate-code")
async def generate_code_api(payload: dict = Body(None)):
    code = generate_code()
    while await redis.exists(f"tgcode:{code}"):
        code = generate_code()
    await redis.set(f"tgcode:{code}", "issued", ex=600) 
    await redis.set(f"real_code:{code}", "issued", ex=600)
    return {"code": code}

@app.post("/api/verify-code")
async def verify_code_api(payload: dict = Body(...)):
    code = payload.get("code")
    if not code or len(str(code)) != 6:
        raise HTTPException(status_code=400, detail="Код невалидный")
    val = await redis.get(f"tgcode:{code}")
    if val == "issued":
        await redis.delete(f"tgcode:{code}")
        return {"status": "ok"}
    raise HTTPException(status_code=400, detail="Код не найден или уже использован")
