import os, io, base64, hashlib, hmac, secrets, math
from datetime import datetime, timezone, timedelta
from typing import Optional

import httpx
import jwt
from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel
from pymongo import MongoClient, ASCENDING

APP_NAME = "AI Navigator"
JWT_SECRET = os.getenv("JWT_SECRET", "change-me-in-render")
JWT_ALGORITHM = "HS256"
JWT_DAYS = 30

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_KEY = os.getenv("CLOUDFLARE_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")

if not MONGODB_URI:
    raise RuntimeError("MONGODB_URI is required")

mongo = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=8000)
db = mongo[os.getenv("MONGODB_DB", "ai_navigator")]
users = db["users"]
routes = db["routes"]
users.create_index("email", unique=True)
routes.create_index([("user_id", ASCENDING), ("created_at", -1)])

app = FastAPI(title=APP_NAME)

class RegisterBody(BaseModel):
    email: str
    password: str

class LoginBody(BaseModel):
    email: str
    password: str

class RouteBody(BaseModel):
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    destination_label: Optional[str] = None

class SearchBody(BaseModel):
    query: str

class WeatherBody(BaseModel):
    lat: float
    lon: float

class VoiceCommandBody(BaseModel):
    transcript: str
    current_lat: Optional[float] = None
    current_lon: Optional[float] = None

class LocationBody(BaseModel):
    lat: float
    lon: float


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210_000)
    return base64.urlsafe_b64encode(salt + digest).decode()


def verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode())
        salt, expected = raw[:16], raw[16:]
        actual = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 210_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def make_token(user_id: str) -> str:
    payload = {"sub": user_id, "exp": datetime.now(timezone.utc) + timedelta(days=JWT_DAYS)}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Authentication required")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = users.find_one({"_id": payload["sub"]})
        if not user:
            raise HTTPException(401, "User not found")
        return user
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token")


async def api_json(method: str, url: str, **kwargs):
    async with httpx.AsyncClient(timeout=35, follow_redirects=True) as client:
        r = await client.request(method, url, **kwargs)
        if r.status_code >= 400:
            raise HTTPException(502, f"External API error {r.status_code}: {r.text[:300]}")
        return r.json()


async def geocode(query: str):
    q = query.strip()
    # Coordinates: "lat, lon"
    parts = [x.strip() for x in q.replace(";", ",").split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return {"lat": lat, "lon": lon, "name": f"{lat:.5f}, {lon:.5f}"}
        except ValueError:
            pass
    data = await api_json("GET", "https://nominatim.openstreetmap.org/search", params={
        "q": q, "format": "jsonv2", "limit": 1, "addressdetails": 1
    }, headers={"User-Agent": "AI-Navigator/1.0"})
    if not data:
        raise HTTPException(404, "Место не найдено")
    x = data[0]
    return {"lat": float(x["lat"]), "lon": float(x["lon"]), "name": x.get("display_name", q)}


async def route_osrm(start_lat, start_lon, end_lat, end_lon):
    url = f"https://router.project-osrm.org/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
    data = await api_json("GET", url, params={"overview": "full", "geometries": "geojson", "steps": "true"})
    if data.get("code") != "Ok" or not data.get("routes"):
        raise HTTPException(400, "Не удалось построить маршрут")
    r = data["routes"][0]
    steps = []
    for leg in r.get("legs", []):
        for s in leg.get("steps", []):
            maneuver = s.get("maneuver", {})
            steps.append({
                "distance": round(s.get("distance", 0)),
                "duration": round(s.get("duration", 0)),
                "name": s.get("name") or "дорогу",
                "type": maneuver.get("type"),
                "modifier": maneuver.get("modifier"),
            })
    return {
        "distance_m": round(r["distance"]),
        "duration_s": round(r["duration"]),
        "geometry": r["geometry"],
        "steps": steps,
    }


async def reverse_geocode(lat, lon):
    data = await api_json("GET", "https://nominatim.openstreetmap.org/reverse", params={
        "lat": lat, "lon": lon, "format": "jsonv2", "zoom": 18
    }, headers={"User-Agent": "AI-Navigator/1.0"})
    return data.get("display_name", f"{lat:.5f}, {lon:.5f}")


async def weather(lat, lon):
    if not OPENWEATHER_API_KEY:
        return None
    return await api_json("GET", "https://api.openweathermap.org/data/2.5/weather", params={
        "lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "ru"
    })


async def groq_chat(messages, model="openai/gpt-oss-120b"):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not configured")
    data = await api_json("POST", "https://api.groq.com/openai/v1/chat/completions", headers={
        "Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"
    }, json={"model": model, "messages": messages, "temperature": 0.2})
    return data["choices"][0]["message"]["content"]


async def cloudflare_image(prompt: str):
    if not CLOUDFLARE_ACCOUNT_ID or not CLOUDFLARE_API_KEY:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT_ID}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    data = await api_json("POST", url, headers={
        "Authorization": f"Bearer {CLOUDFLARE_API_KEY}", "Content-Type": "application/json"
    }, json={"prompt": prompt, "steps": 4})
    result = data.get("result", {})
    image = result.get("image")
    return f"data:image/jpeg;base64,{image}" if image else None


async def groq_tts(text: str):
    if not GROQ_API_KEY:
        return None
    url = "https://api.groq.com/openai/v1/audio/speech"
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}, json={
            "model": "playai-tts", "voice": "Fritz-PlayAI", "input": text, "response_format": "mp3"
        })
        if r.status_code >= 400:
            return None
        return f"data:audio/mpeg;base64,{base64.b64encode(r.content).decode()}"


async def groq_transcribe(file: UploadFile):
    if not GROQ_API_KEY:
        raise HTTPException(500, "GROQ_API_KEY is not configured")
    content = await file.read()
    files = {"file": (file.filename or "audio.webm", content, file.content_type or "audio/webm")}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post("https://api.groq.com/openai/v1/audio/transcriptions", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, files=files, data={"model": "whisper-large-v3-turbo", "language": "ru"})
        if r.status_code >= 400:
            raise HTTPException(502, f"Groq transcription error: {r.text[:300]}")
        return r.json().get("text", "")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse("index.html")

@app.get("/style.css")
async def css():
    return FileResponse("style.css", media_type="text/css")

@app.get("/app.js")
async def js():
    return FileResponse("app.js", media_type="application/javascript")

@app.get("/health")
async def health():
    return {"ok": True, "mongodb": bool(MONGODB_URI), "groq": bool(GROQ_API_KEY), "cloudflare": bool(CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_KEY), "openweather": bool(OPENWEATHER_API_KEY)}

@app.post("/api/register")
async def register(body: RegisterBody):
    email = body.email.strip().lower()
    if len(body.password) < 6:
        raise HTTPException(400, "Пароль должен содержать минимум 6 символов")
    user_id = secrets.token_hex(16)
    try:
        users.insert_one({"_id": user_id, "email": email, "password": hash_password(body.password), "created_at": datetime.now(timezone.utc).isoformat()})
    except Exception as e:
        if "duplicate" in str(e).lower():
            raise HTTPException(409, "Пользователь уже существует")
        raise
    return {"token": make_token(user_id), "email": email}

@app.post("/api/login")
async def login(body: LoginBody):
    user = users.find_one({"email": body.email.strip().lower()})
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(401, "Неверный email или пароль")
    return {"token": make_token(user["_id"]), "email": user["email"]}

@app.get("/api/me")
async def me(user=Depends(current_user)):
    return {"email": user["email"]}

@app.post("/api/search")
async def search(body: SearchBody, user=Depends(current_user)):
    return await geocode(body.query)

@app.post("/api/route")
async def build_route(body: RouteBody, user=Depends(current_user)):
    r = await route_osrm(body.start_lat, body.start_lon, body.end_lat, body.end_lon)
    place = await reverse_geocode(body.end_lat, body.end_lon)
    route_id = secrets.token_hex(12)
    doc = {
        "_id": route_id, "user_id": user["_id"], "start": {"lat": body.start_lat, "lon": body.start_lon},
        "destination": {"lat": body.end_lat, "lon": body.end_lon, "label": body.destination_label or place},
        "distance_m": r["distance_m"], "duration_s": r["duration_s"], "steps": r["steps"],
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    routes.insert_one(doc)
    return {"route_id": route_id, **r, "place": place}

@app.post("/api/route/start-image")
async def start_image(body: LocationBody, user=Depends(current_user)):
    place = await reverse_geocode(body.lat, body.lon)
    prompt = (f"Ultra realistic cinematic aerial street-level view of {place}. "
              "Highly detailed real-world navigation atmosphere, natural lighting, realistic roads, buildings, vegetation, "
              "photorealistic, no text, no map labels, no fantasy.")
    image = await cloudflare_image(prompt)
    return {"place": place, "image": image}

@app.post("/api/weather")
async def get_weather(body: WeatherBody, user=Depends(current_user)):
    return await weather(body.lat, body.lon)

@app.post("/api/voice/transcribe")
async def transcribe(file: UploadFile = File(...), user=Depends(current_user)):
    text = await groq_transcribe(file)
    return {"transcript": text}

@app.post("/api/voice/command")
async def voice_command(body: VoiceCommandBody, user=Depends(current_user)):
    current = f"Текущая позиция: {body.current_lat}, {body.current_lon}." if body.current_lat is not None else "Текущая позиция неизвестна."
    result = await groq_chat([
        {"role": "system", "content": "Ты голосовой помощник навигатора. Пользователь говорит по-русски. Верни только краткую команду/описание назначения для навигации. Если это место или адрес — сохрани его максимально точно. Не выдумывай координаты."},
        {"role": "user", "content": f"{current}\nГолосовая команда: {body.transcript}"}
    ])
    return {"text": result}

@app.post("/api/navigation/instruction")
async def instruction(body: VoiceCommandBody, user=Depends(current_user)):
    text = await groq_chat([
        {"role": "system", "content": "Ты голосовой навигатор. Отвечай одной очень короткой инструкцией на русском, без приветствий. Пример: Через 200 метров поверните направо."},
        {"role": "user", "content": body.transcript}
    ])
    audio = await groq_tts(text)
    return {"text": text, "audio": audio}

@app.get("/api/history")
async def history(user=Depends(current_user)):
    docs = list(routes.find({"user_id": user["_id"]}, {"_id": 1, "destination": 1, "distance_m": 1, "duration_s": 1, "created_at": 1}).sort("created_at", -1).limit(50))
    for d in docs:
        d["id"] = d.pop("_id")
    return docs
