import io
import os

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from pgvector.psycopg2 import register_vector
from openai import AsyncOpenAI, OpenAI
from embedding_util import generate_embeddings
#from diffusers import DiffusionPipeline
#import torch

import socketio  # Új import a Socket.IO-hoz
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import logging
import httpx
import asyncio


#uvicorn main:app --reload --host 0.0.0.0 --port 8000


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = "postgresql+psycopg2://testuser:testpwd@localhost:5432/vectordb"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ConnectedUser(Base):
    __tablename__ = "connected_users"

    id = Column(Integer, primary_key=True, index=True)
    socket_id = Column(String, unique=True, nullable=False)
    connected_at = Column(DateTime, default=datetime.utcnow)
    username = Column(String, nullable=True)



# Adatbázis táblák létrehozása
Base.metadata.create_all(bind=engine)

# Socket.IO szerver inicializálása
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio)


# Socket.IO események
@sio.event
async def connect(sid, environ):
    print(f"Client connected: {sid}")
    db = SessionLocal()
    try:
        new_user = ConnectedUser(socket_id=sid)
        db.add(new_user)
        db.commit()
    except Exception as e:
        print(f"Error saving socket_id to database: {e}")
    finally:
        db.close()
    #await sio.emit("message", {"data": "PEregoBot sikeresen csatlakozva."}, to=sid)

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    db = SessionLocal()
    try:
        user = db.query(ConnectedUser).filter(ConnectedUser.socket_id == sid).first()
        if user:
            db.delete(user)
            db.commit()
    except Exception as e:
        print(f"Error removing socket_id from database: {e}")
    finally:
        db.close()

@sio.event
async def chat_message(sid, data):
    print(f"Message from {sid}: {data}")
    await sio.emit("message", {"data": f"Echo: {data}"}, to=sid)

@sio.event
async def setUsername(sid, data):
    """
    Handles setting the username for a connected user and responds with a personalized message.
    """
    username = data.get("username")
    if not username:
        await sio.emit("error", {"message": "Username is required"}, to=sid)
        return

    db = SessionLocal()
    try:
        # Update the username for the connected user
        user = db.query(ConnectedUser).filter(ConnectedUser.socket_id == sid).first()
        if user:
            user.username = username
            db.commit()
            await sio.emit("message", {"data": f"PErgoBot sikeresen csatlakozva. Üdvözlünk:{username}!"}, to=sid)
        else:
            await sio.emit("error", {"message": "User not found"}, to=sid)
    except Exception as e:
        print(f"Error updating username: {e}")
        await sio.emit("error", {"message": "Failed to update username"}, to=sid)
    finally:
        db.close()


# FastAPI alkalmazás inicializálása
app = FastAPI()
# Socket.IO integráció a FastAPI mellé
app.mount("/socket.io", socket_app)

origins = ["http://localhost:4200",  # Engedélyezett eredet
           "http://127.0.0.1:4200",  # Alternatív localhost cím
           "*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Adatbázis kapcsolat
conn = psycopg2.connect(
    user="testuser",
    password="testpwd",
    host="localhost",
    port=5432,  # The port you exposed in docker-compose.yml
    database="vectordb"
)
register_vector(conn)
cur = conn.cursor()

retitem = ""

client = AsyncOpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',  # required, but unused
)


class Message(BaseModel):
    message: str


class ErgonomicScore(BaseModel):
    name: str
    assessor: str
    task: str
    assessment_type: str
    workstation: str
    working_duration: float
    video_length: float
    scores: dict

    @staticmethod
    def from_dict(data: dict):
        return ErgonomicScore(
            name=data["name"],
            assessor=data["assessor"],
            task=data["task"],
            assessment_type=data["assessment_type"],
            workstation=data["workstation"],
            working_duration=data["working_duration"],
            video_length=data["video_length"],
            scores={
                "upper_arm": {
                    "left": data["u_arm_score_l"],
                    "right": data["u_arm_score_r"]
                },
                "lower_arm": {
                    "left": data["lo_arm_score_l"],
                    "right": data["lo_arm_score_r"]
                },
                "wrist": {
                    "left": data["wrist_score_l"],
                    "right": data["wrist_score_r"]
                },
                "neck": data["neck_score"],
                "trunk": data["trunk_score"],
                "leg": data["leg_score"],
                "combined": {
                    "left": data["c_score_l"],
                    "right": data["c_score_r"]
                }
            }
        )


class Ai():
    pre_loaded_instructions = ""
    messages: list

    def __init__(self, client: OpenAI, model: str):
        with open('ergonomy_advice_prompt.md', 'r', encoding='utf-8') as file:
            content = file.read()

        self.pre_loaded_instructions = content
        self.client = client
        self.model = model
        self.messages = [{"role": "system",
                          "content": "You are now a professional ergonomics specialist! Max words 100! Please answer shortly!"},
                         {"role": "system", "content": self.pre_loaded_instructions}]

    async def AiResponse(self, message: str):
        self.messages.append({"role": "user", "content": message})

        response = await self.client.chat.completions.create(model=self.model, messages=self.messages)
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content

    def CreateImage(self, prompt: str):
        #image = pipeline(prompt, generator=generator).images[0]
        ##output_path = "positure.png"
        #image.save(output_path)
        #Ideiglenesen kikapcsolt képgenerálás nagoyn nag yerőforrás igány miatt
        return None


ai = Ai(client, "deepseek-r1:1.5b")


@app.get("/")
def read_root():
    response = client.chat.completions.create(
        model="deepseek-r1:1.5b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The LA Dodgers won in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
    )
    print(response.choices[0].message.content)
    return {"Hello": response.choices[0].message.content}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/ai/")
async def create_item(item: Message, doImage: bool = True):
    dbResponse = message_query(item.message)
    print(dbResponse)
    response = await ai.AiResponse(item.message)
    if doImage:
        ai.CreateImage(response)
    return response


@app.get("/message")
def message_query(query: str):
    query_embedding = generate_embeddings(query)
    try:
        cur.execute(
            """SELECT id, content, 1 - (embedding <-> %s::vector) AS cosine_similarity
            FROM items
            ORDER BY cosine_similarity DESC LIMIT 5""",
            (query_embedding,)
        )
        print("Query:", query)
        print("Most similar sentences:")
        results = []
        for row in cur.fetchall():
            result = {
                "ID": row[0],
                "CONTENT": row[1],
                "Cosine Similarity": row[2]
            }
            results.append(result)

        for result in results:
            ai.messages.append({"role": "system", "content": str(result)})

        print(result)
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_image_bytes(file_path: str) -> bytes:
    with open(file_path, "rb") as image_file:
        return image_file.read()


@app.get("/image/{filename}")
async def stream_image(filename: str):
    file_path = os.path.join(filename)
    if not os.path.exists(file_path):
        return Response(status_code=404, content="File not found")
    image_bytes = get_image_bytes(file_path)
    return StreamingResponse(io.BytesIO(image_bytes), media_type="image/png")


@app.get("/send-socket-message")
async def send_socket_message(sid: str, message: str):
    """
    Küld egy üzenetet egy adott Socket.IO kliensnek a sid alapján.
    """
    try:
        await sio.emit("message", {"data": message}, to=sid)
        return {"status": "Message sent", "sid": sid, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")

@app.get("/connected-users")
def get_connected_users():
    """
    Returns a list of connected users with their socket_id and username.
    """
    db = SessionLocal()
    try:
        users = db.query(ConnectedUser).all()
        result = [{"socket_id": user.socket_id, "username": user.username} for user in users]
        return {"connected_users": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve connected users: {str(e)}")
    finally:
        db.close()

@app.post("/ergonomics/analyze")
async def analyze_score(payload: dict, socket_id: str):
    logger.info(f"Received payload: {payload}")
    try:
        # Az adatokat az ErgonomicScore osztályba konvertáljuk
        data = ErgonomicScore.from_dict(payload)
        logger.info(f"Received ergonomic data: {data}")
        logger.info(f"Socket ID: {socket_id}")
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in payload: {e}")

    # Kivonjuk a pontszámokat a scores szótárból
    scores = data.scores
    max_u_arm = max(scores["upper_arm"]["left"], scores["upper_arm"]["right"])
    max_lo_arm = max(scores["lower_arm"]["left"], scores["lower_arm"]["right"])
    max_wrist = max(scores["wrist"]["left"], scores["wrist"]["right"])
    max_c_score = max(scores["combined"]["left"], scores["combined"]["right"])

    # Egyszerű kockázatszint becslés
    risk_level = "High" if max_c_score > 4 else "Medium" if max_c_score > 2 else "Low"
    action_required = "Immediately" if risk_level == "High" else "Observe"

    result = {
        "name": data.name,
        "assessment_type": data.assessment_type,
        "duration_seconds": data.working_duration,
        "video_seconds": data.video_length,
        "max_c_score": max_c_score,
        "risk_level": risk_level,
        "action_required": action_required,
        "risk_map": {
            "neck": scores["neck"],
            "trunk": scores["trunk"],
            "leg": scores["leg"],
            "upper_arm": max_u_arm,
            "lower_arm": max_lo_arm,
            "wrist": max_wrist
        }
    }

    # Háttérfeladat indítása az AI válasz generálására és üzenetküldésre
    async def process_ai_response():
        try:
            ai_response = await ai.AiResponse(f"Analyze the following ergonomic data: {result}")
            result["ai_response"] = ai_response
            await sio.emit("message", {"data": result["ai_response"]}, to=socket_id)
        except Exception as e:
            logger.error(f"Failed to process AI response or send message: {e}")
        finally:
            logger.info("Background task completed.")

    asyncio.create_task(process_ai_response())

    # Azonnali válasz a kliensnek
    return {"status": "Message processing started"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
