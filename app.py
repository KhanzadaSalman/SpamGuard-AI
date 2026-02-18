from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model and tokenizer
model = tf.keras.models.load_model("./cnn_spam_model.keras")
with open("./tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class Message(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict_spam(msg: Message):
    seq = tokenizer.texts_to_sequences([msg.text.lower()])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=50)
    prediction = model.predict(padded, verbose=0)[0][0]
    
    label = "SPAM" if prediction > 0.5 else "HAM"
    conf_number = float(prediction * 100) if label == "SPAM" else float((1 - prediction) * 100)
    
    return {"label": label, "confidence": conf_number}