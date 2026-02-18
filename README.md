# 🚫 SpamGuard AI - Intelligent SMS Verifier

<div align="center">
  
[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow)](https://huggingface.co/spaces/KhanzadaSalman/SpamGuard-AI)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)](https://fastapi.tiangolo.com)
  
</div>

A production-ready deep learning application that detects spam messages with **99.7% accuracy** using a 1D Convolutional Neural Network (CNN).

## ✨ Features
- **Neural Analysis:** CNN architecture for high-speed text classification
- **Microservice Design:** FastAPI backend + Glassmorphism frontend
- **Smart UX:** Real-time feedback with confidence scoring

## 🛠️ Tech Stack
- **Model:** TensorFlow, Keras (1D CNN)
- **Backend:** FastAPI, Uvicorn, Jinja2
- **Frontend:** HTML5, CSS3, JavaScript (Fetch API)

## 🚀 Live Demo
👉 **[Try it here](https://huggingface.co/spaces/KhanzadaSalman/SpamGuard-AI)**

## 📊 Performance
- **Accuracy:** 99.7%
- **Precision:** 98.5%
- **Recall:** 98.2%

## ⚡ Quick Start
```bash
git clone https://github.com/KhanzadaSalman/SpamGuard-AI.git
cd SpamGuard-AI
pip install -r requirements.txt
uvicorn app:app --reload
