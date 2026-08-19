#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 15:33:35 2026
@author: madhupahar
"""

import os
import io
import joblib
import numpy as np
import pandas as pd
import librosa
import spacy
import subprocess
import tempfile
from collections import defaultdict
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from dotenv import load_dotenv
import uvicorn
from typing import List, Dict, Optional
import warnings
import groq  # NEW: Groq for ASR

warnings.filterwarnings("ignore")

# ============================================
# Load Environment Variables
# ============================================
load_dotenv()

# ============================================
# Configuration
# ============================================
MODEL_DIR = "models"
MAX_AUDIO_DURATION = 180  # 3 minutes
MIN_AUDIO_DURATION = 10   # 10 seconds

# Features for spider diagram
SPIDER_FEATURES = [
    'silence_ratio',
    'ttr',
    'content_ratio',
    'mlu',
    'word_count',
    'pitch_std'
]

# ============================================
# Load Artifacts at Startup
# ============================================
print("Loading artifacts...")

model = joblib.load(f"{MODEL_DIR}/cognitive_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feature_cols = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
print(f"✅ Loaded model with {len(feature_cols)} features")

normative_stats = None
try:
    normative_stats = joblib.load(f"{MODEL_DIR}/normative_stats.pkl")
    print("✅ Normative stats loaded")
except Exception as e:
    print(f"⚠️ Normative stats not found: {e}")

nlp = spacy.load("en_core_web_sm")
print("✅ spaCy loaded")

# ---- Initialize Groq client ----
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    print("⚠️ GROQ_API_KEY not found. ASR will fail.")
groq_client = groq.Groq(api_key=groq_api_key)
print("✅ Groq client initialized")

# ============================================
# Initialize LangChain (Azure OpenAI / Phi-4)
# ============================================
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

if azure_endpoint and azure_api_key and azure_deployment:
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=azure_api_key,
            api_version=os.getenv("OPENAI_API_VERSION", "2024-02-15-preview"),
            deployment_name=azure_deployment,
            temperature=0.1
        )
        print(f"✅ LangChain initialized with deployment: {azure_deployment}")
    except Exception as e:
        print(f"⚠️ Error initializing LangChain: {e}")
        llm = None
else:
    print("⚠️ Azure OpenAI credentials not found. Explanations disabled.")
    llm = None

# ============================================
# LangChain Prompt Template
# ============================================
explanation_prompt = PromptTemplate(
    input_variables=[
        "risk_score", "risk_text", "transcript",
        "silence_ratio", "ttr", "content_ratio",
        "speech_time", "word_count", "mlu",
        "stop_ratio", "pronoun_ratio", "repetition_rate", "pitch_std"
    ],
    template="""
You are a compassionate clinical AI assistant explaining cognitive screening results to a patient.

The patient described the Cookie Theft picture. Here are the key speech patterns detected:

- Transcript: "{transcript}"
- Silence ratio (pauses): {silence_ratio:.3f} (higher = more pauses)
- Vocabulary diversity (TTR): {ttr:.3f} (lower = less diverse vocabulary)
- Content word usage: {content_ratio:.3f} (lower = fewer meaningful words)
- Speech time: {speech_time:.1f} seconds
- Word count: {word_count:.0f} words
- Mean utterance length (MLU): {mlu:.1f} words
- Stopword ratio: {stop_ratio:.3f} (higher = more filler words)
- Pronoun ratio: {pronoun_ratio:.3f}
- Repetition rate: {repetition_rate:.3f}
- Pitch variation: {pitch_std:.1f} Hz (lower variation = less vocal expressiveness)

The model predicts a {risk_text} risk ({risk_score:.0%}) of cognitive impairment.

Generate a brief, empathetic explanation (3-4 sentences) that:
1. Acknowledges the patient's effort
2. Explains what the speech patterns suggest in simple language
3. Provides a recommendation
4. Includes a clear disclaimer that this is NOT a clinical diagnosis

Keep it warm, supportive, and informative.
"""
)

if llm:
    explanation_chain = LLMChain(llm=llm, prompt=explanation_prompt)
else:
    explanation_chain = None

# ============================================
# Helper: Convert to WAV (for librosa and Groq)
# ============================================
def convert_to_wav(input_bytes):
    """Convert any audio bytes to a proper 16kHz mono WAV file using ffmpeg."""
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as f:
        f.write(input_bytes)
        input_path = f.name

    output_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    try:
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-y",
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return output_path
    except Exception as e:
        print(f"FFmpeg conversion failed: {e}")
        with open(output_path, 'wb') as f:
            f.write(input_bytes)
        return output_path
    finally:
        if os.path.exists(input_path):
            os.unlink(input_path)

# ============================================
# Feature Extraction Functions
# ============================================

def extract_acoustic_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=MAX_AUDIO_DURATION)
        duration = len(y) / sr
        
        if duration < 5.0:
            return None
        
        rms = librosa.feature.rms(y=y).mean()
        
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
        else:
            pitch_mean = 0
            pitch_std = 0
        
        intervals = librosa.effects.split(y, top_db=30)
        silence_duration = sum([end - start for start, end in intervals]) / sr
        silence_ratio = silence_duration / duration if duration > 0 else 0
        speech_time = duration - silence_duration
        
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        if len(pitch_values) > 1:
            jitter = np.std(np.diff(pitch_values)) / (np.mean(pitch_values) + 1e-8)
            rms_frames = librosa.feature.rms(y=y, frame_length=1024, hop_length=512).flatten()
            shimmer = np.std(rms_frames) / (np.mean(rms_frames) + 1e-8)
        else:
            jitter = 0
            shimmer = 0
        
        return {
            'duration': duration,
            'speech_time': speech_time,
            'silence_ratio': silence_ratio,
            'rms': rms,
            'pitch_mean': pitch_mean,
            'pitch_std': pitch_std,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'jitter': jitter,
            'shimmer': shimmer
        }
    except Exception as e:
        print(f"Error in acoustic extraction: {e}")
        return None

def extract_linguistic_features(text):
    if not text or text.strip() == "":
        return None
    
    doc = nlp(text)
    words = [t.text for t in doc if not t.is_punct and not t.is_space]
    
    if len(words) == 0:
        return None
    
    word_count = len(words)
    unique_words = len(set([w.lower() for w in words]))
    ttr = unique_words / word_count if word_count > 0 else 0
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    stopwords = [t for t in doc if t.is_stop]
    stop_ratio = len(stopwords) / len(doc) if len(doc) > 0 else 0
    
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    content_words = [t for t in doc if t.pos_ in content_pos]
    content_ratio = len(content_words) / word_count if word_count > 0 else 0
    
    fillers = ['um', 'uh', 'like', 'er', 'ah', 'you know', 'i mean']
    filler_count = sum([1 for t in doc if t.text.lower() in fillers])
    filler_rate = filler_count / word_count if word_count > 0 else 0
    
    pos_tags = [t.pos_ for t in doc]
    pos_diversity = len(set(pos_tags)) / len(pos_tags) if len(pos_tags) > 0 else 0
    
    sentences = [sent for sent in doc.sents]
    mlu = np.mean([len([t for t in sent if not t.is_punct]) for sent in sentences]) if sentences else word_count
    
    repeats = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i-1].lower())
    repetition_rate = repeats / word_count if word_count > 0 else 0
    
    pronoun_count = sum(1 for t in doc if t.pos_ == 'PRON')
    pronoun_ratio = pronoun_count / word_count if word_count > 0 else 0
    
    return {
        'word_count': word_count,
        'ttr': ttr,
        'avg_word_len': avg_word_len,
        'stop_ratio': stop_ratio,
        'content_ratio': content_ratio,
        'filler_rate': filler_rate,
        'pos_diversity': pos_diversity,
        'mlu': mlu,
        'repetition_rate': repetition_rate,
        'pronoun_ratio': pronoun_ratio
    }

def extract_features_from_path(audio_path, transcript):
    try:
        acoustic = extract_acoustic_features(audio_path)
        if acoustic is None:
            return None
        
        linguistic = extract_linguistic_features(transcript)
        if linguistic is None:
            return None
        
        features = {**acoustic, **linguistic}
        feature_values = [features.get(col, 0) for col in feature_cols]
        return np.array(feature_values).reshape(1, -1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# ============================================
# Explanation Function
# ============================================
def generate_explanation(risk_score, transcript, features):
    if explanation_chain is None:
        return "Explanation unavailable. Please contact the developer."
    
    risk_text = "High" if risk_score > 0.5 else "Low"
    
    try:
        explanation = explanation_chain.run(
            risk_score=risk_score,
            risk_text=risk_text,
            transcript=transcript,
            silence_ratio=features.get('silence_ratio', 0),
            ttr=features.get('ttr', 0),
            content_ratio=features.get('content_ratio', 0),
            speech_time=features.get('speech_time', 0),
            word_count=features.get('word_count', 0),
            mlu=features.get('mlu', 0),
            stop_ratio=features.get('stop_ratio', 0),
            pronoun_ratio=features.get('pronoun_ratio', 0),
            repetition_rate=features.get('repetition_rate', 0),
            pitch_std=features.get('pitch_std', 0)
        )
        return explanation
    except Exception as e:
        print(f"Error generating explanation: {e}")
        return "Explanation temporarily unavailable. Please consult a healthcare professional."

# ============================================
# Spider Diagram Function
# ============================================
def compute_spider_data(features):
    if normative_stats is None:
        return None
    
    spider_data = {}
    for feature in SPIDER_FEATURES:
        user_value = features.get(feature, 0)
        norm = normative_stats.get(feature)
        if norm is None:
            continue
        
        higher_is_better = feature in ['ttr', 'content_ratio', 'mlu', 'word_count']
        lower_is_better = feature in ['silence_ratio']
        
        mean = norm['mean']
        std = norm['std'] + 1e-8
        
        if higher_is_better:
            z_score = (user_value - mean) / std
            normalized = min(1, max(0, 0.5 + z_score / 4))
        elif lower_is_better:
            z_score = (mean - user_value) / std
            normalized = min(1, max(0, 0.5 + z_score / 4))
        else:
            z_score = abs(user_value - mean) / std
            normalized = min(1, max(0, 1 - z_score / 3))
        
        spider_data[feature] = {
            'user_value': float(user_value),
            'normalized_score': round(normalized, 3),
            'healthy_mean': float(mean),
            'healthy_std': float(std),
            'z_score': round((user_value - mean) / std, 2)
        }
    
    return spider_data

# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="Cognitive Decline Predictor",
    description="Predict cognitive decline from voice recordings of Cookie Theft picture description.",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# Usage Limit Middleware (3 attempts per IP)
# ============================================
usage_tracker = defaultdict(int)
MAX_USAGE_PER_IP = 3

@app.middleware("http")
async def limit_usage_middleware(request: Request, call_next):
    client_ip = request.client.host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    
    if request.url.path in ["/health", "/"]:
        return await call_next(request)
    
    if usage_tracker[client_ip] >= MAX_USAGE_PER_IP:
        return JSONResponse(
            status_code=429,
            content={
                "detail": f"Usage limit reached. Maximum {MAX_USAGE_PER_IP} attempts allowed for this demo."
            }
        )
    
    response = await call_next(request)
    if response.status_code == 200:
        usage_tracker[client_ip] += 1
    
    return response

# ============================================
# Routes
# ============================================
@app.get("/")
async def root():
    return {
        "message": "Cognitive Decline Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Upload audio file for prediction",
            "/health": "GET - Check API health"
        }
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.wav', '.mp3', '.m4a')):
        raise HTTPException(
            status_code=400,
            detail="File must be a WAV, MP3, or M4A audio file"
        )
    
    wav_path = None
    try:
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )
        
        if len(audio_bytes) > 30 * 1024 * 1024:
            raise HTTPException(
                status_code=413,
                detail="Audio file too large. Maximum size is 30 MB."
            )
        
        wav_path = convert_to_wav(audio_bytes)
        
        # Check duration
        y, sr = librosa.load(wav_path, sr=16000)
        duration = len(y) / sr
        
        if len(y) == 0:
            raise HTTPException(
                status_code=400,
                detail="The audio contains no audible data."
            )
        
        if duration > MAX_AUDIO_DURATION:
            raise HTTPException(
                status_code=413,
                detail=f"Audio duration ({duration:.1f}s) exceeds maximum of {MAX_AUDIO_DURATION}s"
            )
        
        if duration < MIN_AUDIO_DURATION:
            raise HTTPException(
                status_code=400,
                detail=f"Audio too short ({duration:.1f}s). Please record at least {MIN_AUDIO_DURATION} seconds."
            )
        
        # ---- Transcribe using Groq ----
        if not groq_api_key:
            raise HTTPException(
                status_code=500,
                detail="GROQ_API_KEY not configured on server."
            )
        
        with open(wav_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",  # or "whisper-large-v3-turbo"
                language="en",
                response_format="text"
            )
        transcript = transcription
        
        if not transcript or not transcript.strip():
            raise HTTPException(
                status_code=400,
                detail="No speech detected. Please ensure you are speaking clearly into the microphone."
            )
        
        word_count = len(transcript.split())
        if word_count < 5:
            raise HTTPException(
                status_code=400,
                detail=f"Too little speech detected ({word_count} words). Please speak for at least 10 seconds with clear description."
            )
        
        # ---- Feature extraction and prediction ----
        features = extract_features_from_path(wav_path, transcript)
        if features is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract features from audio. Please ensure the recording contains clear speech."
            )
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        risk_score = probability[1]
        
        if prediction == 0:
            result_text = "Low Risk"
            confidence = probability[0]
        else:
            result_text = "High Risk"
            confidence = probability[1]
        
        feature_dict = {col: float(features[0, i]) for i, col in enumerate(feature_cols)}
        explanation = generate_explanation(risk_score, transcript, feature_dict)
        spider_data = compute_spider_data(feature_dict)
        
        return {
            "prediction": int(prediction),
            "result": result_text,
            "risk_score": float(risk_score),
            "confidence": float(confidence),
            "transcript": transcript,
            "duration_seconds": duration,
            "explanation": explanation,
            "spider_data": spider_data,
            "features": feature_dict
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)

# ============================================
# Run the app
# ============================================
if __name__ == "__main__":
    # Render sets the PORT environment variable, but we default to 7860
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)