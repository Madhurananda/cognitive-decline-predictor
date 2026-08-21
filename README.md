# 🧠 Multimodal Cognitive Decline Predictor

### An End-to-End AI Engineering Platform for Multimodal Speech Analysis and Cognitive Risk Assessment

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black.svg)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue.svg)
![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run-blue.svg)
![Vercel](https://img.shields.io/badge/Vercel-Deployment-black.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🌐 Live Demo

**Try the deployed application:**

👉 **https://cognitive-decline-predictor-7qco-neon.vercel.app/**

> **Note:** The public demonstration is rate-limited to manage API and cloud infrastructure costs.

---

## 📌 Project Overview

The **Multimodal Cognitive Decline Predictor** is an end-to-end AI application that analyses speech produced during a cognitive assessment task and generates an interpretable cognitive risk estimate.

The system accepts an audio recording, converts and validates the input, transcribes speech, extracts acoustic and linguistic features, applies a machine learning classifier, and generates a plain-language explanation of the result.

Rather than presenting this work as an isolated machine learning experiment, the project was designed as a **full AI engineering system**. It includes a production-style backend, real-time progress streaming, API protection, containerisation, cloud deployment, and an interactive web interface.

The complete inference pipeline is:

```text
Audio Input
    │
    ▼
Audio Validation & Conversion
    │
    ▼
Speech-to-Text
    │
    ▼
Acoustic + Linguistic Feature Extraction
    │
    ▼
Machine Learning Classification
    │
    ├── Risk Score
    ├── Binary Prediction
    └── Feature Values
    │
    ▼
LLM-Based Explanation
    │
    ▼
Real-Time Results Streaming
    │
    ▼
Interactive Web Interface
```

The project demonstrates practical experience across:

* Machine Learning
* Speech and Audio Processing
* Natural Language Processing
* Explainable AI
* LLM Integration
* Backend API Development
* Asynchronous Programming
* Real-Time Streaming
* Cloud Deployment
* Docker and Containerisation
* Frontend Integration
* Production-Oriented Error Handling

---

# ✨ Key Engineering Features

## 🎙️ Multimodal Speech Analysis

The system combines **acoustic** and **linguistic** information extracted from spoken responses.

### Acoustic features include:

* Speech and silence characteristics
* RMS energy
* Pitch statistics
* Pitch variability
* Zero-crossing rate
* Spectral centroid
* Jitter
* Shimmer

### Linguistic features include:

* Word count
* Type-Token Ratio
* Mean Length of Utterance
* Stopword ratio
* Content-word ratio
* Filler rate
* Part-of-speech diversity
* Repetition characteristics
* Pronoun usage

These features are processed and passed to a trained machine learning model to generate a continuous cognitive risk score.

---

## 🗣️ Speech-to-Text Pipeline

Audio recordings are transcribed using the **Whisper API through Groq**.

Using an external inference API reduces the computational requirements of the backend container and avoids the need to package and run a large speech recognition model within the cloud service.

Blocking transcription operations are executed using:

```python
asyncio.to_thread()
```

This allows the FastAPI application to continue handling asynchronous operations without blocking the main event loop.

---

## 🤖 Machine Learning Classification

The inference pipeline loads pre-trained machine learning artefacts:

```text
models/
├── cognitive_model.pkl
├── scaler.pkl
├── feature_columns.pkl
└── normative_stats.pkl
```

The system performs the following steps:

1. Extract features from the incoming audio and transcript.
2. Align extracted features with the training feature schema.
3. Apply the trained scaler.
4. Generate a classification probability.
5. Convert the probability into a risk score.
6. Generate a binary low-risk or high-risk prediction.
7. Return feature values for visualisation and explanation.

The architecture supports standard `scikit-learn` models such as:

* Logistic Regression
* Random Forest

Model artefacts are persisted using `joblib`.

---

## 🧠 LLM-Based Explanation

The raw machine learning prediction is converted into a user-facing explanation using an LLM.

The system integrates:

* Azure OpenAI
* Phi reasoning models
* LangChain

The LLM receives structured information derived from the prediction pipeline, including:

* Predicted risk level
* Risk score
* Selected speech characteristics
* Relevant feature deviations

The output is constrained to generate a short, accessible explanation rather than exposing raw model outputs directly to the user.

### Output sanitisation

LLM output may contain reasoning artefacts or formatting that can interfere with streaming JSON responses.

The backend therefore sanitises the generated text before sending it to the frontend:

```text
Raw LLM Output
      │
      ▼
Remove reasoning tags
      │
      ▼
Normalise whitespace
      │
      ▼
JSON-safe text
      │
      ▼
SSE response
```

This prevents malformed streaming payloads and frontend JSON parsing failures.

---

# 🏗️ System Architecture

```text
┌─────────────────────────────────────────────┐
│                 USER BROWSER                │
│                                             │
│   Audio Upload / Recording                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             NEXT.JS FRONTEND                │
│                                             │
│  • Audio upload                             │
│  • API request                              │
│  • SSE stream parser                        │
│  • Progress updates                         │
│  • Risk visualisation                       │
│  • Radar / spider chart                     │
└──────────────────────┬──────────────────────┘
                       │
                       │ POST /predict-stream
                       ▼
┌─────────────────────────────────────────────┐
│         FASTAPI BACKEND — CLOUD RUN         │
│                                             │
│  1. Validate audio                          │
│  2. Convert audio with FFmpeg               │
│  3. Check duration                          │
│  4. Speech-to-text                          │
│  5. Extract acoustic features               │
│  6. Extract linguistic features             │
│  7. Run ML classifier                       │
│  8. Generate explanation                    │
│  9. Stream progress via SSE                 │
└──────────────────────┬──────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Whisper      ML Model      Azure OpenAI
       via Groq    scikit-learn      Phi
```

---

# ⚡ Real-Time Streaming with Server-Sent Events

A standard request-response API would leave the user waiting while multiple AI operations are executed.

Instead, the backend streams progress events using **Server-Sent Events (SSE)**.

The inference process exposes progress updates such as:

```text
starting
    ↓
converting
    ↓
validating
    ↓
transcribing
    ↓
extracting_features
    ↓
predicting
    ↓
generating_explanation
    ↓
preparing_visualisation
    ↓
complete
```

Each event contains structured JSON:

```json
{
  "step": "transcribing",
  "message": "Transcribing speech...",
  "progress": 35
}
```

The frontend progressively updates the interface as events arrive.

### Engineering challenge: fragmented streaming responses

SSE payloads are not guaranteed to arrive in a single network chunk.

The frontend therefore implements a buffered parser:

```text
Incoming TCP Chunk
        │
        ▼
Append to Buffer
        │
        ▼
Search for SSE Delimiter
        │
        ▼
Extract Complete Event
        │
        ▼
JSON.parse()
        │
        ▼
Update UI
```

This prevents parsing failures caused by fragmented network packets.

---

# 🔒 API Security and Rate Limiting

The backend implements multiple layers of basic API protection.

## API Key Authentication

Requests must include:

```text
x-api-key: <API_KEY>
```

Requests without a valid API key are rejected before entering the inference pipeline.

## Rate Limiting

The public demonstration API is rate-limited on a per-IP basis.

The implementation uses:

* `slowapi`
* Redis when available
* In-memory fallback for development or lightweight deployments

This protects expensive external inference calls and prevents uncontrolled API usage.

---

# 🐳 Containerisation

The backend is packaged using Docker.

The container includes the dependencies required for:

* FastAPI
* Audio processing
* FFmpeg
* SoundFile
* spaCy
* Machine learning inference

A simplified container workflow is:

```text
Source Code
    │
    ▼
Docker Build
    │
    ▼
Container Image
    │
    ▼
Cloud Registry
    │
    ▼
Google Cloud Run
```

The application reads the deployment port from:

```text
$PORT
```

allowing the same container to run locally and within the managed Cloud Run environment.

---

# ☁️ Cloud Architecture

The application is deployed using a decoupled frontend and backend architecture.

```text
Frontend
   │
   ▼
Vercel
   │
   │ HTTPS API Request
   ▼
Google Cloud Run
   │
   ├── FastAPI
   ├── ML Inference
   ├── Speech-to-Text
   └── LLM Explanation
```

### Backend

Hosted on **Google Cloud Run**.

Benefits include:

* Container-based deployment
* Automatic scaling
* Scale-to-zero when inactive
* Managed HTTPS infrastructure
* Cloud logging
* Minimal server administration

### Frontend

Hosted independently as a **Next.js application**.

The frontend is responsible for:

* Audio upload
* API communication
* SSE event handling
* Progress display
* Prediction visualisation
* Radar chart rendering

This separation allows the frontend and AI backend to be developed and deployed independently.

---

# 📊 Interactive Results

The frontend presents multiple forms of information rather than returning only a binary prediction.

The result interface includes:

### Risk Score

A continuous probability-derived score.

### Risk Category

A simplified low-risk or high-risk classification.

### Transcript

The speech transcript generated by the transcription pipeline.

### Feature Visualisation

Speech features are compared against normative statistics derived from healthy controls.

### Radar / Spider Chart

The radar chart provides a visual representation of how selected features compare with reference values.

### AI-Generated Explanation

An LLM translates the model output into a concise, human-readable explanation.

---

# 🛠️ Technology Stack

| Area                        | Technologies               |
| --------------------------- | -------------------------- |
| **Programming Language**    | Python 3.10+               |
| **Backend**                 | FastAPI, Uvicorn           |
| **Frontend**                | Next.js, React             |
| **Styling**                 | CSS / Tailwind CSS         |
| **Audio Processing**        | Librosa, SoundFile, FFmpeg |
| **Speech Recognition**      | Whisper via Groq           |
| **NLP**                     | spaCy                      |
| **Machine Learning**        | scikit-learn               |
| **Model Persistence**       | joblib                     |
| **LLM Integration**         | Azure OpenAI               |
| **LLM Orchestration**       | LangChain                  |
| **Streaming**               | Server-Sent Events         |
| **Asynchronous Processing** | asyncio                    |
| **Visualisation**           | Chart.js                   |
| **Rate Limiting**           | slowapi                    |
| **Caching / Rate Store**    | Redis                      |
| **Containerisation**        | Docker                     |
| **Cloud Backend**           | Google Cloud Run           |
| **Frontend Deployment**     | Vercel                     |
| **Logging**                 | Google Cloud Logging       |

---

# 📂 Repository Structure

```text
cognitive-decline-predictor/
│
├── api/
│   ├── app.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── models/
│   ├── cognitive_model.pkl
│   ├── scaler.pkl
│   ├── feature_columns.pkl
│   └── normative_stats.pkl
│
├── frontend/
│   ├── app/
│   │   ├── page.js
│   │   ├── layout.js
│   │   └── globals.css
│   │
│   ├── components/
│   │   ├── Welcome.js
│   │   └── SpiderChart.js
│   │
│   ├── package.json
│   └── next.config.js
│
├── scripts/
│   ├── 01_transcribe.py
│   ├── 02_extract_features.py
│   ├── 03_train_model.py
│   └── compute_normative.py
│
├── cloudbuild.yaml
│
└── README.md
```

---

# 🚀 Local Development

## Prerequisites

Install:

* Python 3.10+
* Node.js
* npm
* FFmpeg

You will also require credentials for:

* Groq
* Azure OpenAI

---

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/cognitive-decline-predictor.git
cd cognitive-decline-predictor
```

---

## 2. Create a Python Environment

Using Conda:

```bash
conda create -n cognitive-ai python=3.10
conda activate cognitive-ai
```

Or using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r api/requirements.txt
```

Download the spaCy language model:

```bash
python -m spacy download en_core_web_sm
```

---

## 3. Configure Environment Variables

Create an environment file or configure the variables in your deployment environment.

```text
GROQ_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_DEPLOYMENT_NAME=
X_APP_API_KEY=
REDIS_URL=
```

Never commit API credentials to the repository.

---

## 4. Start the Backend

```bash
python api/app.py
```

Alternatively:

```bash
uvicorn api.app:app --reload --port 8000
```

The API should then be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## 5. Start the Frontend

Open a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The frontend should be available at:

```text
http://localhost:3050
```

---

# 📡 API Documentation

## `POST /predict-stream`

Runs the complete inference pipeline and streams progress updates.

### Headers

```text
x-api-key: <your_api_key>
```

### Request

```text
Content-Type: multipart/form-data
```

The request contains:

```text
file: audio_file
```

Supported formats include common audio formats such as:

* WAV
* MP3
* M4A

### Response

The endpoint returns a Server-Sent Events stream.

Example progress event:

```json
{
  "step": "extracting_features",
  "message": "Extracting acoustic and linguistic features...",
  "progress": 60
}
```

Example final result:

```json
{
  "prediction": 1,
  "result": "high_risk",
  "risk_score": 0.82,
  "confidence": 0.82,
  "transcript": "...",
  "duration_seconds": 42.1,
  "explanation": "...",
  "spider_data": {},
  "features": {}
}
```

---

## `GET /health`

Health endpoint used for service monitoring.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

# 🔬 Model Training Pipeline

The repository also contains scripts for reproducing the machine learning pipeline.

The workflow is:

```text
Raw Audio
    │
    ▼
Speech Transcription
    │
    ▼
Feature Extraction
    │
    ├── Acoustic Features
    └── Linguistic Features
    │
    ▼
Feature Dataset
    │
    ▼
Model Training
    │
    ├── Logistic Regression
    └── Random Forest
    │
    ▼
Model Selection
    │
    ▼
Saved Model Artefacts
```

Training scripts:

```bash
python scripts/01_transcribe.py
python scripts/02_extract_features.py
python scripts/03_train_model.py
python scripts/compute_normative.py
```

The training process produces:

* Trained classifier
* Feature scaler
* Feature schema
* Normative statistics for visualisation

---

# ⚙️ Engineering Optimisations

Several implementation decisions were made to improve responsiveness and deployment efficiency.

## Fast Audio Duration Validation

Instead of loading the complete waveform with:

```python
librosa.load()
```

the application uses metadata inspection through:

```python
soundfile.info()
```

This avoids unnecessarily decoding the entire audio file when only the duration is required.

---

## Asynchronous Execution of Blocking Operations

External API calls and blocking inference operations are executed using:

```python
asyncio.to_thread()
```

This prevents expensive synchronous operations from blocking the FastAPI event loop.

---

## Buffered SSE Parsing

Streaming responses may arrive in fragmented network chunks.

The frontend maintains a buffer and processes only complete SSE events.

This avoids failures caused by attempting to parse incomplete JSON.

---

## LLM Output Sanitisation

Generated LLM text is cleaned before being embedded in SSE responses.

The sanitisation process:

1. Removes unwanted reasoning artefacts.
2. Normalises whitespace.
3. Produces JSON-safe output.

This addresses issues where malformed generated text could interrupt the streaming response.

---

## Reduced Backend Resource Requirements

Speech recognition is performed through an external API rather than hosting Whisper inside the container.

This reduces:

* Container image size
* Memory requirements
* Cold-start overhead
* GPU infrastructure requirements

---

# 🧪 Testing and Validation

The project includes validation at multiple levels:

### Input Validation

* Audio format handling
* Duration checks
* Invalid file detection

### Backend Validation

* Model loading checks
* API authentication
* Error handling
* Structured JSON responses

### Streaming Validation

* Incremental progress events
* Fragmented response handling
* Buffered SSE parsing

### Deployment Validation

* Container startup
* Environment variable configuration
* Cloud Run health checks
* Frontend-to-backend communication

---

# ⚠️ Limitations and Responsible Use

This project is a **technical demonstration and research-oriented AI engineering system**.

It is **not a medical device** and must not be used for clinical diagnosis or treatment decisions.

The generated risk estimate:

* Does not constitute a clinical diagnosis.
* Should not be interpreted without appropriate clinical context.
* May be affected by recording quality, demographic variation, speech characteristics, language, and dataset limitations.
* Requires further validation before any real-world clinical deployment.

The primary purpose of this repository is to demonstrate the design and implementation of an end-to-end multimodal AI application.

---

# 🔮 Future Development

Potential future improvements include:

### Multilingual Support

Extend transcription, linguistic analysis, and explanation generation to additional languages.

### Additional Modalities

Integrate:

* Facial video
* Longitudinal speech
* Clinical metadata
* Electronic health record information

### Improved Modelling

Explore:

* Transformer-based speech representations
* Self-supervised speech models
* Multimodal fusion
* Personalised baseline modelling

### Production Monitoring

Add:

* Structured metrics
* Model monitoring
* API analytics
* Drift detection
* Observability dashboards

### User Accounts

Enable:

* Secure authentication
* Persistent result storage
* Longitudinal tracking

---

# 💡 What This Project Demonstrates

This repository was developed as an end-to-end AI engineering project rather than only a machine learning model.

It demonstrates the ability to:

* Build a complete machine learning inference pipeline.
* Process real-world audio input.
* Combine speech, audio, NLP, and machine learning techniques.
* Integrate external AI services.
* Build asynchronous backend APIs.
* Implement real-time streaming communication.
* Design frontend and backend services as independent components.
* Containerise AI applications with Docker.
* Deploy services to managed cloud infrastructure.
* Handle API security and rate limiting.
* Debug practical issues involving streaming, JSON serialisation, asynchronous execution, and cloud deployment.

---

# 👨‍💻 About the Developer

**Dr. Madhu Pahar**

Research Fellow and AI researcher specialising in:

* Machine Learning
* Speech and Audio Processing
* Multimodal AI
* Natural Language Processing
* Explainable AI
* Healthcare AI

This project was developed to demonstrate the transition from research-focused machine learning workflows to **end-to-end AI engineering and production-oriented system development**.

The work combines experience in speech-based clinical AI with practical engineering across backend development, cloud infrastructure, LLM integration, and interactive application development.

---

# 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

# 🙏 Acknowledgements

This project builds upon widely used open-source and cloud technologies, including:

* FastAPI
* Next.js
* scikit-learn
* spaCy
* Librosa
* FFmpeg
* Docker
* Google Cloud Run
* Vercel
* Groq
* Azure OpenAI

---

## ⭐ If You Found This Project Interesting

If you found this repository useful or interesting, consider giving it a ⭐.

---

**Built by Dr. Madhu Pahar**

[Google Scholar](https://scholar.google.co.uk/citations?user=P2clDtkAAAAJ&hl=en) · [LinkedIn](https://linkedin.com/in/madhurananda) · [Portfolio](https://madhurananda.github.io/)
