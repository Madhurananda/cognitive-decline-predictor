# 🧠 Multimodal Cognitive Decline Predictor

### An End-to-End AI Engineering Platform for Multimodal Speech Analysis and Cognitive Risk Assessment

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green.svg)
![Next.js](https://img.shields.io/badge/Next.js-Frontend-black.svg)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue.svg)
![Google Cloud Run](https://img.shields.io/badge/Google%20Cloud-Cloud%20Run-blue.svg)
![Vercel](https://img.shields.io/badge/Vercel-Deployment-black.svg)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

---

## 🌐 Live Demo

**Try the deployed application:**

👉 **https://cognitive-decline-predictor-7qco-neon.vercel.app/**

> **Note:** The public demonstration is rate-limited to manage API and cloud infrastructure costs.

---

### 📸 Screenshots & Walkthrough

<br>

| | |
|:---:|:---:|
| **1. Welcome & Terms**<br>The user is presented with a brief introduction to the project, research context, and usage guidelines. Acceptance is required before proceeding. | <img width="800" alt="Welcome page" src="https://github.com/user-attachments/assets/60dd85e9-0ef8-461b-b6fe-093de4249675" /> |
| **2. Recording Interface**<br>The user is prompted to describe the *Cookie Theft* picture. Audio must be between 10 seconds and 3 minutes in length. A live waveform visualises the input. | <img width="800" alt="Recording interface" src="https://github.com/user-attachments/assets/b397f504-7b22-471f-98a0-48a2deefbc13" /> |
| **3. Progress Streaming**<br>Once the audio is submitted, real‑time progress updates (conversion, transcription, feature extraction, prediction, explanation) are streamed via Server‑Sent Events. | <img width="800" alt="Progress stream" src="https://github.com/user-attachments/assets/d82c56c8-eea8-4244-bddf-e7070d75d0ad" /> |
| **4. Risk Assessment & Explanation**<br>The classifier returns a risk score (low/high) and a confidence level. The transcript and an AI‑generated empathetic explanation are displayed. | <img width="800" alt="Risk and explanation" src="https://github.com/user-attachments/assets/837a5d40-cf94-44c5-aa1b-70b8b528a19e" /> |
| **5. Spider Diagram – Speech Biomarkers**<br>An interactive radar chart compares the user’s speech features against a healthy reference population, offering interpretable visual feedback. | <img width="800" alt="Spider diagram" src="https://github.com/user-attachments/assets/26dd6e19-e771-4151-9ba1-feae2c2c5781" /> |
| **6. Detailed Feature View**<br>For transparency, the raw feature values used by the model and LLM can be expanded and inspected. | <img width="800" alt="Detailed features" src="https://github.com/user-attachments/assets/2223ae4e-4236-466c-9571-545dfb97e584" /> |

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

## 🔬 From Research Data to Production Model

One of the central objectives of this project was to demonstrate the complete journey from **clinical research data to a deployable AI application**.

The model used by the application was not developed from a generic demonstration dataset. It was built using speech data originating from the **CognoSpeak research programme**, a real-world study investigating the automatic and remote assessment of early cognitive decline from conversational speech.

The development pathway can be summarised as:

```text
CognoSpeak Research Programme
        │
        ▼
Real-World Clinical Speech Data Collection
        │
        ▼
PROCESS-2 Benchmark Dataset
        │
        ▼
Data Engineering & Quality Control
        │
        ▼
Acoustic + Linguistic Feature Engineering
        │
        ▼
Machine Learning Model Development
        │
        ├── Logistic Regression
        └── Random Forest
        │
        ▼
Validation & Model Selection
        │
        ▼
Predefined Held-Out Test Evaluation
        │
        ▼
Feature Importance & Performance Investigation
        │
        ▼
Saved Model Artefacts
        │
        ▼
FastAPI Inference Service
        │
        ▼
Real-Time Web Application
        │
        ▼
Cloud Deployment
```

This section describes how the underlying research data was transformed into the machine learning model used by the deployed application.

---

### 1. Data Collection and Research Background

The speech data originates from the **CognoSpeak** research programme, which investigates the use of automatically collected conversational speech for the assessment of early cognitive decline.

CognoSpeak was designed around the collection of real-world speech data from cognitive assessment interactions, providing a foundation for investigating acoustic, linguistic, and multimodal markers associated with cognitive impairment.

The data collection methodology and CognoSpeak system are described in:

> **Pahar, Madhurananda, et al.** "CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech." *2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM).* IEEE, 2025.

📄 **Paper:**
https://ieeexplore.ieee.org/abstract/document/10969487

Further work investigating the automatic detection of early cognitive decline using multimodal feature fusion and transfer learning is described in:

> **Pahar, Madhurananda, et al.** "Automatic detection of early cognitive decline using multimodal feature fusion and transfer learning on real-world conversational speech." *IEEE Journal of Biomedical and Health Informatics* 29.12 (2025): 8727-8734.

📄 **Paper:**
https://ieeexplore.ieee.org/abstract/document/11284700

These studies provide the research context for the speech data and modelling approaches underlying this engineering project.

---

### 2. PROCESS-2 Benchmark Dataset

A subset of the CognoSpeak data was subsequently released as **PROCESS-2**, a benchmark speech corpus designed to support reproducible research into early cognitive impairment detection.

The PROCESS-2 dataset is described in:

> **Pahar, Madhurananda, et al.** "PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection." *arXiv preprint arXiv:2605.14888* (2026).

📄 **Paper:**
https://arxiv.org/abs/2605.14888

For this project, the available dataset consisted of:

| Group                | Number of Participants |
| -------------------- | ---------------------: |
| Healthy Controls     |                    200 |
| Cognitively Impaired |                    200 |
| **Total**            |                **400** |

The classification task was formulated as a binary prediction problem:

```text
0 → Healthy Control
1 → Cognitive Impairment
```

The dataset was balanced across the two classes.

---

### 3. Data Engineering and Feature Preparation

The raw speech data was transformed into a structured machine-learning dataset through a data engineering and feature extraction pipeline.

The process involved:

```text
Raw Audio
    │
    ▼
Audio Validation
    │
    ▼
Speech Transcription
    │
    ├── Transcript
    │
    ▼
Acoustic Feature Extraction
    │
    ├── Silence characteristics
    ├── Energy
    ├── Pitch statistics
    ├── Jitter
    └── Shimmer
    │
    ▼
Linguistic Feature Extraction
    │
    ├── Lexical diversity
    ├── Content-word ratio
    ├── Stopword ratio
    ├── Mean Length of Utterance
    └── Other linguistic statistics
    │
    ▼
Feature Alignment & Cleaning
    │
    ▼
Machine Learning Feature Matrix
```

The final modelling dataset contained **21 handcrafted acoustic and linguistic features**.

This deliberately compact representation was chosen to create a lightweight and inspectable baseline model rather than relying on a large end-to-end deep learning architecture.

The approach demonstrates the complete transformation of speech data into a structured representation suitable for machine learning:

* audio processing
* speech transcription
* acoustic feature extraction
* linguistic feature extraction
* feature alignment
* data cleaning
* feature scaling
* reproducible model artefact generation

---

### 4. Normative Feature Modelling

In addition to training the classification model, normative statistics were calculated from the **200 healthy control participants**.

These statistics were saved to:

```text
models/normative_stats.pkl
```

The normative reference distributions are used by the deployed application to contextualise an individual's extracted speech characteristics and support the radar/spider-chart visualisation.

Examples include:

| Feature                  |     Mean | Standard Deviation |            Range |
| ------------------------ | -------: | -----------------: | ---------------: |
| Silence Ratio            |    0.739 |              0.093 |      0.492–0.953 |
| Type-Token Ratio         |    0.574 |              0.091 |      0.372–0.810 |
| Content Ratio            |    0.430 |              0.032 |      0.351–0.517 |
| Mean Length of Utterance |   16.757 |              4.926 |     6.417–31.000 |
| Word Count               |  160.626 |             77.172 |           31–376 |
| Pitch Standard Deviation | 1142.445 |             69.457 | 978.994–1332.233 |

These values are not used as clinical thresholds. Instead, they provide a reference distribution derived from the healthy control group for visualising how extracted speech features relate to the normative dataset.

---

### 5. Model Development Strategy

The 400-participant dataset was organised into training, validation, and held-out test partitions:

| Split      | Samples | Percentage |
| ---------- | ------: | ---------: |
| Training   |     256 |        64% |
| Validation |      64 |        16% |
| Test       |      80 |        20% |
| **Total**  | **400** |   **100%** |

Two deliberately simple and interpretable machine learning approaches were evaluated:

#### Logistic Regression

A linear baseline providing a transparent reference model.

#### Random Forest

A non-linear ensemble model capable of learning interactions between acoustic and linguistic features.

The modelling pipeline consisted of:

```text
Feature Matrix
      │
      ▼
Training Partition
      │
      ├── Feature Scaling
      │
      ├── Logistic Regression
      │
      └── Random Forest
                │
                ▼
        Validation Evaluation
                │
                ▼
        Model Performance Review
                │
                ▼
     Predefined Held-Out Test Set
```

The feature scaler, feature column order, and final trained model were saved as reusable inference artefacts:

```text
models/
├── cognitive_model.pkl
├── scaler.pkl
├── feature_columns.pkl
└── normative_stats.pkl
```

---

### 6. Model Performance

Both models were evaluated using:

* Accuracy
* Binary F1 Score
* Macro F1 Score
* Area Under the ROC Curve (AUC-ROC)
* Confusion Matrices

#### Performance Summary

| Model               | Split      |  Accuracy | Binary F1 |  Macro F1 |   AUC-ROC |
| ------------------- | ---------- | --------: | --------: | --------: | --------: |
| Logistic Regression | Training   |     0.715 |     0.704 |     0.714 |     0.764 |
| Logistic Regression | Validation |     0.688 |     0.677 |     0.687 |     0.735 |
| Logistic Regression | **Test**   | **0.738** | **0.747** | **0.737** | **0.804** |
| Random Forest       | Training   |     1.000 |     1.000 |     1.000 |     1.000 |
| Random Forest       | Validation |     0.703 |     0.678 |     0.701 |     0.752 |
| Random Forest       | **Test**   | **0.750** | **0.767** | **0.749** | **0.806** |

The Random Forest achieved the strongest held-out test performance and was selected as the model used by the deployed application.

#### Final Held-Out Test Performance

```text
Accuracy:        0.750
Binary F1 Score: 0.767
Macro F1 Score:  0.749
AUC-ROC:         0.806
```

The test results were obtained using the held-out evaluation partition rather than the training data.

---

### 7. Performance Investigation

The performance investigation deliberately considered results across the training, validation, and test partitions rather than reporting only the strongest metric.

The Random Forest achieved:

```text
Training F1:   1.000
Validation F1: 0.678
Test F1:       0.767
```

The perfect training performance, combined with substantially lower validation and test performance, indicates that the model can fit the training data extremely closely and highlights the importance of evaluation on unseen data.

This was one reason for retaining the held-out test evaluation as an important part of the development workflow.

Rather than interpreting training accuracy as evidence of real-world performance, the deployed model was selected based on its performance on unseen data.

---

### 8. Held-Out Test Confusion Matrix

The final Random Forest model produced the following results on the held-out test set:

```text
                    Predicted

                  Healthy   Impaired
Actual Healthy       27         13

Actual Impaired       7         33
```

This corresponds to:

| Metric                              | Value |
| ----------------------------------- | ----: |
| Sensitivity / Recall for Impairment | 82.5% |
| Specificity for Healthy Controls    | 67.5% |
| Precision for Impairment            | 71.7% |

The model showed higher sensitivity than specificity on the held-out test partition, indicating that impaired participants were identified more reliably than healthy participants.

The resulting false-positive and false-negative trade-off would need to be considered carefully in any future clinical application.

This repository is a technical and research-oriented demonstration and does not represent a clinically validated diagnostic system.

---

### 9. Feature Importance and Model Interpretation

One advantage of the Random Forest approach is that the contribution of individual features can be inspected.

The ten most important features were:

| Rank | Feature         | Importance |
| ---: | --------------- | ---------: |
|    1 | `silence_ratio` |     0.0838 |
|    2 | `content_ratio` |     0.0802 |
|    3 | `rms`           |     0.0727 |
|    4 | `shimmer`       |     0.0654 |
|    5 | `speech_time`   |     0.0612 |
|    6 | `stop_ratio`    |     0.0587 |
|    7 | `avg_word_len`  |     0.0583 |
|    8 | `mlu`           |     0.0563 |
|    9 | `pronoun_ratio` |     0.0499 |
|   10 | `ttr`           |     0.0490 |

Several of the highest-ranked features are consistent with speech and language characteristics that have been investigated in previous research on cognitive impairment.

These include:

* **Silence ratio** — reflecting pausing and speech continuity.
* **Content ratio** — reflecting the proportion of semantically informative words.
* **RMS energy** — capturing aspects of vocal intensity.
* **Shimmer** — reflecting cycle-to-cycle variability in vocal amplitude.
* **Speech time** — representing the amount of verbal output.
* **Lexical and syntactic features** — including lexical diversity and mean length of utterance.

Feature importance values should be interpreted as model-specific indicators of contribution rather than causal explanations. They nevertheless provide an additional level of transparency compared with a purely black-box inference pipeline.

The complete feature importance analysis and confusion matrix were also saved as model artefacts:

```text
models/
├── cognitive_model.pkl
├── scaler.pkl
├── feature_columns.pkl
├── normative_stats.pkl
└── confusion_matrix.png
```

---

### 10. Relationship to Previous Research

The modelling approach used in this application builds upon a broader body of research investigating speech as a source of potential biomarkers for cognitive decline.

The CognoSpeak and multimodal feature-fusion studies provide the research foundation for the data collection and modelling context:

> **Pahar, Madhurananda, et al.** "CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech." *2025 IEEE Symposium on Computational Intelligence in Health and Medicine (CIHM).* IEEE, 2025.

https://ieeexplore.ieee.org/abstract/document/10969487

> **Pahar, Madhurananda, et al.** "Automatic detection of early cognitive decline using multimodal feature fusion and transfer learning on real-world conversational speech." *IEEE Journal of Biomedical and Health Informatics* 29.12 (2025): 8727-8734.

https://ieeexplore.ieee.org/abstract/document/11284700

The PROCESS-2 benchmark provides the dataset and predefined evaluation framework used for the current implementation:

> **Pahar, Madhurananda, et al.** "PROCESS-2: A Benchmark Speech Corpus for Early Cognitive Impairment Detection." *arXiv preprint arXiv:2605.14888* (2026).

https://arxiv.org/abs/2605.14888

The results obtained by the compact 21-feature baseline are broadly consistent with the range of performance reported in comparable speech-based cognitive impairment studies described in this research literature.

Importantly, the objective of this repository was not to establish a new state-of-the-art clinical model. Instead, the focus was on demonstrating the complete engineering process required to transform research data and machine learning models into a reproducible, inspectable, and deployable AI system.

---

### 11. From Model to Deployed Application

After model development and evaluation, the selected Random Forest model was integrated into the application backend.

The saved artefacts are loaded when the FastAPI service starts:

```text
Application Startup
        │
        ▼
Load Model
        │
        ▼
Load Feature Scaler
        │
        ▼
Load Feature Schema
        │
        ▼
Load Normative Statistics
        │
        ▼
API Ready
```

When a user submits an audio recording, the backend reproduces the feature engineering pipeline and ensures that the generated features match the model's expected feature schema.

The resulting workflow is:

```text
New User Audio
      │
      ▼
Audio Processing
      │
      ▼
Speech Transcription
      │
      ▼
21 Feature Extraction
      │
      ▼
Feature Alignment
      │
      ▼
Scaler
      │
      ▼
Random Forest Model
      │
      ├── Prediction
      ├── Risk Score
      └── Feature Values
              │
              ▼
      Normative Comparison
              │
              ▼
      LLM Explanation
              │
              ▼
      Real-Time Frontend
```

This closes the loop between **research data, machine learning development, model evaluation, and production-oriented deployment**.

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
