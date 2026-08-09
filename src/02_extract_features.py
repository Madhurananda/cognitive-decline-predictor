#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:38:20 2026

@author: madhupahar
"""



import os
import numpy as np
import pandas as pd
import librosa
import spacy
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Paths
RAW_DIR = "data/raw"
META_PATH = os.path.join(RAW_DIR, "meta_with_transcripts.csv")

# Load data with transcripts
df = pd.read_csv(META_PATH)
print(f"Total subjects: {len(df)}")

# Load spaCy model for linguistic features
print("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")
print("✅ spaCy loaded!")

# ------------------------------------------------------------
# Acoustic Features (using librosa)
# ------------------------------------------------------------
def extract_acoustic_features(audio_path):
    """
    Extract acoustic features from an audio file.
    Returns a dictionary of features.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000, duration=60)
        duration = len(y) / sr
        
        # If audio is too short, skip
        if duration < 5.0:
            return None
        
        # RMS Energy (loudness)
        rms = librosa.feature.rms(y=y).mean()
        
        # Pitch (F0) using piptrack
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            pitch_mean = np.mean(pitch_values)
            pitch_std = np.std(pitch_values)
        else:
            pitch_mean = 0
            pitch_std = 0
        
        # Silence / Pauses using librosa.split (energy-based VAD)
        intervals = librosa.effects.split(y, top_db=30)
        silence_duration = sum([end - start for start, end in intervals]) / sr
        silence_ratio = silence_duration / duration if duration > 0 else 0
        speech_time = duration - silence_duration
        
        # Zero-crossing rate (roughness)
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()
        
        # Jitter (pitch perturbation) and Shimmer (amplitude perturbation)
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
        print(f"Error extracting acoustic features: {e}")
        return None

# ------------------------------------------------------------
# Linguistic Features (using spaCy)
# ------------------------------------------------------------
def extract_linguistic_features(text):
    """
    Extract linguistic features from a transcript.
    Returns a dictionary of features.
    """
    if not text or pd.isna(text) or text.strip() == "":
        return None
    
    doc = nlp(text)
    
    # Extract words (excluding punctuation and spaces)
    words = [t.text for t in doc if not t.is_punct and not t.is_space]
    
    if len(words) == 0:
        return None
    
    word_count = len(words)
    
    # Type-Token Ratio (vocabulary richness)
    unique_words = len(set([w.lower() for w in words]))
    ttr = unique_words / word_count if word_count > 0 else 0
    
    # Average word length
    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    
    # Stopword ratio
    stopwords = [t for t in doc if t.is_stop]
    stop_ratio = len(stopwords) / len(doc) if len(doc) > 0 else 0
    
    # Content word ratio (Nouns, Verbs, Adjectives, Adverbs)
    content_pos = {'NOUN', 'VERB', 'ADJ', 'ADV'}
    content_words = [t for t in doc if t.pos_ in content_pos]
    content_ratio = len(content_words) / word_count if word_count > 0 else 0
    
    # Filler words (um, uh, like, er, ah, etc.)
    fillers = ['um', 'uh', 'like', 'er', 'ah', 'you know', 'i mean']
    filler_count = sum([1 for t in doc if t.text.lower() in fillers])
    filler_rate = filler_count / word_count if word_count > 0 else 0
    
    # POS diversity
    pos_tags = [t.pos_ for t in doc]
    pos_diversity = len(set(pos_tags)) / len(pos_tags) if len(pos_tags) > 0 else 0
    
    # Mean Length of Utterance (MLU) – split by sentences
    sentences = [sent for sent in doc.sents]
    if sentences:
        mlu = np.mean([len([t for t in sent if not t.is_punct]) for sent in sentences])
    else:
        mlu = word_count
    
    # Repetition rate (consecutive repeated words)
    repeats = sum(1 for i in range(1, len(words)) if words[i].lower() == words[i-1].lower())
    repetition_rate = repeats / word_count if word_count > 0 else 0
    
    # Pronoun ratio
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

# ------------------------------------------------------------
# Main Loop: Process All Subjects
# ------------------------------------------------------------
print("\nStarting feature extraction...")
all_features = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    subj = row['IDs']
    diagnosis = row['diagnosis']
    split = row['Split']
    
    # Map labels: 1 = MCI or Dementia, 0 = HC
    if diagnosis in ['MCI', 'Dementia']:
        label = 1
    else:
        label = 0
    
    # Audio path
    audio_path = os.path.join(RAW_DIR, subj, f"{subj}__CTD.wav")
    if not os.path.exists(audio_path):
        continue
    
    # Extract acoustic features
    acoustic = extract_acoustic_features(audio_path)
    if acoustic is None:
        continue
    
    # Extract linguistic features
    transcript = row.get('transcript', '')
    linguistic = extract_linguistic_features(transcript)
    if linguistic is None:
        continue
    
    # Combine all features
    features = {
        'subject_id': subj,
        'diagnosis': diagnosis,
        'label': label,
        'split': split,
        **acoustic,
        **linguistic
    }
    all_features.append(features)

# Create DataFrame
feature_df = pd.DataFrame(all_features)

# Save to CSV
os.makedirs("data/processed", exist_ok=True)
feature_df.to_csv("data/processed/features.csv", index=False)

print(f"\n✅ Extracted features for {len(feature_df)} subjects")
print(f"Feature columns: {list(feature_df.columns)}")
print(f"Class distribution:")
print(feature_df['label'].value_counts())

# Show sample features
print("\n📊 Sample features:")
print(feature_df.head(3).to_string())