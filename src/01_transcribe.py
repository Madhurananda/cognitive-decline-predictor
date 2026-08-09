#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:44:20 2026

@author: madhupahar
"""

# import os
# import whisper
# import pandas as pd
# from tqdm import tqdm

# # Paths
# RAW_DIR = "data/raw"
# META_PATH = os.path.join(RAW_DIR, "meta-info.csv")

# # Load metadata
# df = pd.read_csv(META_PATH)

# # Load Whisper Tiny model (use "base" or "small" if you prefer)
# # model = whisper.load_model("tiny")

# model = whisper.load_model("tiny")


# def transcribe_ctd(subject_id):
#     """Transcribe the CTD.wav file for a subject."""
#     audio_path = os.path.join(RAW_DIR, subject_id, f"{subject_id}__CTD.wav")
#     if not os.path.exists(audio_path):
#         return None
#     try:
#         result = model.transcribe(audio_path)
#         return result["text"]
#     except Exception as e:
#         print(f"Error transcribing {subject_id}: {e}")
#         return None

# # Process each subject
# transcripts = []
# for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing CTD"):
#     subj = row['IDs']
#     transcript = transcribe_ctd(subj)
#     transcripts.append(transcript)

# df['transcript'] = transcripts

# # Save the metadata with transcripts
# df.to_csv(os.path.join(RAW_DIR, "meta_with_transcripts.csv"), index=False)
# print("✅ Transcripts saved to data/raw/meta_with_transcripts.csv")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 12:44:20 2026
@author: madhupahar
"""

import os
import whisper
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Paths
RAW_DIR = "data/raw"
META_PATH = os.path.join(RAW_DIR, "meta-info.csv")

# Load metadata
df = pd.read_csv(META_PATH)
print(f"Total subjects to transcribe: {len(df)}")

# Load Whisper model (you can change "medium" to "tiny", "base", or "small")
print("Loading Whisper model...")
model = whisper.load_model("medium")
print("Model loaded successfully!")

def transcribe_ctd(subject_id):
    """Transcribe the CTD.wav file for a subject."""
    audio_path = os.path.join(RAW_DIR, subject_id, f"{subject_id}__CTD.wav")
    
    if not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        return None
    
    try:
        # Force language to English for better accuracy
        result = model.transcribe(
            audio_path,
            language="en",           # Force English
            task="transcribe",       # Transcribe (not translate)
            fp16=False               # Use float32 for CPU stability
        )
        return result["text"]
    except Exception as e:
        print(f"❌ Error transcribing {subject_id}: {e}")
        return None

# Process each subject
print("\nStarting transcription...")
transcripts = []

for _, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing CTD"):
    subj = row['IDs']
    transcript = transcribe_ctd(subj)
    transcripts.append(transcript)

# Add transcripts to dataframe
df['transcript'] = transcripts

# Count successful transcriptions
successful = df['transcript'].notna().sum()
print(f"\n✅ Successfully transcribed {successful} out of {len(df)} subjects")

# Save the metadata with transcripts
df.to_csv(os.path.join(RAW_DIR, "meta_with_transcripts.csv"), index=False)
print("✅ Transcripts saved to data/raw/meta_with_transcripts.csv")

# Show sample transcripts
print("\n📝 Sample transcripts:")
sample_df = df[df['transcript'].notna()].head(3)
for _, row in sample_df.iterrows():
    transcript_preview = row['transcript'][:150] if row['transcript'] else "None"
    print(f"\n{row['IDs']}: {transcript_preview}...")


