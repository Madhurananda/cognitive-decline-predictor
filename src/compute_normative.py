#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 16:30:00 2026
@author: madhupahar
"""

import pandas as pd
import numpy as np
import joblib
import os

# Load your feature data
df = pd.read_csv("data/processed/features.csv")
print(f"Total samples: {len(df)}")

# Filter only healthy controls (HC)
healthy_df = df[df['diagnosis'] == 'HC']
print(f"Healthy controls: {len(healthy_df)}")

# Define the features to show in the spider diagram
# These should be the most clinically meaningful ones
SPIDER_FEATURES = [
    'silence_ratio',    # Lower is better (fewer pauses)
    'ttr',              # Higher is better (more vocabulary diversity)
    'content_ratio',    # Higher is better (more meaningful words)
    'mlu',              # Higher is better (longer utterances)
    'word_count',       # Higher is better (more verbal output)
    'pitch_std'         # Moderate is good (natural variation)
]

# Compute normative stats from healthy controls
normative_stats = {}

for feature in SPIDER_FEATURES:
    values = healthy_df[feature].values
    
    # Remove outliers (optional but recommended)
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    clean_values = values[(values >= lower_bound) & (values <= upper_bound)]
    
    if len(clean_values) > 0:
        values = clean_values
    
    normative_stats[feature] = {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'p25': float(np.percentile(values, 25)),
        'p50': float(np.percentile(values, 50)),
        'p75': float(np.percentile(values, 75)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }

# Save normative data
os.makedirs("models", exist_ok=True)
joblib.dump(normative_stats, "models/normative_stats.pkl")
print("✅ Normative stats saved to models/normative_stats.pkl")

# Display
print("\n📊 Normative Data (Healthy Controls):")
print("=" * 50)
for feature, stats in normative_stats.items():
    print(f"\n{feature}:")
    print(f"  Mean: {stats['mean']:.3f}")
    print(f"  Std:  {stats['std']:.3f}")
    print(f"  p25:  {stats['p25']:.3f}")
    print(f"  p50:  {stats['p50']:.3f}")
    print(f"  p75:  {stats['p75']:.3f}")
    print(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
