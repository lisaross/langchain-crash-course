"""
Configuration constants for the learning profile chain.
"""

import os

# Debug Configuration
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Model Configuration
MODEL_CONFIGS = {
    # O1 Model (Preview)
    'gpt-o1': {
        'max_tokens': 128000,
        'name': 'gpt-o1',
        'description': 'Most capable model optimized for reasoning and coding tasks'
    },
    
    # O4 Models (Production)
    'gpt-4o': {
        'max_tokens': 128000,
        'name': 'gpt-4o',
        'description': 'High-intelligence flagship model for complex, multi-step tasks'
    },
    'gpt-4o-mini': {
        'max_tokens': 8192,
        'name': 'gpt-4o-mini',
        'description': 'Cost-efficient small model with vision capabilities, smarter than GPT-3.5'
    },
    'gpt-4o-mini-audio-preview': {
        'max_tokens': 8192,
        'name': 'gpt-4o-mini-audio-preview',
        'description': 'GPT-4o mini with audio processing capabilities'
    },
    
    # GPT-4 Models
    'gpt-4': {
        'max_tokens': 8192,
        'name': 'gpt-4',
        'description': 'Standard GPT-4 model for general use'
    },
    'gpt-4-32k': {
        'max_tokens': 32768,
        'name': 'gpt-4-32k',
        'description': 'Extended context GPT-4 model for longer tasks'
    }
}

# Set default model from environment variable or fallback to gpt-4o-mini
DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-4o-mini')

# Validate default model
if DEFAULT_MODEL not in MODEL_CONFIGS:
    print(f"Warning: Invalid default model '{DEFAULT_MODEL}'. Falling back to 'gpt-4o-mini'")
    DEFAULT_MODEL = 'gpt-4o-mini'

TOKEN_WARNING_THRESHOLD = 0.8

# Sensitive Data Keys
SENSITIVE_KEYS = {'api_key', 'secret', 'password', 'token', 'authorization'}

# Validation Constants
MIN_OUTPUT_LENGTH = 100
PROFILE_REQUIRED_CONCEPTS = {
    'jtbd': ['jtbd', 'jobs to be done', 'job to be done', 'needs to accomplish'],
    'challenges': ['challenges', 'obstacles', 'difficulties', 'pain points'],
    'skills': ['skills gap', 'skill gaps', 'missing skills', 'needs to learn', 'areas for improvement']
}

# File Paths
TEMPLATE_FILE = "templates.yaml"
