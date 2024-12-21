"""
Learning Profile Chain - A modular LangChain pipeline for generating personalized learning goals.
"""

from dotenv import load_dotenv
from .core.chain import process_profile

# Load environment variables
load_dotenv()

__version__ = "0.1.0"
__all__ = ["process_profile"]
