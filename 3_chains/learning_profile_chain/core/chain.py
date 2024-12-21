"""
Core chain functionality for the learning profile generation pipeline.
"""

import logging
from typing import Dict, Any
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from ..config.constants import DEBUG, DEFAULT_MODEL
from ..handlers.callbacks import ChainProgressCallbackHandler
from ..models.profile import ProfileInput
from ..models.validators import ProfileValidationError, ChainOutputError, ChainOutputValidator
from ..templates.loader import load_templates

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'chain_debug.log') if DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

def combine_inputs(data: Dict[str, Any]) -> Dict[str, str]:
    """Combine profile chain output with original inputs for goals generation."""
    try:
        if not data.get("profile") or not isinstance(data["profile"], str):
            raise ChainOutputError("Invalid or empty profile output from chain")
            
        ChainOutputValidator.validate_profile_output(data["profile"])
        
        if not data.get("original_input"):
            raise ProfileValidationError("Missing original input data")
            
        original = data["original_input"]
        if not isinstance(original, dict):
            raise ProfileValidationError("Original input must be a dictionary")
            
        for field in ["coach_goal", "coach_outcomes"]:
            if not original.get(field) or not isinstance(original[field], str):
                raise ProfileValidationError(f"Missing or invalid {field} in original input")
        
        return {
            "learner_profile": data["profile"],
            "coach_goal": original["coach_goal"],
            "coach_outcomes": original["coach_outcomes"]
        }
    except Exception as e:
        raise ChainOutputError(f"Error combining inputs: {str(e)}") from e

def create_chain(model_name: str = DEFAULT_MODEL):
    """
    Create the full learning goals generation chain using LCEL.
    
    Args:
        model_name: Name of the OpenAI model to use. Defaults to DEFAULT_MODEL.
    """
    try:
        # Create handlers and model
        progress_handler = ChainProgressCallbackHandler()
        stream_handler = StreamingStdOutCallbackHandler()
        model = ChatOpenAI(
            model=model_name, 
            callbacks=[progress_handler, stream_handler],
            streaming=True
        )

        # Load templates
        profile_template, goals_template = load_templates()

        # Create the profile chain
        profile_chain = (
            profile_template 
            | model.with_config({"callbacks": [progress_handler], "tags": ["profile_generation"]}) 
            | StrOutputParser()
        ).with_config({"run_name": "Profile Generation Chain"})

        # Create the full chain
        chain_map = RunnableMap({
            "profile": profile_chain,
            "original_input": RunnablePassthrough()
        }).with_config({"run_name": "Profile Map Chain"})

        return (
            chain_map
            | combine_inputs
            | goals_template 
            | model.with_config({"callbacks": [progress_handler], "tags": ["goals_generation"]})
            | StrOutputParser()
        ).with_config({"run_name": "Learning Goals Generation Chain"})
    except Exception as e:
        raise RuntimeError(f"Failed to create chain: {str(e)}") from e

def process_profile(input_data: Dict[str, Any], model_name: str = DEFAULT_MODEL) -> str:
    """
    Process a learner profile to generate personalized learning goals.
    
    Args:
        input_data: Dictionary containing profile information
        model_name: Name of the OpenAI model to use. Defaults to DEFAULT_MODEL.
    """
    try:
        # Validate input
        validated_input = ProfileInput.from_dict(input_data)
        
        # Create and run chain
        chain = create_chain(model_name)
        print(f"\n🚀 Starting the learning profile and goals generation pipeline using {model_name}...\n")
        
        result = chain.invoke(
            input_data,
            config={"callbacks": [ChainProgressCallbackHandler()]}
        )
        
        # Validate output
        if not result or not isinstance(result, str):
            raise ChainOutputError("Chain produced invalid or empty output")
            
        ChainOutputValidator.validate_goals_output(
            result,
            input_data["coach_goal"],
            input_data["coach_outcomes"]
        )
        
        return result
        
    except ProfileValidationError as e:
        print(f"\n❌ Input Validation Error: {str(e)}")
        raise
    except ChainOutputError as e:
        print(f"\n❌ Chain Processing Error: {str(e)}")
        raise
    except Exception as e:
        print(f"\n❌ Unexpected Error: {str(e)}")
        raise RuntimeError(f"Failed to process profile: {str(e)}") from e
