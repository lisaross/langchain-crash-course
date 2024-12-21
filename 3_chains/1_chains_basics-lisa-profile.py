from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import yaml
import os
from pathlib import Path

# Custom Exceptions
class ProfileValidationError(Exception):
    """Raised when profile data validation fails"""
    pass

class ChainOutputError(Exception):
    """Raised when chain output is invalid or empty"""
    pass

class TemplateLoadError(Exception):
    """Raised when template loading fails"""
    pass

@dataclass
class ProfileInput:
    """Data class for profile input validation"""
    background: str
    current_role: str
    experience_level: str
    coach_goal: str
    coach_outcomes: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileInput':
        """Validate and create ProfileInput from dictionary"""
        required_fields = ['background', 'current_role', 'experience_level', 'coach_goal', 'coach_outcomes']
        
        # Check for missing fields
        missing_fields = [field for field in required_fields if field not in data or not data[field]]
        if missing_fields:
            raise ProfileValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate string types
        for field in required_fields:
            if not isinstance(data[field], str):
                raise ProfileValidationError(f"Field '{field}' must be a string")
            
            # Check for minimum content length
            if len(data[field].strip()) < 3:
                raise ProfileValidationError(f"Field '{field}' must contain meaningful content (min 3 characters)")
        
        return cls(
            background=data['background'],
            current_role=data['current_role'],
            experience_level=data['experience_level'],
            coach_goal=data['coach_goal'],
            coach_outcomes=data['coach_outcomes']
        )

def load_templates(template_path: str = "templates.yaml") -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """
    Load templates from YAML file.
    
    Args:
        template_path: Path to the YAML template file
        
    Returns:
        Tuple of (profile_template, goals_template)
        
    Raises:
        TemplateLoadError: If template file is missing or malformed
    """
    try:
        # Resolve template path relative to the script location
        script_dir = Path(__file__).parent
        full_path = script_dir / template_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Template file not found: {full_path}")
            
        with open(full_path, 'r') as file:
            templates = yaml.safe_load(file)
            
        # Validate template structure
        required_templates = ['profile_template', 'goals_template']
        for template_name in required_templates:
            if template_name not in templates:
                raise TemplateLoadError(f"Missing template: {template_name}")
            if not isinstance(templates[template_name], dict):
                raise TemplateLoadError(f"Invalid template structure for: {template_name}")
            if 'system' not in templates[template_name] or 'human' not in templates[template_name]:
                raise TemplateLoadError(f"Missing system or human message in: {template_name}")
                
        # Create ChatPromptTemplates
        profile_template = ChatPromptTemplate.from_messages([
            ("system", templates['profile_template']['system']),
            ("human", templates['profile_template']['human'])
        ])
        
        goals_template = ChatPromptTemplate.from_messages([
            ("system", templates['goals_template']['system']),
            ("human", templates['goals_template']['human'])
        ])
        
        return profile_template, goals_template
        
    except yaml.YAMLError as e:
        raise TemplateLoadError(f"Failed to parse YAML template file: {str(e)}")
    except Exception as e:
        raise TemplateLoadError(f"Failed to load templates: {str(e)}")

class ChainProgressCallbackHandler(BaseCallbackHandler):
    def __init__(self, debug_mode: bool = False):
        """Initialize the handler with optional debug mode"""
        self.debug_mode = debug_mode
        self._sensitive_keys = {'api_key', 'secret', 'password', 'token', 'authorization'}

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from dictionaries"""
        if not isinstance(data, dict):
            return data
            
        sanitized = {}
        for key, value in data.items():
            # Check if the key contains any sensitive terms
            if any(sensitive in key.lower() for sensitive in self._sensitive_keys):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_dict(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
        return sanitized

    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print(f"\n🔄 Starting Chain: {serialized.get('name', 'Unnamed Chain')}")
        if inputs and self.debug_mode:
            sanitized_inputs = self._sanitize_dict(inputs)
            print(f"Debug - Chain Inputs: {sanitized_inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"✅ Chain completed")
        if self.debug_mode and outputs:
            sanitized_outputs = self._sanitize_dict(outputs)
            print(f"Debug - Chain Outputs: {sanitized_outputs}")
        
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs):
        print("\n🤔 LLM is thinking...")
        if self.debug_mode:
            print(f"Debug - Prompt count: {len(prompts)}")
    
    def on_llm_end(self, response, **kwargs):
        print("✨ LLM completed response")
        if self.debug_mode and hasattr(response, 'generations'):
            for gen_idx, generation in enumerate(response.generations):
                for output_idx, output in enumerate(generation):
                    print(f"Debug - Generation {gen_idx+1}, Output {output_idx+1}:")
                    print(f"Content: {output.text}")
                    if hasattr(output, 'generation_info'):
                        sanitized_info = self._sanitize_dict(output.generation_info)
                        print(f"Generation Info: {sanitized_info}")
    
    def on_llm_error(self, error: Exception, **kwargs):
        print(f"❌ LLM Error: {str(error)}")

# Load environment variables from .env
load_dotenv()

# Create handlers and model
progress_handler = ChainProgressCallbackHandler(debug_mode=True)
stream_handler = StreamingStdOutCallbackHandler()
model = ChatOpenAI(
    model="gpt-4", 
    callbacks=[stream_handler],
    streaming=True
)

# Load templates
try:
    profile_template, goals_template = load_templates()
except TemplateLoadError as e:
    print(f"Failed to load templates: {str(e)}")
    raise

# Create the LCEL chain
profile_chain = profile_template | model | StrOutputParser()

# Create a function to combine the profile with original inputs
def combine_inputs(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Combine profile output with original inputs for goals generation.
    
    Args:
        data: Dictionary containing profile output and original input
        
    Returns:
        Combined dictionary for goals template
        
    Raises:
        ChainOutputError: If profile output is missing or invalid
        ProfileValidationError: If original input is invalid
    """
    try:
        # Validate profile output
        if not data.get("profile") or not isinstance(data["profile"], str):
            raise ChainOutputError("Invalid or empty profile output from chain")
        
        # Validate original input
        if not data.get("original_input"):
            raise ProfileValidationError("Missing original input data")
            
        original = data["original_input"]
        if not isinstance(original, dict):
            raise ProfileValidationError("Original input must be a dictionary")
            
        # Validate required fields
        for field in ["coach_goal", "coach_outcomes"]:
            if not original.get(field) or not isinstance(original[field], str):
                raise ProfileValidationError(f"Missing or invalid {field} in original input")
        
        return {
            "learner_profile": data["profile"],
            "coach_goal": original["coach_goal"],
            "coach_outcomes": original["coach_outcomes"]
        }
    except Exception as e:
        # Add context to the error
        raise ChainOutputError(f"Error combining inputs: {str(e)}") from e

# Build the full chain using LCEL with error handling
def create_chain():
    """Create the full chain with error handling"""
    try:
        chain_map = RunnableMap({
            "profile": profile_chain,
            "original_input": RunnablePassthrough()
        })

        return (
            chain_map
            | combine_inputs
            | goals_template 
            | model 
            | StrOutputParser()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create chain: {str(e)}") from e

# Example usage with error handling
def process_profile(input_data: Dict[str, Any]) -> str:
    """
    Process the profile with full error handling.
    
    Args:
        input_data: Dictionary containing profile input data
        
    Returns:
        Generated learning goals
        
    Raises:
        ProfileValidationError: If input validation fails
        ChainOutputError: If chain processing fails
    """
    try:
        # Validate input using ProfileInput
        validated_input = ProfileInput.from_dict(input_data)
        
        # Create and invoke chain
        chain = create_chain()
        print("\n🚀 Starting the learning profile and goals generation pipeline...\n")
        
        result = chain.invoke(
            input_data,
            config={"callbacks": [progress_handler]}
        )
        
        if not result or not isinstance(result, str):
            raise ChainOutputError("Chain produced invalid or empty output")
            
        print("\n📝 Final Result:\n")
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

# Example usage
if __name__ == "__main__":
    example_input = {
        "background": "Computer Science degree, 5 years working with Java enterprise applications",
        "current_role": "Backend Developer at a financial services company",
        "experience_level": "Senior developer, experienced with monolithic applications",
        "coach_goal": "Transform team to adopt microservices architecture",
        "coach_outcomes": "Team should independently design and implement microservices by Q4"
    }

    try:
        result = process_profile(example_input)
        print(result)
    except Exception as e:
        print(f"\n❌ Failed to process profile: {str(e)}")