from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Custom Exceptions
class ProfileValidationError(Exception):
    """Raised when profile data validation fails"""
    pass

class ChainOutputError(Exception):
    """Raised when chain output is invalid or empty"""
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

class ChainProgressCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print(f"\n🔄 Starting Chain: {serialized.get('name', 'Unnamed Chain')}")
        if inputs:
            print(f"Inputs: {inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"✅ Chain completed")
        
    def on_llm_start(self, *args, **kwargs):
        print("\n🤔 LLM is thinking...")
    
    def on_llm_end(self, *args, **kwargs):
        print("✨ LLM completed response")

# Load environment variables from .env
load_dotenv()

# Create handlers and model
progress_handler = ChainProgressCallbackHandler()
stream_handler = StreamingStdOutCallbackHandler()
model = ChatOpenAI(
    model="gpt-4", 
    callbacks=[stream_handler],
    streaming=True
)

# First Template: Create Learner Profile
profile_template = ChatPromptTemplate.from_messages([
    ("system", """Create comprehensive learner profiles for technical professionals."""),
    
    ("human", """Profile analysis for:
Background: {background}
Role: {current_role}
Tech Experience: {experience_level}

Analyze:
1. Jobs to be Done
2. Challenges
3. Career Path
4. Work Environment
5. Skill Gaps""")
])

# Second Template: Derive Learning Goals
goals_template = ChatPromptTemplate.from_messages([
    ("system", """Transform professional needs into actionable learning goals."""),
    
    ("human", """Given profile:
{learner_profile}

Organization needs:
{coach_goal}
{coach_outcomes}

Create goals that:
1. Address JTBD
2. Resolve challenges
3. Close skill gaps
4. Support career growth
5. Meet org objectives

Per goal, show:
- JTBD alignment
- Challenge resolution
- Org goal fit""")
])

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