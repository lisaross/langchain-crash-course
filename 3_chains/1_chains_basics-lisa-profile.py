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
import logging
from datetime import datetime
import tiktoken
from typing import Dict, Any, Optional, List, Union

# Debug Configuration
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'chain_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log') if DEBUG else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Model configuration
GPT4_MAX_TOKENS = 8192  # Maximum context length for GPT-4
TOKEN_WARNING_THRESHOLD = 0.8  # Warn when token usage reaches 80% of limit

class TokenMonitor:
    """Monitors token usage for prompts and responses"""
    
    def __init__(self, model_name: str = "gpt-4"):
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.max_tokens = GPT4_MAX_TOKENS if "gpt-4" in model_name else 4096
    
    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """Count tokens in text or message list"""
        if isinstance(text, str):
            return len(self.tokenizer.encode(text))
        elif isinstance(text, list):
            total = 0
            for message in text:
                if isinstance(message, dict) and "content" in message:
                    total += len(self.tokenizer.encode(message["content"]))
            return total
        return 0
    
    def log_usage(self, prompt_tokens: int, completion_tokens: int):
        """Log token usage statistics"""
        total_tokens = prompt_tokens + completion_tokens
        
        # Calculate cost for this specific usage
        current_cost = self._calculate_cost(
            total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        # Always print token usage summary
        print(f"\n📊 Token Usage Summary:")
        print(f"  • Prompt Tokens: {prompt_tokens:,}")
        print(f"  • Completion Tokens: {completion_tokens:,}")
        print(f"  • Total Tokens: {total_tokens:,}")
        print(f"  • Cost: ${current_cost:.4f}")
        
        # Log detailed information in debug mode
        logger.info(
            f"Token Usage - Prompt: {prompt_tokens}, "
            f"Completion: {completion_tokens}, "
            f"Total: {total_tokens}, "
            f"Cost: ${current_cost:.4f}"
        )
    
    def _calculate_cost(self, total_tokens: int, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None) -> float:
        """
        Calculate estimated cost based on current OpenAI pricing
        
        Args:
            total_tokens: Total tokens (used for backward compatibility)
            prompt_tokens: Number of prompt tokens (if provided)
            completion_tokens: Number of completion tokens (if provided)
        """
        # If individual token counts aren't provided, estimate based on total
        if prompt_tokens is None or completion_tokens is None:
            # Assume a typical 40/60 split between prompt and completion
            prompt_tokens = int(total_tokens * 0.4)
            completion_tokens = total_tokens - prompt_tokens
        
        # GPT-4 pricing (as of 2024)
        if "gpt-4" in self.model_name:
            prompt_cost = (prompt_tokens / 1000) * 0.03  # $0.03 per 1K tokens
            completion_cost = (completion_tokens / 1000) * 0.06  # $0.06 per 1K tokens
        else:
            # GPT-3.5 pricing
            prompt_cost = (prompt_tokens / 1000) * 0.0015  # $0.0015 per 1K tokens
            completion_cost = (completion_tokens / 1000) * 0.002  # $0.002 per 1K tokens
        return prompt_cost + completion_cost
    
    def check_token_limit(self, prompt_tokens: int) -> Tuple[bool, str]:
        """Check if token usage is approaching limits"""
        usage_ratio = prompt_tokens / self.max_tokens
        
        # Always show current usage
        print(f"\n📈 Current Token Usage: {prompt_tokens:,} ({usage_ratio:.1%} of limit)")
        
        if usage_ratio >= TOKEN_WARNING_THRESHOLD:
            warning = (
                f"⚠️ Token usage ({prompt_tokens:,}) is at {usage_ratio:.1%} of the model's "
                f"context limit ({self.max_tokens:,}). Consider optimizing the prompt."
            )
            suggestions = self._get_optimization_suggestions(prompt_tokens)
            return False, f"{warning}\n{suggestions}"
        return True, ""
    
    def _get_optimization_suggestions(self, current_tokens: int) -> str:
        suggestions = ["Optimization suggestions:"]
        if current_tokens > self.max_tokens * 0.5:
            suggestions.extend([
                "• Remove redundant context or examples",
                "• Summarize lengthy descriptions",
                "• Split the task into smaller chunks",
                "• Use more concise language"
            ])
        return "\n".join(suggestions)

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

class ChainOutputValidator:
    """Validates chain outputs against expected criteria"""
    
    PROFILE_REQUIRED_CONCEPTS = {
        'jtbd': ['jtbd', 'jobs to be done', 'job to be done', 'needs to accomplish'],
        'challenges': ['challenges', 'obstacles', 'difficulties', 'pain points'],
        'skills': ['skills gap', 'skill gaps', 'missing skills', 'needs to learn', 'areas for improvement']
    }
    MIN_OUTPUT_LENGTH = 100  # Minimum characters for meaningful output
    
    @classmethod
    def _contains_concept(cls, text: str, concept_variants: List[str]) -> bool:
        """Check if text contains any variant of a concept"""
        return any(variant in text.lower() for variant in concept_variants)
    
    @classmethod
    def validate_profile_output(cls, output: str) -> None:
        """
        Validates the profile chain output.
        
        Args:
            output: String output from profile chain
            
        Raises:
            ChainOutputError: If validation fails
        """
        if not output or len(output.strip()) < cls.MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Profile output is too short or empty")
            
        # Convert to lowercase for case-insensitive checking
        output_lower = output.lower()
        
        # Check for required concepts using their variants
        missing_concepts = [
            concept for concept, variants in cls.PROFILE_REQUIRED_CONCEPTS.items()
            if not cls._contains_concept(output_lower, variants)
        ]
        
        if missing_concepts:
            raise ChainOutputError(
                f"Profile output missing required concepts: {', '.join(missing_concepts)}"
            )
    
    @classmethod
    def validate_goals_output(cls, output: str, coach_goal: str, coach_outcomes: str) -> None:
        """
        Validates the final goals output.
        
        Args:
            output: String output from goals chain
            coach_goal: Original coaching goal
            coach_outcomes: Original desired outcomes
            
        Raises:
            ChainOutputError: If validation fails
        """
        if not output or len(output.strip()) < cls.MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Goals output is too short or empty")
            
        # Check if goals align with organizational objectives
        coach_keywords = set(word.lower() for word in 
                           (coach_goal + " " + coach_outcomes).split())
        output_words = set(output.lower().split())
        
        # Ensure at least 30% of coach keywords are present in output
        matching_keywords = coach_keywords.intersection(output_words)
        if len(matching_keywords) < len(coach_keywords) * 0.3:
            raise ChainOutputError(
                "Goals output does not sufficiently address organizational objectives"
            )
            
        # Verify JTBD alignment using variants
        if not cls._contains_concept(output, cls.PROFILE_REQUIRED_CONCEPTS['jtbd']):
            raise ChainOutputError(
                "Goals output does not explicitly address learner's Jobs To Be Done (JTBD)"
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
    def __init__(self, debug_mode: bool = DEBUG):
        """Initialize the handler with debug mode defaulting to global DEBUG setting"""
        self.debug_mode = debug_mode
        self._sensitive_keys = {'api_key', 'secret', 'password', 'token', 'authorization'}
        self.logger = logging.getLogger(__name__)
        self.token_monitor = TokenMonitor()
        self._current_prompt_tokens = 0  # Track current prompt tokens
        self._current_completion_tokens = 0  # Track current completion tokens
        self._is_final_chain = False  # Track if we're in the final chain
        self._final_chain_complete = False  # Track if final chain is done

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
        chain_name = serialized.get('name', 'Unnamed Chain')
        # Check if this is the final chain
        self._is_final_chain = chain_name == "Learning Goals Generation Chain"
        print(f"\n🔄 Starting Chain: {chain_name}")
        
        if self.debug_mode:
            sanitized_inputs = self._sanitize_dict(inputs)
            self.logger.debug(f"Chain '{chain_name}' starting with inputs:")
            self.logger.debug(f"Serialized config: {serialized}")
            self.logger.debug(f"Sanitized inputs: {sanitized_inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"\n✅ Chain completed")
        if self._is_final_chain:
            self._final_chain_complete = True
            # Print final token usage summary here
            total_prompt_tokens = self.token_monitor.total_prompt_tokens
            total_completion_tokens = self.token_monitor.total_completion_tokens
            total_tokens = total_prompt_tokens + total_completion_tokens
            total_cost = self.token_monitor._calculate_cost(
                total_tokens,
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens
            )
            
            print("\n" + "="*50)
            print("📊 Final Token Usage Summary:")
            print(f"  • Total Prompt Tokens: {total_prompt_tokens:,}")
            print(f"  • Total Completion Tokens: {total_completion_tokens:,}")
            print(f"  • Total Tokens Used: {total_tokens:,}")
            print(f"  • Total Cost: ${total_cost:.4f}")
            print(f"  • Average Cost per 1K tokens: ${(total_cost / total_tokens * 1000):.4f}")
            
            # Add recommendations if token usage is high
            if total_tokens > GPT4_MAX_TOKENS * 0.5:  # If using more than 50% of max tokens
                print("\n💡 Optimization Recommendations:")
                print("  • Consider breaking down the task into smaller chunks")
                print("  • Reduce redundant information in prompts")
                print("  • Use more concise language in prompts")
                print("  • Cache intermediate results for reuse")
                if total_cost > 0.10:  # If cost is over $0.10
                    print("  • Consider using a smaller model for initial drafts")
                    print("  • Implement token usage budgets per request")
            print("="*50 + "\n")
            
        if self.debug_mode and outputs:
            sanitized_outputs = self._sanitize_dict(outputs)
            self.logger.debug("Chain outputs:")
            self.logger.debug(f"Sanitized outputs: {sanitized_outputs}")
    
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs):
        print("\n🤔 LLM is thinking...")
        
        # Token counting for prompts (always show)
        self._current_prompt_tokens = 0
        for idx, prompt in enumerate(prompts, 1):
            prompt_tokens = self.token_monitor.count_tokens(prompt)
            self._current_prompt_tokens += prompt_tokens
            
            # Check token limits and show current usage
            is_safe, warning = self.token_monitor.check_token_limit(prompt_tokens)
            if not is_safe:
                print(warning)
            
        if self.debug_mode:
            self.logger.debug(f"LLM Start - Model: {serialized.get('name', 'Unknown')}")
            self.logger.debug(f"Number of prompts: {len(prompts)}")
            for idx, prompt in enumerate(prompts, 1):
                self.logger.debug(f"\nPrompt {idx}:")
                self.logger.debug("-" * 50)
                self.logger.debug(prompt)
                self.logger.debug("-" * 50)
    
    def on_llm_end(self, response, **kwargs):
        print("\n✨ LLM completed response")
        
        # Get completion tokens from response if available
        completion_tokens = 0
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            completion_tokens = token_usage.get('completion_tokens', 0)
            prompt_tokens = token_usage.get('prompt_tokens', self._current_prompt_tokens)
        else:
            # If not available in response, use our tracked values
            completion_tokens = self._current_completion_tokens
            prompt_tokens = self._current_prompt_tokens
        
        # Always add to totals
        self.token_monitor.total_prompt_tokens += prompt_tokens
        self.token_monitor.total_completion_tokens += completion_tokens
        
        # Show token usage summary only if not in final chain
        if not self._is_final_chain:
            self.token_monitor.log_usage(prompt_tokens, completion_tokens)
        
        # Reset counters
        self._current_completion_tokens = 0
        self._current_prompt_tokens = 0
        
        # Debug mode additional information
        if self.debug_mode and hasattr(response, 'generations'):
            self.logger.debug("\nLLM Response Details:")
            for gen_idx, generation in enumerate(response.generations):
                for output_idx, output in enumerate(generation):
                    self.logger.debug(f"\nGeneration {gen_idx+1}, Output {output_idx+1}:")
                    self.logger.debug("-" * 50)
                    self.logger.debug(f"Content: {output.text}")
                    if hasattr(output, 'generation_info'):
                        sanitized_info = self._sanitize_dict(output.generation_info)
                        self.logger.debug(f"Generation Info: {sanitized_info}")
                    self.logger.debug("-" * 50)
    
    def on_llm_new_token(self, token: str, **kwargs):
        """Track tokens in streaming responses"""
        if token:
            token_count = self.token_monitor.count_tokens(token)
            self._current_completion_tokens += token_count

    def on_chain_error(self, error: Exception, **kwargs):
        error_msg = f"❌ Chain Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)
            
    def on_llm_error(self, error: Exception, **kwargs):
        error_msg = f"❌ LLM Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        if self.debug_mode:
            self.logger.debug(f"Tool Start - {serialized.get('name', 'Unknown Tool')}")
            self.logger.debug(f"Input: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        if self.debug_mode:
            self.logger.debug(f"Tool Output: {output}")

    def on_tool_error(self, error: Exception, **kwargs):
        error_msg = f"❌ Tool Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)

# Load environment variables from .env
load_dotenv()

# Create handlers and model
progress_handler = ChainProgressCallbackHandler()  # Will use global DEBUG setting
stream_handler = StreamingStdOutCallbackHandler()
model = ChatOpenAI(
    model="gpt-4", 
    callbacks=[progress_handler, stream_handler],
    streaming=True
)

# Load templates
try:
    profile_template, goals_template = load_templates()
except TemplateLoadError as e:
    print(f"Failed to load templates: {str(e)}")
    raise

# Create the LCEL chain
profile_chain = (
    profile_template 
    | model.with_config({"callbacks": [progress_handler], "tags": ["profile_generation"]}) 
    | StrOutputParser()
).with_config({"run_name": "Profile Generation Chain"})

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
            
        # Validate profile output content
        ChainOutputValidator.validate_profile_output(data["profile"])
        
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
            
        # Validate final goals output
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
        process_profile(example_input)
    except Exception as e:
        print(f"\n❌ Failed to process profile: {str(e)}")