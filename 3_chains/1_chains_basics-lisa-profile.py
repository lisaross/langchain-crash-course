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
    """
    Monitors and tracks token usage for LLM prompts and responses.

    This class provides functionality to count tokens, track usage statistics,
    and calculate associated costs for different LLM models. It includes
    built-in support for GPT-4 and GPT-3.5 pricing tiers.

    Attributes:
        model_name (str): Name of the LLM model being monitored
        tokenizer: Instance of tiktoken encoder for the specified model
        total_prompt_tokens (int): Running count of tokens used in prompts
        total_completion_tokens (int): Running count of tokens used in completions
        max_tokens (int): Maximum context length for the model

    Key Constraints:
        - Assumes GPT-4 or GPT-3.5 pricing tiers
        - Token limits are model-specific (8192 for GPT-4, 4096 for others)
        - Warning threshold set at 80% of token limit
    """
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize the TokenMonitor with a specific model configuration.

        Args:
            model_name (str, optional): Name of the LLM model to monitor. Defaults to "gpt-4".
        """
        self.model_name = model_name
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.max_tokens = GPT4_MAX_TOKENS if "gpt-4" in model_name else 4096
    
    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """
        Count the number of tokens in text or a list of messages.

        Args:
            text: Either a string or a list of message dictionaries with 'content' keys

        Returns:
            int: Total number of tokens in the input

        Note:
            For message lists, only processes dictionaries with 'content' keys
        """
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
        """
        Log token usage statistics and calculate associated costs.

        Args:
            prompt_tokens (int): Number of tokens used in the prompt
            completion_tokens (int): Number of tokens used in the completion

        Outputs:
            Prints a formatted summary of token usage and costs
            Logs detailed information when in debug mode
        """
        total_tokens = prompt_tokens + completion_tokens
        
        current_cost = self._calculate_cost(
            total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens
        )
        
        print(f"\n📊 Token Usage Summary:")
        print(f"  • Prompt Tokens: {prompt_tokens:,}")
        print(f"  • Completion Tokens: {completion_tokens:,}")
        print(f"  • Total Tokens: {total_tokens:,}")
        print(f"  • Cost: ${current_cost:.4f}")
        
        logger.info(
            f"Token Usage - Prompt: {prompt_tokens}, "
            f"Completion: {completion_tokens}, "
            f"Total: {total_tokens}, "
            f"Cost: ${current_cost:.4f}"
        )
    
    def _calculate_cost(self, total_tokens: int, prompt_tokens: Optional[int] = None, completion_tokens: Optional[int] = None) -> float:
        """
        Calculate the estimated cost based on current OpenAI pricing tiers.

        Args:
            total_tokens (int): Total number of tokens used
            prompt_tokens (Optional[int]): Number of tokens in prompts
            completion_tokens (Optional[int]): Number of tokens in completions

        Returns:
            float: Estimated cost in USD

        Note:
            If individual token counts aren't provided, assumes 40/60 prompt/completion split
        """
        if prompt_tokens is None or completion_tokens is None:
            prompt_tokens = int(total_tokens * 0.4)
            completion_tokens = total_tokens - prompt_tokens
        
        if "gpt-4" in self.model_name:
            prompt_cost = (prompt_tokens / 1000) * 0.03
            completion_cost = (completion_tokens / 1000) * 0.06
        else:
            prompt_cost = (prompt_tokens / 1000) * 0.0015
            completion_cost = (completion_tokens / 1000) * 0.002
        return prompt_cost + completion_cost
    
    def check_token_limit(self, prompt_tokens: int) -> Tuple[bool, str]:
        """
        Check if token usage is approaching model limits and provide warnings.

        Args:
            prompt_tokens (int): Number of tokens to check against limit

        Returns:
            Tuple[bool, str]: (is_safe, warning_message)
                - is_safe: False if usage exceeds warning threshold
                - warning_message: Empty string or warning with optimization suggestions

        Note:
            Warning threshold is set at TOKEN_WARNING_THRESHOLD (80%) of model's context limit
        """
        usage_ratio = prompt_tokens / self.max_tokens
        
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
        """
        Generate optimization suggestions based on current token usage.

        Args:
            current_tokens (int): Current token count to evaluate

        Returns:
            str: Formatted string of optimization suggestions if usage is high
        """
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
    """
    Data class for validating and storing learner profile input data.

    This class ensures that all required profile fields are present and valid
    before processing through the LLM chain.

    Attributes:
        background (str): Professional and educational background
        current_role (str): Current job role or position
        experience_level (str): Level of professional experience
        coach_goal (str): Primary goal for the coaching engagement
        coach_outcomes (str): Desired outcomes from the coaching

    Raises:
        ProfileValidationError: If any field is missing, empty, or invalid
    """
    background: str
    current_role: str
    experience_level: str
    coach_goal: str
    coach_outcomes: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileInput':
        """
        Create and validate a ProfileInput instance from a dictionary.

        Args:
            data (Dict[str, Any]): Dictionary containing profile input fields

        Returns:
            ProfileInput: Validated instance of ProfileInput

        Raises:
            ProfileValidationError: If validation fails for any field
                - Missing required fields
                - Non-string field values
                - Empty or too short field values (< 3 chars)

        Example:
            >>> data = {
                    "background": "Computer Science degree",
                    "current_role": "Backend Developer",
                    "experience_level": "Senior",
                    "coach_goal": "Learn microservices",
                    "coach_outcomes": "Build scalable systems"
                }
            >>> profile = ProfileInput.from_dict(data)
        """
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
    """
    Validates the output from LLM chains against predefined criteria.

    This class ensures that chain outputs meet quality standards by checking for:
    - Required concepts and themes
    - Minimum content length
    - Alignment with coaching goals and outcomes

    Class Attributes:
        PROFILE_REQUIRED_CONCEPTS (dict): Required concepts and their variants
        MIN_OUTPUT_LENGTH (int): Minimum character length for valid outputs

    Note:
        The validator uses case-insensitive matching for concept detection
    """
    
    PROFILE_REQUIRED_CONCEPTS = {
        'jtbd': ['jtbd', 'jobs to be done', 'job to be done', 'needs to accomplish'],
        'challenges': ['challenges', 'obstacles', 'difficulties', 'pain points'],
        'skills': ['skills gap', 'skill gaps', 'missing skills', 'needs to learn', 'areas for improvement']
    }
    MIN_OUTPUT_LENGTH = 100

    @classmethod
    def _contains_concept(cls, text: str, concept_variants: List[str]) -> bool:
        """
        Check if text contains any variant of a given concept.

        Args:
            text (str): Text to check for concept variants
            concept_variants (List[str]): List of possible concept variations

        Returns:
            bool: True if any variant is found in the text
        """
        return any(variant in text.lower() for variant in concept_variants)
    
    @classmethod
    def validate_profile_output(cls, output: str) -> None:
        """
        Validates the profile chain output against required criteria.

        Args:
            output (str): Generated profile text to validate

        Raises:
            ChainOutputError: If validation fails due to:
                - Empty or too short output
                - Missing required concepts
                - Insufficient content quality

        Note:
            All concept checking is case-insensitive
        """
        if not output or len(output.strip()) < cls.MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Profile output is too short or empty")
            
        output_lower = output.lower()
        
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
        Validates the final goals output against coaching objectives.

        Args:
            output (str): Generated goals text to validate
            coach_goal (str): Original coaching goal
            coach_outcomes (str): Original desired outcomes

        Raises:
            ChainOutputError: If validation fails due to:
                - Empty or too short output
                - Insufficient alignment with coaching objectives
                - Missing JTBD (Jobs To Be Done) alignment

        Note:
            Requires at least 30% keyword overlap with coaching objectives
        """
        if not output or len(output.strip()) < cls.MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Goals output is too short or empty")
            
        coach_keywords = set(word.lower() for word in 
                           (coach_goal + " " + coach_outcomes).split())
        output_words = set(output.lower().split())
        
        matching_keywords = coach_keywords.intersection(output_words)
        if len(matching_keywords) < len(coach_keywords) * 0.3:
            raise ChainOutputError(
                "Goals output does not sufficiently address organizational objectives"
            )
            
        if not cls._contains_concept(output, cls.PROFILE_REQUIRED_CONCEPTS['jtbd']):
            raise ChainOutputError(
                "Goals output does not explicitly address learner's Jobs To Be Done (JTBD)"
            )

def load_templates(template_path: str = "templates.yaml") -> Tuple[ChatPromptTemplate, ChatPromptTemplate]:
    """
    Load and validate prompt templates from a YAML file.

    Args:
        template_path (str, optional): Path to the YAML template file. Defaults to "templates.yaml".

    Returns:
        Tuple[ChatPromptTemplate, ChatPromptTemplate]: A tuple containing:
            - profile_template: Template for generating learner profiles
            - goals_template: Template for generating learning goals

    Raises:
        TemplateLoadError: If template file is:
            - Missing or inaccessible
            - Malformed YAML
            - Missing required templates
            - Missing required message components

    Note:
        Templates must contain both 'system' and 'human' message components
    """
    try:
        script_dir = Path(__file__).parent
        full_path = script_dir / template_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Template file not found: {full_path}")
            
        with open(full_path, 'r') as file:
            templates = yaml.safe_load(file)
            
        required_templates = ['profile_template', 'goals_template']
        for template_name in required_templates:
            if template_name not in templates:
                raise TemplateLoadError(f"Missing template: {template_name}")
            if not isinstance(templates[template_name], dict):
                raise TemplateLoadError(f"Invalid template structure for: {template_name}")
            if 'system' not in templates[template_name] or 'human' not in templates[template_name]:
                raise TemplateLoadError(f"Missing system or human message in: {template_name}")
                
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

def combine_inputs(data: Dict[str, Any]) -> Dict[str, str]:
    """
    Combine profile chain output with original inputs for goals generation.

    This function validates and merges the profile generation output with the
    original coaching inputs to prepare data for the goals generation phase.

    Args:
        data (Dict[str, Any]): Dictionary containing:
            - profile: Generated profile text from first chain
            - original_input: Original coaching input data

    Returns:
        Dict[str, str]: Combined dictionary containing:
            - learner_profile: Validated profile text
            - coach_goal: Original coaching goal
            - coach_outcomes: Original desired outcomes

    Raises:
        ChainOutputError: If profile output is invalid or missing
        ProfileValidationError: If original input is invalid or missing required fields

    Note:
        All inputs are validated before combination to ensure data integrity
    """
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

def create_chain():
    """
    Create the full learning goals generation chain using LCEL.

    This function constructs a chain that:
    1. Generates a learner profile
    2. Combines profile with original inputs
    3. Generates personalized learning goals

    Returns:
        Runnable: A complete chain that can be invoked with profile input data

    Raises:
        RuntimeError: If chain creation fails due to configuration or setup issues

    Note:
        The chain includes progress tracking and validation at each step
    """
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

def process_profile(input_data: Dict[str, Any]) -> str:
    """
    Process a learner profile to generate personalized learning goals.

    This is the main entry point for the learning goals generation pipeline.
    It handles the complete process from input validation through goals generation.

    Args:
        input_data (Dict[str, Any]): Dictionary containing:
            - background: Professional and educational background
            - current_role: Current job role
            - experience_level: Level of professional experience
            - coach_goal: Primary coaching goal
            - coach_outcomes: Desired coaching outcomes

    Returns:
        str: Generated learning goals aligned with coaching objectives

    Raises:
        ProfileValidationError: If input validation fails
        ChainOutputError: If chain processing fails
        RuntimeError: For unexpected errors during processing

    Example:
        >>> input_data = {
                "background": "Computer Science degree, 5 years Java experience",
                "current_role": "Backend Developer",
                "experience_level": "Senior developer",
                "coach_goal": "Transform team to microservices",
                "coach_outcomes": "Independent microservices implementation"
            }
        >>> goals = process_profile(input_data)
    """
    try:
        validated_input = ProfileInput.from_dict(input_data)
        
        chain = create_chain()
        print("\n🚀 Starting the learning profile and goals generation pipeline...\n")
        
        result = chain.invoke(
            input_data,
            config={"callbacks": [progress_handler]}
        )
        
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