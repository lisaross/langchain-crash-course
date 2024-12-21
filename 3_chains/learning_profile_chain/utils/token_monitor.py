"""
Token monitoring utility for tracking LLM token usage.
"""

from typing import Dict, Any, List, Union, Tuple
import tiktoken
from ..config.constants import MODEL_CONFIGS, DEFAULT_MODEL, TOKEN_WARNING_THRESHOLD

class TokenMonitor:
    """
    Monitors and tracks token usage for LLM prompts and responses.
    Supports O1, O4, and GPT-4 models.
    """
    
    def __init__(self, model_name: str = DEFAULT_MODEL):
        """
        Initialize the token monitor for a specific model.
        
        Args:
            model_name: Name of the model to use. Defaults to DEFAULT_MODEL.
                Available models: gpt-o1, gpt-4o, gpt-4o-mini, gpt-4o-mini-audio-preview, gpt-4, gpt-4-32k
        """
        # Normalize model name and get config
        self.model_name = self._normalize_model_name(model_name)
        if self.model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model: {model_name}")
            
        model_config = MODEL_CONFIGS[self.model_name]
        
        # Initialize tokenizer and limits
        self.tokenizer = tiktoken.encoding_for_model('gpt-4')  # Use GPT-4 tokenizer for all models
        self.max_tokens = model_config['max_tokens']
    
    def _normalize_model_name(self, model_name: str) -> str:
        """
        Normalize the model name to match our config keys.
        Handles common variations in model names.
        """
        model_name = model_name.lower().strip()
        
        # Handle O1 model
        if 'o1' in model_name:
            return 'gpt-o1'
            
        # Handle O4 models
        elif model_name.startswith('gpt-4o'):
            if 'mini-audio' in model_name:
                return 'gpt-4o-mini-audio-preview'
            elif 'mini' in model_name:
                return 'gpt-4o-mini'
            return 'gpt-4o'
        
        # Handle GPT-4 variants
        elif model_name.startswith('gpt-4'):
            if '32k' in model_name:
                return 'gpt-4-32k'
            return 'gpt-4'
            
        # Default to gpt-4o-mini if unknown
        return DEFAULT_MODEL
    
    def count_tokens(self, text: Union[str, List[Dict[str, str]]]) -> int:
        """Count tokens in text or message list."""
        if isinstance(text, str):
            return len(self.tokenizer.encode(text))
        elif isinstance(text, list):
            total = 0
            for message in text:
                if isinstance(message, dict) and "content" in message:
                    total += len(self.tokenizer.encode(message["content"]))
            return total
        return 0
    
    def check_token_limit(self, prompt_tokens: int) -> Tuple[bool, str]:
        """Check if token usage is approaching model limits."""
        usage_ratio = prompt_tokens / self.max_tokens
        
        if usage_ratio >= TOKEN_WARNING_THRESHOLD:
            warning = (
                f"⚠️ Token usage ({prompt_tokens:,}) is at {usage_ratio:.1%} of the model's "
                f"context limit ({self.max_tokens:,}). Consider optimizing the prompt."
            )
            suggestions = self._get_optimization_suggestions(prompt_tokens)
            return False, f"{warning}\n{suggestions}"
        return True, ""
    
    def _get_optimization_suggestions(self, current_tokens: int) -> str:
        """Generate optimization suggestions based on token usage."""
        suggestions = ["Optimization suggestions:"]
        if current_tokens > self.max_tokens * 0.5:
            suggestions.extend([
                "• Remove redundant context or examples",
                "• Summarize lengthy descriptions",
                "• Split the task into smaller chunks",
                "• Use more concise language",
                f"• Consider using {self._suggest_alternative_model()}"
            ])
        return "\n".join(suggestions)
    
    def _suggest_alternative_model(self) -> str:
        """Suggest an alternative model with larger context if available."""
        if self.model_name == 'gpt-4':
            return 'gpt-4-32k for larger context, or gpt-4o/gpt-o1 for enhanced capabilities'
        elif self.model_name == 'gpt-4-32k':
            return 'gpt-o1 or gpt-4o for enhanced capabilities with large context'
        elif self.model_name == 'gpt-o1':
            return 'already using most capable reasoning model'
        elif self.model_name == 'gpt-4o':
            return 'gpt-o1 for specialized reasoning and coding tasks'
        elif self.model_name == 'gpt-4o-mini':
            return 'gpt-4o for complex tasks or gpt-4-32k for larger context'
        elif self.model_name == 'gpt-4o-mini-audio-preview':
            return 'standard gpt-4o-mini if audio processing is not needed'
        return 'a model with larger context limits if available'
