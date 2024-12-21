"""
Callback handlers for monitoring chain progress.
"""

import logging
from typing import Dict, Any, List
from langchain.callbacks.base import BaseCallbackHandler
from ..utils.token_monitor import TokenMonitor
from ..config.constants import DEBUG, SENSITIVE_KEYS

class ChainProgressCallbackHandler(BaseCallbackHandler):
    """Callback handler for monitoring chain progress."""
    
    def __init__(self, debug_mode: bool = DEBUG):
        """Initialize the handler with debug mode."""
        self.debug_mode = debug_mode
        self._sensitive_keys = SENSITIVE_KEYS
        self.logger = logging.getLogger(__name__)
        self.token_monitor = TokenMonitor()
        self._is_final_chain = False

    def _sanitize_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove or mask sensitive information from dictionaries."""
        if not isinstance(data, dict):
            return data
            
        sanitized = {}
        for key, value in data.items():
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
        """Handle chain start event."""
        chain_name = serialized.get('name', 'Unnamed Chain')
        self._is_final_chain = chain_name == "Learning Goals Generation Chain"
        print(f"\n🔄 Starting Chain: {chain_name}")
        
        if self.debug_mode:
            sanitized_inputs = self._sanitize_dict(inputs)
            self.logger.debug(f"Chain '{chain_name}' starting with inputs:")
            self.logger.debug(f"Serialized config: {serialized}")
            self.logger.debug(f"Sanitized inputs: {sanitized_inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        """Handle chain end event."""
        print(f"\n✅ Chain completed")
        
        if self.debug_mode and outputs:
            sanitized_outputs = self._sanitize_dict(outputs)
            self.logger.debug("Chain outputs:")
            self.logger.debug(f"Sanitized outputs: {sanitized_outputs}")
    
    def on_llm_start(self, serialized: dict, prompts: List[str], **kwargs):
        """Handle LLM start event."""
        print("\n🤔 LLM is thinking...")
        
        # Check token limits
        for prompt in prompts:
            prompt_tokens = self.token_monitor.count_tokens(prompt)
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
        """Handle LLM end event."""
        print("\n✨ LLM completed response")
        
        if self.debug_mode and hasattr(response, 'generations'):
            self._log_llm_response_details(response)
    
    def _log_llm_response_details(self, response):
        """Log detailed LLM response information in debug mode."""
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

    def on_chain_error(self, error: Exception, **kwargs):
        """Handle chain errors."""
        error_msg = f"❌ Chain Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)
            
    def on_llm_error(self, error: Exception, **kwargs):
        """Handle LLM errors."""
        error_msg = f"❌ LLM Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)

    def on_tool_start(self, serialized: dict, input_str: str, **kwargs):
        """Handle tool start event."""
        if self.debug_mode:
            self.logger.debug(f"Tool Start - {serialized.get('name', 'Unknown Tool')}")
            self.logger.debug(f"Input: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        """Handle tool end event."""
        if self.debug_mode:
            self.logger.debug(f"Tool Output: {output}")

    def on_tool_error(self, error: Exception, **kwargs):
        """Handle tool errors."""
        error_msg = f"❌ Tool Error: {str(error)}"
        print(error_msg)
        if self.debug_mode:
            self.logger.error(error_msg, exc_info=True)
