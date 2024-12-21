"""
Validation classes and exceptions for the learning profile chain.
"""

from typing import Dict, Any, List
from ..config.constants import MIN_OUTPUT_LENGTH, PROFILE_REQUIRED_CONCEPTS

class ProfileValidationError(Exception):
    """Raised when profile data validation fails."""
    pass

class ChainOutputError(Exception):
    """Raised when chain output is invalid or empty."""
    pass

class TemplateLoadError(Exception):
    """Raised when template loading fails."""
    pass

class ChainOutputValidator:
    """Validates the output from LLM chains against predefined criteria."""
    
    @classmethod
    def _contains_concept(cls, text: str, concept_variants: List[str]) -> bool:
        """Check if text contains any variant of a given concept."""
        return any(variant in text.lower() for variant in concept_variants)
    
    @classmethod
    def validate_profile_output(cls, output: str) -> None:
        """Validates the profile chain output against required criteria."""
        if not output or len(output.strip()) < MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Profile output is too short or empty")
            
        output_lower = output.lower()
        
        missing_concepts = [
            concept for concept, variants in PROFILE_REQUIRED_CONCEPTS.items()
            if not cls._contains_concept(output_lower, variants)
        ]
        
        if missing_concepts:
            raise ChainOutputError(
                f"Profile output missing required concepts: {', '.join(missing_concepts)}"
            )
    
    @classmethod
    def validate_goals_output(cls, output: str, coach_goal: str, coach_outcomes: str) -> None:
        """Validates the final goals output against coaching objectives."""
        if not output or len(output.strip()) < MIN_OUTPUT_LENGTH:
            raise ChainOutputError("Goals output is too short or empty")
            
        coach_keywords = set(word.lower() for word in 
                           (coach_goal + " " + coach_outcomes).split())
        output_words = set(output.lower().split())
        
        matching_keywords = coach_keywords.intersection(output_words)
        if len(matching_keywords) < len(coach_keywords) * 0.3:
            raise ChainOutputError(
                "Goals output does not sufficiently address organizational objectives"
            )
            
        if not cls._contains_concept(output, PROFILE_REQUIRED_CONCEPTS['jtbd']):
            raise ChainOutputError(
                "Goals output does not explicitly address learner's Jobs To Be Done (JTBD)"
            )
