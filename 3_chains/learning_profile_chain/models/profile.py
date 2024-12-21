"""
Data models for learner profiles and related data structures.
"""

from dataclasses import dataclass
from typing import Dict, Any
from .validators import ProfileValidationError

@dataclass
class ProfileInput:
    """
    Data class for validating and storing learner profile input data.
    """
    background: str
    current_role: str
    experience_level: str
    coach_goal: str
    coach_outcomes: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileInput':
        """Create and validate a ProfileInput instance from a dictionary."""
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
