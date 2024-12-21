"""
Data models for learner profiles and related data structures.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from .validators import ProfileValidationError

@dataclass
class TargetAudience:
    """Data class for target audience information."""
    role: str
    experience_level: str
    current_tech_stack: str
    industry: str
    team_context: str
    primary_challenges: List[str]
    learning_goals: List[str]

@dataclass
class ProfileInput:
    """
    Data class for validating and storing learner profile input data.
    """
    # Original required fields
    background: str
    current_role: str
    experience_level: str
    coach_goal: str
    coach_outcomes: str
    
    # Profile template fields
    course_title: str
    delivery_format: str
    learning_domain: str
    organization_context: str
    
    # JTBD template fields
    target_audience: TargetAudience
    learning_objectives: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileInput':
        """Create and validate a ProfileInput instance from a dictionary."""
        # Original required fields
        original_fields = ['background', 'current_role', 'experience_level', 'coach_goal', 'coach_outcomes']
        
        # New required fields
        profile_fields = ['course_title', 'delivery_format', 'learning_domain', 'organization_context']
        
        # All required string fields
        required_str_fields = original_fields + profile_fields
        
        # Check for missing fields
        missing_fields = [field for field in required_str_fields if field not in data or not data[field]]
        if missing_fields:
            raise ProfileValidationError(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate string fields
        for field in required_str_fields:
            if not isinstance(data[field], str):
                raise ProfileValidationError(f"Field '{field}' must be a string")
            if len(data[field].strip()) < 3:
                raise ProfileValidationError(f"Field '{field}' must contain meaningful content (min 3 characters)")
        
        # Validate target audience
        if 'target_audience' not in data:
            raise ProfileValidationError("Missing target_audience data")
            
        target_fields = ['role', 'experience_level', 'current_tech_stack', 'industry', 
                        'team_context', 'primary_challenges', 'learning_goals']
        
        target_data = data['target_audience']
        missing_target_fields = [field for field in target_fields if field not in target_data]
        if missing_target_fields:
            raise ProfileValidationError(f"Missing target audience fields: {', '.join(missing_target_fields)}")
            
        # Validate lists in target audience
        for list_field in ['primary_challenges', 'learning_goals']:
            if not isinstance(target_data[list_field], list):
                raise ProfileValidationError(f"Field '{list_field}' must be a list")
            if not target_data[list_field]:
                raise ProfileValidationError(f"Field '{list_field}' cannot be empty")
                
        # Validate learning objectives
        if 'learning_objectives' not in data:
            raise ProfileValidationError("Missing learning_objectives")
        if not isinstance(data['learning_objectives'], list):
            raise ProfileValidationError("learning_objectives must be a list")
        if not data['learning_objectives']:
            raise ProfileValidationError("learning_objectives cannot be empty")
        
        target_audience = TargetAudience(
            role=target_data['role'],
            experience_level=target_data['experience_level'],
            current_tech_stack=target_data['current_tech_stack'],
            industry=target_data['industry'],
            team_context=target_data['team_context'],
            primary_challenges=target_data['primary_challenges'],
            learning_goals=target_data['learning_goals']
        )
        
        return cls(
            # Original fields
            background=data['background'],
            current_role=data['current_role'],
            experience_level=data['experience_level'],
            coach_goal=data['coach_goal'],
            coach_outcomes=data['coach_outcomes'],
            
            # Profile template fields
            course_title=data['course_title'],
            delivery_format=data['delivery_format'],
            learning_domain=data['learning_domain'],
            organization_context=data['organization_context'],
            
            # JTBD template fields
            target_audience=target_audience,
            learning_objectives=data['learning_objectives']
        )
