"""
Template loading functionality for the learning profile chain.
"""

from pathlib import Path
import yaml
from typing import Tuple
from langchain.prompts import ChatPromptTemplate
from ..models.validators import TemplateLoadError
from ..config.constants import TEMPLATE_FILE

def load_templates(template_path: str = TEMPLATE_FILE) -> Tuple[ChatPromptTemplate, ChatPromptTemplate, ChatPromptTemplate]:
    """
    Load and validate prompt templates from a YAML file.

    Args:
        template_path: Path to the YAML template file.

    Returns:
        Tuple containing profile_template, jtbd_template, and goals_template.

    Raises:
        TemplateLoadError: If template loading or validation fails.
    """
    try:
        script_dir = Path(__file__).parent
        full_path = script_dir / template_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"Template file not found: {full_path}")
            
        with open(full_path, 'r') as file:
            templates = yaml.safe_load(file)
            
        required_templates = ['profile_template', 'jtbd_template', 'goals_template']
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
        
        jtbd_template = ChatPromptTemplate.from_messages([
            ("system", templates['jtbd_template']['system']),
            ("human", templates['jtbd_template']['human'])
        ])
        
        goals_template = ChatPromptTemplate.from_messages([
            ("system", templates['goals_template']['system']),
            ("human", templates['goals_template']['human'])
        ])
        
        return profile_template, jtbd_template, goals_template
        
    except yaml.YAMLError as e:
        raise TemplateLoadError(f"Failed to parse YAML template file: {str(e)}")
    except Exception as e:
        raise TemplateLoadError(f"Failed to load templates: {str(e)}")
