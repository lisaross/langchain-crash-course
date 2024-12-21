import pytest
from typing import Dict, Any
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys
import os
import importlib.util

# Import the module from the original file
spec = importlib.util.spec_from_file_location(
    "chains_module",
    os.path.join(os.path.dirname(__file__), "1_chains_basics-lisa-profile.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules["chains_module"] = module
spec.loader.exec_module(module)

# Import the required classes and functions from the module
process_profile = module.process_profile
ProfileInput = module.ProfileInput
ProfileValidationError = module.ProfileValidationError
ChainOutputError = module.ChainOutputError
ChainOutputValidator = module.ChainOutputValidator

# Configure logging for tests
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'test_chain_debug_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Test Data
@pytest.fixture
def junior_developer_input() -> Dict[str, Any]:
    return {
        "background": "Recent Computer Science graduate with 1 year internship experience in web development",
        "current_role": "Junior Software Developer at a startup",
        "experience_level": "Entry-level, familiar with basic JavaScript and Python",
        "coach_goal": "Develop strong foundation in software engineering best practices",
        "coach_outcomes": "Able to write clean, maintainable code and contribute effectively to team projects"
    }

@pytest.fixture
def senior_to_leadership_input() -> Dict[str, Any]:
    return {
        "background": "10 years of software development experience, tech lead for last 2 years",
        "current_role": "Senior Software Engineer transitioning to Engineering Manager",
        "experience_level": "Senior level, expert in system design and architecture",
        "coach_goal": "Develop leadership and people management skills",
        "coach_outcomes": "Successfully lead and mentor a team of 8 developers, improve team productivity and satisfaction"
    }

@pytest.fixture
def non_technical_pm_input() -> Dict[str, Any]:
    return {
        "background": "5 years in project management, MBA, non-technical background",
        "current_role": "Product Manager in tech company",
        "experience_level": "Mid-level PM, basic understanding of software development lifecycle",
        "coach_goal": "Better understand technical aspects to improve dev team collaboration",
        "coach_outcomes": "Effectively prioritize technical debt and make informed product decisions"
    }

@pytest.fixture
def invalid_input() -> Dict[str, Any]:
    return {
        "background": "",  # Empty background
        "current_role": "Software Developer",
        "experience_level": "Mid-level",
        "coach_goal": "Improve skills",
        "coach_outcomes": None  # Missing outcomes
    }

class TestProfileValidation:
    def test_valid_junior_developer(self, junior_developer_input):
        """Test validation of junior developer profile input"""
        profile_input = ProfileInput.from_dict(junior_developer_input)
        assert isinstance(profile_input, ProfileInput)
        assert profile_input.background == junior_developer_input["background"]
        assert profile_input.coach_goal == junior_developer_input["coach_goal"]

    def test_valid_senior_to_leadership(self, senior_to_leadership_input):
        """Test validation of senior developer transitioning to leadership"""
        profile_input = ProfileInput.from_dict(senior_to_leadership_input)
        assert isinstance(profile_input, ProfileInput)
        assert profile_input.current_role == senior_to_leadership_input["current_role"]
        assert profile_input.experience_level == senior_to_leadership_input["experience_level"]

    def test_valid_non_technical_pm(self, non_technical_pm_input):
        """Test validation of non-technical product manager profile"""
        profile_input = ProfileInput.from_dict(non_technical_pm_input)
        assert isinstance(profile_input, ProfileInput)
        assert profile_input.background == non_technical_pm_input["background"]
        assert profile_input.coach_outcomes == non_technical_pm_input["coach_outcomes"]

    def test_invalid_input(self, invalid_input):
        """Test validation fails with invalid input"""
        with pytest.raises(ProfileValidationError) as exc_info:
            ProfileInput.from_dict(invalid_input)
        assert "Missing required fields" in str(exc_info.value)

class TestChainProcessing:
    @pytest.mark.asyncio
    async def test_process_junior_developer(self, junior_developer_input):
        """Test processing of junior developer profile"""
        result = process_profile(junior_developer_input)
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > ChainOutputValidator.MIN_OUTPUT_LENGTH
        
        # Verify presence of key concepts in a more flexible way
        lower_result = result.lower()
        skills_terms = ['skill', 'skills', 'competenc', 'abilit', 'knowledge']
        challenges_terms = ['challeng', 'difficult', 'struggle', 'issue', 'problem']
        
        assert any(term in lower_result for term in skills_terms), "No skills-related terms found in output"
        assert any(term in lower_result for term in challenges_terms), "No challenges-related terms found in output"

    @pytest.mark.asyncio
    async def test_process_senior_to_leadership(self, senior_to_leadership_input):
        """Test processing of senior developer transitioning to leadership"""
        result = process_profile(senior_to_leadership_input)
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > ChainOutputValidator.MIN_OUTPUT_LENGTH
        
        # Verify leadership-specific content
        lower_result = result.lower()
        leadership_terms = ['lead', 'manage', 'mentor', 'team']
        assert any(term in lower_result for term in leadership_terms)

    @pytest.mark.asyncio
    async def test_process_non_technical_pm(self, non_technical_pm_input):
        """Test processing of non-technical product manager profile"""
        result = process_profile(non_technical_pm_input)
        assert result is not None
        assert isinstance(result, str)
        assert len(result.strip()) > ChainOutputValidator.MIN_OUTPUT_LENGTH
        
        # Verify technical learning focus
        lower_result = result.lower()
        technical_terms = ['technical', 'development', 'engineering', 'architecture']
        assert any(term in lower_result for term in technical_terms)

    def test_process_invalid_input(self, invalid_input):
        """Test processing fails with invalid input"""
        with pytest.raises(ProfileValidationError):
            process_profile(invalid_input)

class TestChainOutputValidation:
    def test_output_validation_requirements(self):
        """Test chain output validation requirements"""
        # Test missing required concepts
        incomplete_output = "This is a short output without required concepts."
        with pytest.raises(ChainOutputError) as exc_info:
            ChainOutputValidator.validate_profile_output(incomplete_output)
        assert "Profile output is too short" in str(exc_info.value)

        # Test output length requirement
        short_output = "Too short"
        with pytest.raises(ChainOutputError) as exc_info:
            ChainOutputValidator.validate_profile_output(short_output)
        assert "Profile output is too short" in str(exc_info.value)

    def test_goals_validation_requirements(self):
        """Test goals output validation requirements"""
        coach_goal = "Improve team's technical skills"
        coach_outcomes = "Team should complete advanced certifications"
        
        # Test insufficient alignment with objectives
        misaligned_output = "Focus on project management and soft skills development."
        with pytest.raises(ChainOutputError) as exc_info:
            ChainOutputValidator.validate_goals_output(misaligned_output, coach_goal, coach_outcomes)
        assert "Goals output is too short" in str(exc_info.value)

if __name__ == "__main__":
    pytest.main(["-v", __file__]) 