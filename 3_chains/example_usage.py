"""
Example usage of the Learning Profile Chain.
"""

from learning_profile_chain import process_profile
from learning_profile_chain.config.constants import MODEL_CONFIGS, DEFAULT_MODEL

def print_active_model():
    """Print information about the currently active model."""
    model_config = MODEL_CONFIGS[DEFAULT_MODEL]
    print("\n🤖 Active Model Configuration:")
    print(f"  Model: {DEFAULT_MODEL}")
    print(f"  Max Tokens: {model_config['max_tokens']:,}")
    print(f"  Description: {model_config['description']}")
    print("=" * 80)

def main():
    # Example input data
    example_input = {
        "background": "Computer Science degree, 5 years working with Java enterprise applications",
        "current_role": "Backend Developer at a financial services company",
        "experience_level": "Senior developer, experienced with monolithic applications",
        "coach_goal": "Transform team to adopt microservices architecture",
        "coach_outcomes": "Team should independently design and implement microservices by Q4"
    }

    try:
        # Process with default model
        learning_goals = process_profile(example_input)
        print("\n✅ Generated Learning Goals:")
        print(learning_goals)
        
    except Exception as e:
        print(f"\n❌ Failed to process profile: {str(e)}")

if __name__ == "__main__":
    # Print active model configuration
    print_active_model()
    main()
