"""
Example usage of the Learning Profile Chain with updated templates.
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
    # Example input data for course creation
    example_input = {
        # Required fields from original template
        "background": "Java enterprise applications with 5+ years in software development",
        "current_role": "Backend Developer in Enterprise Financial Services",
        "experience_level": "Senior developer with expertise in monolithic architecture, Java, Spring Framework",
        "coach_goal": "Transform development teams to effectively adopt microservices architecture",
        "coach_outcomes": "Teams should be able to design, implement, and deploy scalable microservices by end of training",
        
        # Required fields for profile template
        "course_title": "Microservices Architecture Mastery",
        "delivery_format": "Hybrid (online + live sessions)",
        "learning_domain": "Software Architecture",
        "organization_context": "Enterprise Financial Services",
        
        # Required fields for JTBD template
        "target_audience": {
            "role": "Backend Developer",
            "experience_level": "Senior (5+ years)",
            "current_tech_stack": "Java, Spring Framework, Monolithic Architecture",
            "industry": "Financial Services",
            "team_context": "Part of a development team transitioning to microservices",
            "primary_challenges": [
                "Managing complex dependencies in monolithic systems",
                "Ensuring system stability during architectural transition",
                "Coordinating team adoption of new practices"
            ],
            "learning_goals": [
                "Master microservices design patterns",
                "Implement effective service communication",
                "Lead team in architectural transformation"
            ]
        },
        
        # Required learner profile concepts
        "jtbd": [
            "Modernize legacy monolithic applications",
            "Design and implement scalable microservices",
            "Improve system reliability and maintainability",
            "Enable faster feature delivery through independent services"
        ],
        "challenges": [
            "Complex dependencies in existing monolithic system",
            "Limited experience with distributed systems",
            "Need to maintain system stability during transition",
            "Team resistance to architectural changes"
        ],
        "skills": {
            "current": [
                "Java development",
                "Spring Framework",
                "Monolithic architecture",
                "SQL databases",
                "REST APIs"
            ],
            "desired": [
                "Microservices architecture patterns",
                "Service discovery",
                "Event-driven architecture",
                "Container orchestration",
                "Distributed systems design"
            ]
        },
        
        # Additional context (new template fields)
        "company_context": {
            "values": "Innovation, Collaboration, Excellence",
            "pov": "Practical, hands-on learning with real-world applications",
            "mission": "Empowering developers to master modern architecture patterns"
        },
        
        "course_info": {
            "topic": "Microservices Architecture and Design Patterns",
            "length": "8 weeks"
        },
        
        "learning_objectives": [
            "Design scalable microservices architectures",
            "Implement service discovery and communication patterns",
            "Apply domain-driven design principles",
            "Deploy and monitor microservices effectively"
        ],
        
        "delivery_details": {
            "self_paced_modules": True,
            "live_sessions": True,
            "hands_on_workshops": True,
            "office_hours": True
        }
    }

    try:
        # Process with default model
        course_profile = process_profile(example_input)
        
        print("\n✅ Generated Course Profile:")
        print("\nLearner Profile:")
        print(course_profile.get('learner_profile', 'No learner profile generated'))
        
        print("\nCourse Goals:")
        print(course_profile.get('course_goals', 'No course goals generated'))
        
        print("\nCourse Structure:")
        print(course_profile.get('course_structure', 'No course structure generated'))
        
        print("\nJTBD Analysis:")
        print(course_profile.get('jtbd_analysis', 'No JTBD analysis generated'))
        
    except Exception as e:
        print(f"\n❌ Failed to process course profile: {str(e)}")

if __name__ == "__main__":
    # Print active model configuration
    print_active_model()
    main()
