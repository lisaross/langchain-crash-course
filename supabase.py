# Import required libraries
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import json
from datetime import datetime
import uuid

# Load environment variables
load_dotenv()

# Initialize OpenAI
llm = ChatOpenAI(
    model_name="gpt-4-turbo-preview",
    temperature=0.7,
    openai_api_key=os.getenv('OPENAI_API_KEY')
)

# Initialize Supabase
supabase: Client = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

# Company Context Template
company_context_template = """You are an instructional design expert. Your task is to define the company's learning goals, values, and unique selling points for creating a tailored course.

Define our company context:
- Values: {company_values}
- Unique Perspective (POV): {company_pov}
- Mission: {company_mission}

Include:
1. Key values and goals influencing course design.
2. What makes the company unique in the learning space.

Please format your response as a JSON object with the following structure:
{
    "values": {
        "key_values": [],
        "goals": []
    },
    "pov": {
        "unique_factors": [],
        "market_differentiators": []
    },
    "mission": {
        "core_mission": "",
        "learning_alignment": ""
    }
}
"""

# Create the chain
company_context_prompt = ChatPromptTemplate.from_template(company_context_template)
company_context_chain = LLMChain(
    llm=llm,
    prompt=company_context_prompt,
    verbose=True
)

async def create_course_with_context(
    course_title: str,
    company_values: str,
    company_pov: str,
    company_mission: str
) -> dict:
    try:
        # Create new course
        course_data = {
            "title": course_title,
            "is_active": True
        }
        course_response = supabase.table('courses').insert(course_data).execute()
        course_id = course_response.data[0]['id']

        # Create initial version
        version_data = {
            "course_id": course_id,
            "version_number": 1,
            "is_current": True,
            "metadata": {
                "created_at": datetime.utcnow().isoformat(),
                "template_version": "1.0"
            }
        }
        version_response = supabase.table('course_versions').insert(version_data).execute()
        version_id = version_response.data[0]['id']

        # Generate company context using LangChain
        context_response = await company_context_chain.arun({
            "company_values": company_values,
            "company_pov": company_pov,
            "company_mission": company_mission
        })
        
        # Parse the response to ensure it's valid JSON
        context_json = json.loads(context_response)
        
        # Store company context
        context_data = {
            "version_id": version_id,
            "values": context_json["values"],
            "pov": context_json["pov"],
            "mission": context_json["mission"]
        }
        context_response = supabase.table('company_context').insert(context_data).execute()

        return {
            "course_id": course_id,
            "version_id": version_id,
            "company_context": context_json
        }

    except Exception as e:
        print(f"Error: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        result = await create_course_with_context(
            course_title="Introduction to Python Programming",
            company_values="We value innovation, continuous learning, and practical application of knowledge.",
            company_pov="Our unique approach combines theory with hands-on practice in real-world scenarios.",
            company_mission="To empower individuals with practical programming skills that drive career growth."
        )
        print(json.dumps(result, indent=2))

    asyncio.run(main())