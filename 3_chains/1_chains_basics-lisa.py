from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are an expert learning consultant specializing in adult professional development and technical education. 
Your task is to analyze learner information and create detailed, actionable learning profiles that will guide course creation.

Follow these principles:
- Use evidence from their background to justify your analysis
- Focus on practical, job-related learning needs
- Consider both technical and soft skill development
- Identify potential obstacles and learning preferences"""),
    
    ("human", """Please create a detailed learner profile based on this information:

Background: {background}
Experience Level: {experience_level}
Learning Goals: {goals}

Create a structured profile that includes:
1. Key characteristics and current capabilities
2. Learning style preferences and optimal learning environment
3. Potential challenges and growth areas
4. Jobs to be done (JTBD)
5. Success metrics for their learning journey""")
])

# Example variables for testing
variables = {
    "background": "Software developer with 3 years experience in frontend development",
    "experience_level": "Mid-level, comfortable with React and basic JavaScript",
    "goals": "Want to become a technical lead and improve architecture skills"
}

# Create the combined chain using LangChain Expression Language (LCEL)
chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model

# Run the chain
result = chain.invoke(variables)

# Output
print(result)
