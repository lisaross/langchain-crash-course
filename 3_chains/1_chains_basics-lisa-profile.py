from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableMap
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

class ChainProgressCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized: dict, inputs: dict, **kwargs):
        print(f"\n🔄 Starting Chain: {serialized.get('name', 'Unnamed Chain')}")
        if inputs:
            print(f"Inputs: {inputs}")
    
    def on_chain_end(self, outputs: dict, **kwargs):
        print(f"✅ Chain completed")
        
    def on_llm_start(self, *args, **kwargs):
        print("\n🤔 LLM is thinking...")
    
    def on_llm_end(self, *args, **kwargs):
        print("✨ LLM completed response")

# Load environment variables from .env
load_dotenv()

# Create handlers and model
progress_handler = ChainProgressCallbackHandler()
stream_handler = StreamingStdOutCallbackHandler()
model = ChatOpenAI(
    model="gpt-4", 
    callbacks=[stream_handler],
    streaming=True
)

# First Template: Create Learner Profile
profile_template = ChatPromptTemplate.from_messages([
    ("system", """Create comprehensive learner profiles for technical professionals."""),
    
    ("human", """Profile analysis for:
Background: {background}
Role: {current_role}
Tech Experience: {experience_level}

Analyze:
1. Jobs to be Done
2. Challenges
3. Career Path
4. Work Environment
5. Skill Gaps""")
])

# Second Template: Derive Learning Goals
goals_template = ChatPromptTemplate.from_messages([
    ("system", """Transform professional needs into actionable learning goals."""),
    
    ("human", """Given profile:
{learner_profile}

Organization needs:
{coach_goal}
{coach_outcomes}

Create goals that:
1. Address JTBD
2. Resolve challenges
3. Close skill gaps
4. Support career growth
5. Meet org objectives

Per goal, show:
- JTBD alignment
- Challenge resolution
- Org goal fit""")
])

# Create the LCEL chain
profile_chain = profile_template | model | StrOutputParser()

# Create a function to combine the profile with original inputs
def combine_inputs(data):
    return {
        "learner_profile": data["profile"],
        "coach_goal": data["original_input"]["coach_goal"],
        "coach_outcomes": data["original_input"]["coach_outcomes"]
    }

# Build the full chain using LCEL
chain_map = RunnableMap({
    "profile": profile_chain,
    "original_input": RunnablePassthrough()
})

full_chain = (
    chain_map
    | combine_inputs
    | goals_template 
    | model 
    | StrOutputParser()
)

# Example usage
example_input = {
    "background": "Computer Science degree, 5 years working with Java enterprise applications",
    "current_role": "Backend Developer at a financial services company",
    "experience_level": "Senior developer, experienced with monolithic applications",
    "coach_goal": "Transform team to adopt microservices architecture",
    "coach_outcomes": "Team should independently design and implement microservices by Q4"
}

# Invoke the chain with callbacks
print("\n🚀 Starting the learning profile and goals generation pipeline...\n")
result = full_chain.invoke(
    example_input,
    config={"callbacks": [progress_handler]}
)
print("\n📝 Final Result:\n")
print(result)