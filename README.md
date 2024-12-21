# **Instructional Design Workflow with AI Templates**

This project provides a streamlined workflow for creating, managing, and enhancing instructional design processes using AI-powered templates. The system leverages tools like LangChain, dynamic templates, and a database backend (e.g., Firebase or Supabase) to design, deliver, and iterate on learning programs efficiently.

---

## **Features**

- **AI-Powered Templates**: Automates key instructional design tasks, such as creating JTBD analyses, learner profiles, course objectives, and more.
- **Version Control**: Tracks and manages multiple versions of template outputs for iterative design and auditing.
- **Database Integration**: Stores outputs for modules, learner profiles, and program transformations for reuse across workflows.
- **Engagement Enhancements**:
  - Suggests polls, memes, comics, and pop culture references to improve learner engagement.
  - Recommends opportunities for personalization using AI or manual methods.
- **Output Customization**:
  - Generates slide decks, case studies, and learning assessments tailored to the course objectives.
  - Incorporates visual elements like DALL-E-generated images for enhanced learning materials.
- **Feedback and Analytics**:
  - Collects learner feedback and performance data to refine programs iteratively.
  - Defines success metrics for courses and modules.

---

## **Getting Started**

### Prerequisites

1. **Python Environment**:
   - Python 3.8 or later.
   - Install required libraries:
     ```bash
     pip install langchain firebase-admin supabase-client openai
     ```

2. **Database**:
   - Firebase Firestore or Supabase (PostgreSQL-based).
   - Set up a database schema to store template outputs and version control.

3. **AI API Key**:
   - OpenAI API key for generating template outputs.

4. **Optional**:
   - Access to DALL-E for generating visuals.
   - A Zoom account for implementing polls in live sessions.

---

## **Folder Structure**

project/
├── templates/                  # AI templates for each task
│   ├── jtbd_template.yaml      # Jobs To Be Done template
│   ├── learner_profile.yaml    # Learner profile creation
│   ├── course_objectives.yaml  # SMART course objectives
│   ├── engagement.yaml         # Polls, memes, and pop culture suggestions
│   ├── assessments.yaml        # Assessment and feedback generation
│   └── transformation.yaml     # Learner transformation analysis
├── db/                         # Database integration scripts
│   ├── firebase_client.py      # Firebase database management
│   ├── supabase_client.py      # Supabase database management
│   └── versioning.py           # Version control logic
├── outputs/                    # Example outputs for templates
├── README.md                   # Project documentation
└── main.py                     # Main execution file

---

## **Usage**

### 1. **Generate Template Outputs**
Run individual templates by invoking them with relevant inputs. For example:

```python
from templates.jtbd_template import jtbd_template
from langchain.schema.runnable import RunnableMap

# Define inputs
inputs = {
    "course_title": "Mastering Microservices Architecture",
    "target_audience": "Mid-level software developers and technical leads",
    "learning_objectives": [
        "Understand the fundamentals of microservices",
        "Design scalable, modular systems",
        "Implement API gateways and scalable solutions"
    ],
    "delivery_format": "Self-paced with live Q&A"
}

# Invoke template
output = jtbd_template.invoke(inputs)
print(output)

2. Store Outputs in the Database

Save outputs for reuse and version tracking:

from db.versioning import store_template_output_with_version

store_template_output_with_version(
    course_id="1234-5678",
    course_title="Mastering Microservices Architecture",
    template_name="jtbd_template",
    output_data=output
)

3. Retrieve Outputs for Reuse

Fetch stored outputs dynamically for future templates or workflows:

from db.versioning import get_current_template_output

output = get_current_template_output(course_id="1234-5678", template_name="jtbd_template")
print(output)

4. Generate Slide Decks and Visuals

Generate slides and add visuals using integrated AI tools:

from templates.transformation import transformation_template
from dalle_client import generate_image

# Generate a DALL-E image
prompt = "A high-tech server room with glowing API gateway icons connected to client devices."
image_url = generate_image(prompt)

# Use image in slide creation
print(f"Generated image URL: {image_url}")

5. Polls and Engagement

Embed polls, memes, and comics into the course for better learner engagement:

from templates.engagement import engagement_template

inputs = {
    "course_title": "Mastering Microservices Architecture",
    "modules": [
        {"title": "Introduction to Microservices", "focus": "Key concepts and benefits"},
        {"title": "Designing Scalable Systems", "focus": "Best practices for scalability"},
    ]
}
output = engagement_template.invoke(inputs)
print(output)
