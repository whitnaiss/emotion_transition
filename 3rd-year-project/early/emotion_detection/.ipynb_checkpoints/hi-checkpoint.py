from langchain_core.prompts.prompt import PromptTemplate
from langsmith.evaluation import LangChainStringEvaluator
from langchain_openai import ChatOpenAI
import requests

_PROMPT_TEMPLATE = """ 
You are an expert professor specializing in emotion detection and analysis.

I will provide you with a document. Your task is to analyze it sentence by sentence and assign the most appropriate emotion label 
to each sentence. The possible emotion labels, along with their corresponding numbers, are as follows:

1. Aesthetic Experience
2. Anger
3. Anxiety
4. Compassion
5. Depression
6. Envy
7. Fright
8. Gratitude
9. Guilt
10. Happiness
11. Hope
12. Jealousy
13. Love
14. Pride
15. Relief
16. Sadness
17. Shame
Please ensure that each sentence is labeled with only one number corresponding to the emotion that best reflects its content or tone.

Response format:
First sentence: 5
Second sentence: 8
Third sentence: 12
...
"""

PROMPT = PromptTemplate(
    input_variables=["query1", "query2"], template=_PROMPT_TEMPLATE
)

# Initialize the evaluator with the local model
chat = ChatOpenAI(
    model="llama",
    openai_api_key="EMPTY",
    openai_api_base="http://localhost:8111/v1",
    temperature=0.0,
)

qa_evaluator = LangChainStringEvaluator("qa", config={"llm": chat, "prompt": PROMPT})

# Define the length evaluator function
from langsmith.schemas import Run, Example


# Function to interact with the locally deployed vLLM model
def my_app_1(question):
    response = requests.post(
        "http://localhost:8110/v1/chat/completions",
        json={
            "model": "qwen",
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": "Respond to the user's question in a short, concise manner (one short sentence)."
                },
                {
                    "role": "user",
                    "content": question,
                }
            ],
        }
    )
    response_json = response.json()
    return response_json["choices"][0]["message"]["content"]

# Function to wrap the local model interaction for evaluation
def langsmith_app_1(inputs):
    output = my_app_1(inputs["question"])
    return {"output": output}

# Evaluate the model using LangSmith
from langsmith.evaluation import evaluate

experiment_results = evaluate(
    langsmith_app_1, 
    data="QA Example Dataset", 
    evaluators=[qa_evaluator], 
    experiment_prefix="Qwen2", 
)

