import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize shared LLM instance
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)