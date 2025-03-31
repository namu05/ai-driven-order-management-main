import random
from typing import Literal
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import END
import pandas as pd
from src.tools import cancel_order
from src.config import llm
from src.state import State

from langchain.document_loaders.pdf import PyPDFDirectoryLoader # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain_openai import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_chroma import Chroma # Importing Chroma vector store from Langchain
import shutil # Importing shutil module for high-level file operations
from langchain_community.document_loaders import PyPDFLoader
import chromadb

# Load data
inventory_df = pd.read_csv("data/inventory.csv")
customers_df = pd.read_csv("data/customers.csv")
# Directory to your pdf files:
DATA_PATH = "data/Company_FAQ.pdf"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""


# Convert to dictionaries
inventory = inventory_df.set_index("item_id").T.to_dict()
customers = customers_df.set_index("customer_id").T.to_dict()

# Add near the top of the file, after llm initialization
tools_2 = [cancel_order]
llm_with_tools_2 = llm.bind_tools(tools_2)


#Testing going for Assistant and Chroma DB
def load_documents():
    """
    Load PDF documents using PyPDFLoader.
    Returns:
    List of Document objects: Loaded PDF documents represented as Langchain
                                Document objects.
    """
    # Initialize the PDF loader
    pdf_loader = PyPDFLoader(DATA_PATH)
    # Load the document
    documents = pdf_loader.load()
    return documents

def split_text(documents: list[Document]):
  """
  Split the text content of the given list of Document objects into smaller chunks.
  Args:
    documents (list[Document]): List of Document objects containing text content to split.
  Returns:
    list[Document]: List of Document objects representing the split text chunks.
  """
  # Initialize text splitter with specified parameters
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, # Size of each chunk in characters
    chunk_overlap=100, # Overlap between consecutive chunks
    length_function=len, # Function to compute the length of the text
    add_start_index=True, # Flag to add start index to each chunk
  )

  # Split documents into smaller chunks using text splitter
  chunks = text_splitter.split_documents(documents)
 # print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

  #Print example of page content and metadata for a chunk
  #document = chunks[0]
  #print(document.page_content)
  #print(document.metadata)

  return chunks # Return the list of split text chunks


# Path to the directory to save Chroma database
CHROMA_PATH = "data/chroma"
def save_to_chroma(chunks: list[Document]):
  """
  Save the given list of Document objects to a Chroma database.
  Args:
  chunks (list[Document]): List of Document objects representing text chunks to save.
  Returns:
  None
  """

  # Clear out the existing database directory if it exists
  if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

  # Create a new Chroma database from the documents using OpenAI embeddings
  db = Chroma.from_documents(
    chunks,
    OpenAIEmbeddings(),
    persist_directory=CHROMA_PATH
  )

  # Persist the database to disk
  persist_client=chromadb.PersistentClient()
  #print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def generate_data_store():
  """
  Function to generate vector database in chroma from documents.
  """
  documents = load_documents() # Load documents from a source
  chunks = split_text(documents) # Split documents into manageable chunks
  save_to_chroma(chunks) # Save the processed data to a data store


def query_rag(state: MessagesState) -> Literal["tools_2", "end"]:
    """
    Query a Retrieval-Augmented Generation (RAG) system using Chroma database and OpenAI.
    Args:
        state (MessagesState): The state containing the query text.
    Returns:
        formatted_response (str): Formatted response including the generated text and sources.
        response_text (str): The generated response text.
    """
    # Extract query text from the state (assuming `state` has `query_text` attribute)
    query_text = str(state['messages'])

    # YOU MUST - Use the same embedding function as before
    embedding_function = OpenAIEmbeddings()

    # Prepare the database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Retrieving the context from the DB using similarity search
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    # Check if there are any matching results or if the relevance score is too low
    if len(results) == 0 or results[0][1] < 0.5:
        # Return a default response if no good results found
        return {
            "response_text": "Sorry, I couldn't find any relevant information."
            "end"
        }

    # Combine context from matching documents
    context_text = "\n\n - -\n\n".join([doc.page_content for doc, _score in results])
 
    # Create prompt template using context and query text
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Initialize OpenAI chat model
    model = ChatOpenAI()

    # Generate response text based on the prompt
    response_text = model.invoke(prompt).content
    print(response_text)
    
    return {
        "formatted_response": response_text
    }


def categorize_query(state: MessagesState) -> MessagesState:
    """Categorize user query into PlaceOrder or CancelOrder or Assistant"""
    prompt = ChatPromptTemplate.from_template(
        "Categorize user query into PlaceOrder or CancelOrder or Assistant"
        "Respond with either 'PlaceOrder', 'CancelOrder', 'Assistant' Query: {state}"
    )

    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"state": state}).content
    print(category)
    return {"query":state,"category": category}


def check_inventory(state: MessagesState) -> MessagesState:
    """Check if the requested item is in stock."""

    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']

    if not item_id or not quantity:
        return {"error": "Missing 'item_id' or 'quantity'."}

    if inventory.get(item_id, {}).get("stock", 0) >= quantity:
        print("IN STOCK")
        return {"status": "In Stock"}
    return {"query":state,"order_status": "Out of Stock"}

def compute_shipping(state: MessagesState) -> MessagesState:
    """Calculate shipping costs."""
    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']
    customer_id = llm.with_structured_output(method='json_mode').invoke(f'Extract customer_id from the following text in json format: {state}')['customer_id']
    location = customers[customer_id]['location']


    if not item_id or not quantity or not location:
        return {"error": "Missing 'item_id', 'quantity', or 'location'."}

    weight_per_item = inventory[item_id]["weight"]
    total_weight = weight_per_item * quantity
    rates = {"local": 5, "domestic": 10, "international": 20}
    cost = total_weight * rates.get(location, 10)
    print(cost,location)

    return {"query":state,"cost": f"${cost:.2f}"}

def process_payment(state: State) -> State:
    """Simulate payment processing."""
    cost = llm.with_structured_output(method='json_mode').invoke(f'Extract cost from the following text in json format: {state}')

    if not cost:
        return {"error": "Missing 'amount'."}
    print(f"PAYMENT PROCESSED: {cost} and order successfully placed!")
    payment_outcome = random.choice(["Success", "Failed"])
    return {"payment_status": payment_outcome}

def call_model_2(state: MessagesState):
    """Use the LLM to decide the next step."""
    messages = state["messages"]
    response = llm_with_tools_2.invoke(str(messages))
    return {"messages": [response]}

def call_tools_2(state: MessagesState) -> Literal["tools_2", "end"]:
    """Route workflow based on tool calls."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools_2"
    return "end"

def route_query_1(state: State) -> str:
    """Route the query based on its category."""
    print(state)
    if state["category"] == "PlaceOrder":
        return "PlaceOrder"
    if state["category"] == "Assistant":
        return "Assistant"
    if state["category"] == "CancelOrder":
        return "CancelOrder" 
