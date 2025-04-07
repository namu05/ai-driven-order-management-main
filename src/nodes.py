import random
from typing import Literal
import os
import uuid 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.graph import END
import pandas as pd
from src.tools import cancel_order
from src.config import llm
from src.state import State

from langchain.text_splitter import RecursiveCharacterTextSplitter # Importing text splitter from Langchain
from langchain_openai import OpenAIEmbeddings # Importing OpenAI embeddings from Langchain
from langchain.schema import Document # Importing Document schema from Langchain
from langchain_chroma import Chroma # Importing Chroma vector store from Langchain
import shutil # Importing shutil module for high-level file operations
from langchain_community.document_loaders import PyPDFLoader
import chromadb

from sqlalchemy import create_engine, Column, Integer, String, Float, VARCHAR
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

# Database connection setup (replace with your credentials)
DATABASE_URL = "mysql+mysqlconnector://root:@localhost:3306/orders"

# Set up the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# # Base class for declarative models
Base = declarative_base()

# Define the Inventory model
class Inventory(Base):
    __tablename__ = "inventory"
    
    # Define the columns
    item_id = Column(Integer, primary_key=True) 
    category = Column(String)  
    stock = Column(Integer) 
    weight = Column(Float)  
    price = Column(Float) 

# Define the Customer model
class Customer(Base):
    __tablename__ = "customers"
    
    # Define the columns
    customer_id = Column(Integer, primary_key=True) 
    name = Column(String) 
    location = Column(String)

class Orders(Base):
    __tablename__ = "orders"
    
    # Define the columns with appropriate types
    customer_id = Column(VARCHAR(255))  # Assuming customer_id is a string (e.g., UUID or name)
    order_id = Column(VARCHAR(255), primary_key=True)  # Unique order_id as an integer
    order_name = Column(String)  # Assuming this is a string, use a length for VARCHAR
    item_id = Column(VARCHAR(255))  # Assuming item_id is an integer
    quantity = Column(Integer)  # Quantity of the item
    order_price = Column(VARCHAR(255)) 
    
# Create a session to interact with the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
session = SessionLocal()

# Function to fetch inventory data from the database
def fetch_inventory_data():
    try:
        # Query the inventory table
        inventory_data = session.query(Inventory).all()
        inventory_dict = {item.item_id: item for item in inventory_data}
        return inventory_dict
    except SQLAlchemyError as e:
        print(f"Error fetching inventory: {e}")
        return {}

# Function to fetch customer data from the database
def fetch_customer_data():
    try:
        # Query the customer table
        customer_data = session.query(Customer).all()
        customer_dict = {customer.customer_id: customer for customer in customer_data}
        return customer_dict
    except SQLAlchemyError as e:
        print(f"Error fetching customers: {e}")
        return {}

# Function to fetch customer data from the database
def fetch_order_data():
    try:
        # Query the customer table
        order_data = session.query(Orders).all()
        order_dict = {order.order_id: order for order in order_data}
        return order_dict
    except SQLAlchemyError as e:
        print(f"Error fetching Orders: {e}")
        return {}
    
# Initialize the inventory and customers data
inventory = fetch_inventory_data()
customers = fetch_customer_data()
orders = fetch_order_data()
# Directory to your pdf files:
DATA_PATH = "data/Company_FAQ.pdf"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:
{context}
 - -
Answer the question based on the above context: {question}
"""

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


def query_rag(state: MessagesState) -> MessagesState:
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
    if len(results) == 0 or results[0][1] < 0.7:
        # Return a default response if no good results found
        return {
            "response": "Sorry, I couldn't find any relevant information."
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
        "response": response_text
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

    # Extract item_id and quantity from the state using structured output
    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']

    if not item_id or not quantity:
        return {"error": "Missing 'item_id' or 'quantity'."}

    # Query the inventory to get the item by item_id
    item = session.query(Inventory).filter_by(item_id=item_id).first()

    # Check if the item exists and if the stock is sufficient
    if item and item.stock >= quantity:
        print("IN STOCK")
        return {"order_status": "In Stock", "query": state}
    
    return {"order_status": "Out of Stock", "query": state}

def compute_shipping(state: MessagesState) -> MessagesState:
    """Calculate shipping costs."""
    
    # Extract item_id, quantity, and customer_id from the state using structured output
    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']
    name = llm.with_structured_output(method='json_mode').invoke(f'Extract customer_name from the following text in json format: {state}')['customer_name']
    
    # Look up customer using the extracted customer_id
    customer = session.query(Customer).filter_by(name=name).first()

    location = customer.location

    if not item_id or not quantity or not location:
        return {"error": "Missing 'item_id', 'quantity', or 'location'."}

    # Fetch item from inventory using item_id
    item = session.query(Inventory).filter_by(item_id=item_id).first()
    
    if not item:
        return {"error": f"Item with item_id {item_id} not found."}

    weight_per_item = item.weight
    total_weight = weight_per_item * quantity
    
    # Shipping cost rates based on location
    rates = {"local": 5, "domestic": 10, "international": 20}
    cost = total_weight * rates.get(location, 10)  # Default rate is 10 if the location is not recognized
    
    print(cost,location)

    return {"query": state, "cost": f"${cost:.2f}"}

def process_payment(state: State) -> State:
    """Simulate payment processing and update inventory."""
    
    # Extract cost, item_id, quantity from the state
    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']
    cost = llm.with_structured_output(method='json_mode').invoke(f'Extract cost from the following text in json format: {state}')['cost']
    name = llm.with_structured_output(method='json_mode').invoke(f'Extract customer_name from the following text in json format: {state}')['customer_name']
 
    
    # Ensure that item_id, quantity, and cost are provided
    if not item_id or not quantity or not cost:
        return {"error": "Missing 'item_id', 'quantity', or 'cost'."}

    # Fetch the item from the inventory
    item = session.query(Inventory).filter_by(item_id=item_id).first()

    if not item:
        return {"error": f"Item with item_id {item_id} not found."}

    # Check if there is enough stock
    if item.stock < quantity:
        return {"error": f"Not enough stock for item {item_id}. Available stock: {item.stock}"}
    
    # Fetch customer by name (assuming customer name is unique)
    customer = session.query(Customer).filter_by(name=name).first()

    # Process payment (simulated)
    payment_outcome = random.choice(["Success", "Failed"])

    if payment_outcome == "Success":
        # Update inventory: Reduce stock by the ordered quantity
        item.stock -= quantity
        session.commit()  # Commit the changes to the database

        unique_order_id = str(uuid.uuid4()) 

        # Create and insert the new order into the Orders table
        new_order = Orders(
            customer_id=customer.customer_id,  # Use the customer's ID
            order_id= unique_order_id,
            order_name=item.category,  # Order placed by the customer
            item_id=item_id,
            quantity=quantity,
            order_price=cost  # Use the cost of the order
        )
        
        # Add the new order to the session and commit
        session.add(new_order)
        session.commit()

        print(f"Inventory updated: Reduced stock of item {item_id} by {quantity}. New stock: {item.stock}")
        print(f"PAYMENT PROCESSED: {cost} and order successfully placed!")

        return {"payment_status": "Success", "order_status": "Order successfully placed!"}

    print(f"PAYMENT FAILED: {cost} and order not placed!")
    return {"payment_status": "Failed", "order_status": "Payment failed, order not placed."}


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
    if state["category"] == "PlaceOrder":
        return "PlaceOrder"
    if state["category"] == "Assistant":
        return "Assistant"
    if state["category"] == "CancelOrder":
        return "CancelOrder" 
