import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from langserve import add_routes

from src.workflow import create_workflow
from src.nodes import generate_data_store
from src.database import get_session
from src.crud import fetch_inventory_data, fetch_customer_data, fetch_order_data

# Load environment variables
load_dotenv()

# Generate vector DB for RAG (do this once at startup)
generate_data_store()

# Define FastAPI app
app = FastAPI()

# Add LangGraph agent to the API
agent = create_workflow()
add_routes(app, agent, path="/workflow")

# Add extra endpoints
@app.get("/inventory")
def get_inventory(session: Session = Depends(get_session)):
    return fetch_inventory_data(session)

@app.get("/customers")
def get_customers(session: Session = Depends(get_session)):
    return fetch_customer_data(session)

@app.get("/orders")
def get_orders(session: Session = Depends(get_session)):
    return fetch_order_data(session)
