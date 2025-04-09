import os
from dotenv import load_dotenv
from src.workflow import create_workflow
from src.nodes import generate_data_store
from IPython.display import display, Image
from sqlalchemy.orm import Session
from fastapi import FastAPI, Depends
from src.database import get_session
from src.crud import fetch_inventory_data, fetch_customer_data, fetch_order_data

def main():
    # Load environment variables
    load_dotenv()
    
    # Create the workflow
    agent = create_workflow()
    # Generate the data store
    generate_data_store()

    app = FastAPI()

    @app.get("/inventory")
    def get_inventory(session: Session = Depends(get_session)):
        return fetch_inventory_data(session)

    @app.get("/customers")
    def get_customers(session: Session = Depends(get_session)):
        return fetch_customer_data(session)

    @app.get("/orders")
    def get_orders(session: Session = Depends(get_session)):
        return fetch_order_data(session)
        
# Test cancel order
    print("\nTesting Cancel Order:")
    user_query = "I wish to cancel order_id 4ed4146c-fd2f-4593-9c49-27f33fe2dba6"
    for chunk in agent.stream(
        {"messages": [("user", user_query)]},
        stream_mode="values",
    ):
        chunk["messages"][-1].pretty_print()
    
#Test place order
        # print("\nTesting Place Order:")
        # user_query = "David Brown : I wish to place order for item_28 with order quantity as 4"
        # for chunk in agent.stream(
        #     {"messages": [("user", user_query)]},
        #     stream_mode="values",
        # ):
        #     chunk["messages"][-1].pretty_print()

#Test assitant        
        # user_query = "How cam i contact customer support?"
        # for chunk in agent.stream(
        #     {"messages": [("user", user_query)]},
        #     stream_mode="values",
        # ):
        #     chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    main() 