import os
from dotenv import load_dotenv
from src.workflow import create_workflow
from src.nodes import generate_data_store
from IPython.display import display, Image

def main():
    # Load environment variables
    load_dotenv()
    
    # Create the workflow
    agent = create_workflow()
    # Generate the data store
    generate_data_store()
    
    # Test cancel order
    print("\nTesting Cancel Order:")
    user_query = "I wish to cancel order_id 6bd358a9-a6e0-4cfe-9e85-3aac3fde1778"
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
    # user_query = "What is my shipment time for order?"
    # for chunk in agent.stream(
    #     {"messages": [("user", user_query)]},
    #     stream_mode="values",
    # ):
    #     chunk["messages"][-1].pretty_print()

if __name__ == "__main__":
    main() 