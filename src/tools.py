from langchain_core.tools import tool
from src.config import llm
from langchain.prompts import ChatPromptTemplate
from sqlalchemy.exc import SQLAlchemyError
from src.models import Inventory, Customer, Orders
from src.database import get_session

@tool
def cancel_order(query: str) -> dict:
    """Simulate order cancelling using the LLM"""
    
    # Construct prompt to extract the order_id
    prompt = ChatPromptTemplate.from_template("""
        Extract the order_id from this text. 
        Text: {text}
        Rules:
        - The order_id should be a string.
        - The order_id may appear after 'order_id' or 'order'.
        - Return the order_id without any additional text.
    """)
    
    # Send the query to the LLM and extract the order_id
    result = llm.invoke(prompt.format(text=query)).content
    
    # If no order_id found, return an error
    if not result:
        return {"error": "No order_id found in the query."}
    
    # Use session with context manager
    with next(get_session()) as session:
    
        # Query the Orders table to get the order based on the order_id
        order = session.query(Orders).filter_by(order_id=result).first()

        if not order:
            return {"error": f"Order with order_id {result} not found."}

        # Get the item from the inventory using item_id from the order
        item = session.query(Inventory).filter_by(item_id=order.item_id).first()

        if not item:
            return {"error": f"Item with item_id {order.item_id} not found in inventory."}

        # Update the inventory: increase the stock by the quantity of the cancelled order
        item.stock += order.quantity

        # Now, delete the order from the Orders table
        try:
            session.delete(order)  # Delete the order from the table
            session.commit()  # Commit the deletion

            # Commit the changes to the inventory
            session.commit()

            print(f"Order {result} has been cancelled.")
            print(f"Inventory updated: Item {order.item_id} stock increased by {order.quantity}. New stock: {item.stock}")
            
            return {"order_status": "Order stands cancelled", "order_id": result}

        except SQLAlchemyError as e:
            session.rollback()  # Rollback the transaction in case of error
            print(f"Error occurred: {str(e)}")
            return {"error": f"Failed to cancel the order. Error: {str(e)}"}