from sqlalchemy.orm import Session
from src.models import Inventory, Customer, Orders

# Function to fetch inventory data
def fetch_inventory_data(session: Session):
    try:
        inventory_data = session.query(Inventory).all()
        inventory_dict = {item.item_id: item for item in inventory_data}
        return inventory_dict
    except Exception as e:
        print(f"Error fetching inventory: {e}")
        return {}

# Function to fetch customer data
def fetch_customer_data(session: Session):
    try:
        customer_data = session.query(Customer).all()
        customer_dict = {customer.customer_id: customer for customer in customer_data}
        return customer_dict
    except Exception as e:
        print(f"Error fetching customers: {e}")
        return {}

# Function to fetch order data
def fetch_order_data(session: Session):
    try:
        order_data = session.query(Orders).all()
        order_dict = {order.order_id: order for order in order_data}
        return order_dict
    except Exception as e:
        print(f"Error fetching Orders: {e}")
        return {}
