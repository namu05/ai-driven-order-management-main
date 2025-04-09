from sqlalchemy import Column, Integer, String, Float, VARCHAR
from src.database import Base  # Import the Base from the database.py file

# Define the Inventory model
class Inventory(Base):
    __tablename__ = "inventory"
    
    item_id = Column(Integer, primary_key=True) 
    category = Column(String)  
    stock = Column(Integer) 
    weight = Column(Float)  
    price = Column(Float) 

# Define the Customer model
class Customer(Base):
    __tablename__ = "customers"
    
    customer_id = Column(Integer, primary_key=True) 
    name = Column(String) 
    location = Column(String)

# Define the Orders model
class Orders(Base):
    __tablename__ = "orders"
    
    customer_id = Column(VARCHAR(255)) 
    order_id = Column(VARCHAR(255), primary_key=True)  
    order_name = Column(String)  
    item_id = Column(VARCHAR(255))  
    quantity = Column(Integer)  
    order_price = Column(VARCHAR(255)) 
