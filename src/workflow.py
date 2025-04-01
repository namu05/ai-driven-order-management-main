from src.nodes import (
    categorize_query, check_inventory, compute_shipping,
    process_payment, call_model_2, call_tools_2, route_query_1, 
    query_rag
)
from src.tools import cancel_order
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from IPython.display import display, Image

def create_workflow():
    # Create the workflow
    workflow = StateGraph(MessagesState)
    
    # Add nodes
    workflow.add_node("RouteQuery", categorize_query)
    workflow.add_node("CheckInventory", check_inventory)
    workflow.add_node("ComputeShipping", compute_shipping)
    workflow.add_node("ProcessPayment", process_payment)
    
    # Add tool nodes
    tools_2 = [cancel_order]
    tool_node_2 = ToolNode(tools_2)
    
    workflow.add_conditional_edges(
        "RouteQuery",
        route_query_1,
        {
            "PlaceOrder": "CheckInventory",
            "CancelOrder": "CancelOrder",
            "Assistant": "Assistant"
        }
    )
    workflow.add_node("CancelOrder", call_model_2)
    workflow.add_node("tools_2", tool_node_2)

    #Testing going for assistant
    workflow.add_node("Assistant",query_rag)
    
    # Define edges
    workflow.add_edge(START, "RouteQuery")
    workflow.add_edge("Assistant", END)    
    workflow.add_edge("CheckInventory", "ComputeShipping")
    workflow.add_edge("ComputeShipping", "ProcessPayment")
    workflow.add_conditional_edges(
        "CancelOrder",
        call_tools_2,
        {
            "tools_2": "tools_2",
            "end": END
        }
    )
    workflow.add_edge("tools_2", "CancelOrder")
    workflow.add_edge("ProcessPayment", END)

    graph = workflow.compile() 

    # Generate the image
    image_object = Image(graph.get_graph(xray=True).draw_mermaid_png())

    # Get the image data
    image_data = image_object.data

    # Save the image data to a file
    with open("langgraph_image.png", "wb") as f:
        f.write(image_data)
    
    return graph