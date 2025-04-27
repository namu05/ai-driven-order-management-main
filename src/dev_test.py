from src.workflow import create_workflow

agent = create_workflow()

print("\nTesting Place Order:")
user_query = "David Brown : I wish to place order for item_28 with order quantity as 4"
for chunk in agent.stream({"messages": [("user", user_query)]}, stream_mode="values"):
    chunk["messages"][-1].pretty_print()

# Test cancel order
    # print("\nTesting Cancel Order:")
    # user_query = "I wish to cancel order_id 4ed4146c-fd2f-4593-9c49-27f33fe2dba6"
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
