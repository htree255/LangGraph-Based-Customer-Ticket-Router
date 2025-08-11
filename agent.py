from dotenv import load_dotenv

# Imports for Graph and state
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END

# Imports for LLM and prompts
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Load environmant variables
load_dotenv()

# Define the graph state
class TicketState(TypedDict):
    """
    Represents the state of the customer support ticket as it moves through the graph.

    """
    query: str
    category: Literal[
        "Technical Support",
        "Billing Inquiry",
        "Product Feedback",
        "General Question",
        "Unclear"
    ]
    response: str

# Define the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

# Define the prompt for the classification tasks
classification_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant specialized in classifying customer support tickets.
         Your goal is to accurately categorize the user's query into one of the following categories:
         'Technical Support', 'Billing Inquiry', 'Product Feedback', 'General Question'.
         If the query does not fit clearly into any of these, classify it as 'Unclear'.
         Respond ONLY with the exact category name and nothing else.""",
        ),
        ("human", "Customer query: {query}"),
    ]
)

# Define prompts for each response type
technical_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and concise customer support agent for technical issues.",
        ),
        ("human", "Customer query: {query}"),
    ]
)

billing_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and concise customer support agent for billing inquiries.",
        ),
        ("human", "Customer query: {query}"),
    ]
)

feedback_response_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a friendly and concise customer support agent."),
        ("human", "Customer query: {query}"),
    ]
)

general_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful and concise customer support agent for general questions.",
        ),
        ("human", "Customer query: {query}"),
    ]
)

unclear_response_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a customer support agent. The user's query was unclear. Politely ask them to rephrase.",
        ),
        ("human", "Customer query: {query}"),
    ]
)


# Define the nodes

# Classification node
def classify_ticket(state: TicketState) -> TicketState:
    """Classifies the incoming customer query using the Gemini LLM."""
    print("\n---CLASSIFYING TICKET---")
    query = state["query"]
    classification_chain = classification_prompt | llm
    predicted_category = classification_chain.invoke({"query": query}).content.strip()

    valid_categories = {
            "Technical Support",
            "Billing Inquiry",
            "Product Feedback",
            "General Question",
            "Unclear"
        }
    if predicted_category not in valid_categories:
         print(
            f"WARNING: LLM returned an unexpected category: '{predicted_category}'. Defaulting to 'Unclear'."
        )
         predicted_category = "Unclear"
    print(f"Ticket classified as: {predicted_category}")
    return {"category": predicted_category} 

# Handler nodes
def handle_technical(state: TicketState) -> TicketState:
    """Generates a response for a 'Technical Support' ticket."""
    print("\n---HANDLING TECHNICAL ISSUE---")
    response_chain = technical_response_prompt | llm
    generated_response = response_chain.invoke({"query": state["query"]}).content
    print(f"Technical Response:  {generated_response}")
    return {"response": generated_response}

def handle_billing(state: TicketState) -> TicketState:
    """Generates a response for a 'Billing Inquiry' ticket."""
    print("\n---HANDLING BILLING INQUIRY---")
    response_chain = billing_response_prompt | llm
    generated_response = response_chain.invoke({"query": state["query"]}).content
    print(f"Billing Response: {generated_response}")
    return {"response": generated_response}


def handle_feedback(state: TicketState) -> TicketState:
    """Generates a response for a 'Product Feedback' ticket."""
    print("\n---HANDLING PRODUCT FEEDBACK---")
    response_chain = feedback_response_prompt | llm
    generated_response = response_chain.invoke({"query": state["query"]}).content
    print(f"Feedback Response: {generated_response}")
    return {"response": generated_response}


def handle_general(state: TicketState) -> TicketState:
    """Generates a response for a 'General Question' ticket."""
    print("\n---HANDLING GENERAL QUESTION---")
    response_chain = general_response_prompt | llm
    generated_response = response_chain.invoke({"query": state["query"]}).content
    print(f"General Response: {generated_response}")
    return {"response": generated_response}


def handle_unclear(state: TicketState) -> TicketState:
    """Generates a response for an 'Unclear' ticket."""
    print("\n---HANDLING UNCLEAR TICKET---")
    response_chain = unclear_response_prompt | llm
    generated_response = response_chain.invoke({"query": state["query"]}).content
    print(f"Unclear Response: {generated_response}")
    return {"response": generated_response}

# --Define routing logic --

# The router function for our conditional edges.
def route_ticket(state: TicketState) -> str:
    """Routes the ticket to the appropriate handler node based on its classified category."""
    print("---ROUTING TICKET---")
    category = state["category"]
    print(f"Routing based on category: {category}")

    if category == "Technical Support":
        return "handle_technical"
    elif category == "Billing Inquiry":
        return "handle_billing"
    elif category == "Product Feedback":
        return "handle_feedback"
    elif category == "General Question":
        return "handle_general"
    else:
        return "handle_unclear"


#create langgraph - put it all together 
def build_agent_graph():
    # Initialize the graph with our TicketState.
    workflow = StateGraph(TicketState)

    # Add all our functions as nodes in the graph.
    workflow.add_node("classify_ticket", classify_ticket)
    workflow.add_node("handle_technical", handle_technical)
    workflow.add_node("handle_billing", handle_billing)
    workflow.add_node("handle_feedback", handle_feedback)
    workflow.add_node("handle_general", handle_general)
    workflow.add_node("handle_unclear", handle_unclear)

    # Set the entry point. All tickets start here.
    workflow.set_entry_point("classify_ticket")

    # Add conditional edges from the classification node.
    # This is where the routing logic runs.
    workflow.add_conditional_edges(
        "classify_ticket",
        route_ticket,
        
    )

    # After each handler, the process is complete.
    workflow.add_edge("handle_technical", END)
    workflow.add_edge("handle_billing", END)
    workflow.add_edge("handle_feedback", END)
    workflow.add_edge("handle_general", END)
    workflow.add_edge("handle_unclear", END)

    # Compile the graph into a runnable application.
    app = workflow.compile()
    return app

#run the agent with test queries
if __name__ == "__main__":
    app = build_agent_graph()
    print("\n--- Running Customer Support Agent ---")

    test_queries = [
        "My Wi-Fi keeps disconnecting, how can I fix this?",
        "I see an unfamiliar charge of $100 on my last statement.",
        "I love the new dark mode feature in your app!",
        "What payment methods do you accept?",
        "I need help with... something. It's not working.",
    ]

    for query in test_queries:
        print(f"\nProcessing: '{query}'")
        initial_state = {"query": query, "category": "Unclear", "response": ""}
        final_state = app.invoke(initial_state)
        print(f"\nFinal State: {final_state}")
        print("=" * 60)

    print("\nAgent execution finished.")

