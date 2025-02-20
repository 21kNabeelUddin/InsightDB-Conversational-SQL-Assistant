from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from pymongo import MongoClient
import os
import streamlit as st

# Access the GROQ API key from environment variable
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("GROQ_API_KEY environment variable is not set. Please set it and restart the app.")
    st.stop()

# Access the MongoDB connection string from Streamlit Secrets
mongodb_connection_string = st.secrets["MONGODB_CONNECTION_STRING"]

# Initialize MongoDB connection
def init_mongodb(connection_string: str, db_name: str):
    try:
        client = MongoClient(connection_string)
        db = client[db_name]
        return db
    except Exception as e:
        st.error(f"Failed to connect to MongoDB: {e}")
        return None

# Function to generate a MongoDB query
def get_mongodb_query(user_query: str, chat_history: list):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's MongoDB database.
    Based on the collection schema below, write a MongoDB query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the MongoDB query and nothing else. Do not wrap the query in any other text, not even backticks.
    
    Question: {question}
    MongoDB Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=groq_api_key)
    
    # Schema for the sample_mflix database
    schema = """
    Collections in sample_mflix:
    - comments: { name: string, email: string, movie_id: ObjectId, text: string, date: datetime }
    - movies: { title: string, year: int, cast: list, genres: list }
    - theaters: { theaterId: int, location: { address: string, city: string, state: string } }
    - users: { name: string, email: string, password: string }
    """
    
    return (
        RunnablePassthrough.assign(schema=lambda _: schema)
        | prompt
        | llm
        | StrOutputParser()
    )

# Function to get a response from MongoDB
def get_response(user_query: str, db, chat_history: list):
    query_chain = get_mongodb_query(user_query, chat_history)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's MongoDB database.
    Based on the collection schema below, question, MongoDB query, and query response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    MongoDB Query: <QUERY>{query}</QUERY>
    User question: {question}
    Query Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=groq_api_key)
    
    query = query_chain.invoke({"question": user_query, "chat_history": chat_history})
    print(f"Generated MongoDB Query: {query}")  # Debug statement
    
    try:
        # Determine which collection to query based on the user's question
        if "comments" in user_query.lower():
            collection = db["comments"]
        elif "movies" in user_query.lower():
            collection = db["movies"]
        elif "theaters" in user_query.lower():
            collection = db["theaters"]
        elif "users" in user_query.lower():
            collection = db["users"]
        else:
            return "Please specify a collection (e.g., comments, movies, theaters, users)."
        
        # Execute the MongoDB query
        result = list(collection.find(eval(query)))  # Evaluate the query string as a Python expression
        response = f"Query result: {result}"
    except Exception as e:
        response = f"Error executing MongoDB query: {e}"
    
    chain = (
        RunnablePassthrough.assign(
            schema=lambda _: "Collections in sample_mflix",  # Replace with actual schema
            response=lambda _: response,
            query=lambda _: query,  # Wrap query in a lambda function
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

# Streamlit App
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a MongoDB assistant. Ask me anything about the sample_mflix database."),
    ]

st.title("Chat with MongoDB")

# Sidebar for MongoDB connection
with st.sidebar:
    st.subheader("MongoDB Connection")
    st.markdown('<style> div[data-testid="stSidebar"] { background: rgba(255, 255, 255, 0.8); } </style>', unsafe_allow_html=True)
    
    db_name = st.text_input("Database Name", value="sample_mflix", key="Database")  # Default to sample_mflix
    
    if st.button("Connect"):
        with st.spinner("Connecting to MongoDB..."):
            try:
                db = init_mongodb(mongodb_connection_string, db_name)
                st.session_state.db = db
                st.success("Connected to MongoDB!")
            except Exception as e:
                st.error(f"Connection failed: {e}")

# Chat History Display
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# User Input
user_query = st.chat_input("Type a message...")
if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)
        
    with st.chat_message("AI"):
        if "db" in st.session_state:
            response = get_response(user_query, st.session_state.db, st.session_state.chat_history)
            st.markdown(response)
            st.session_state.chat_history.append(AIMessage(content=response))
        else:
            st.markdown("❌ Please connect to a database first.")
