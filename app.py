
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import os
import streamlit as st

# Access the GROQ API key from Streamlit Secrets
groq_api_key = st.secrets["GROQ_API_KEY"]


def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Use the API key in your code
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=groq_api_key)
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
    
    template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, SQL query, and SQL response, write a natural language response.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0, api_key=groq_api_key)
    
    query = sql_chain.invoke({"question": user_query, "chat_history": chat_history})
    
    chain = (
        RunnablePassthrough.assign(
            schema=lambda _: db.get_table_info(),
            response=lambda _: db.run(query),
            query=lambda _: query,
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm a SQL assistant. Ask me anything about your database."),
    ]

st.title("Chat with MySQL")

# Sidebar
with st.sidebar:
    st.subheader("Settings")
    st.markdown('<style> div[data-testid="stSidebar"] { background: rgba(255, 255, 255, 0.8); } </style>', unsafe_allow_html=True)
    
    host = st.text_input("Host", placeholder="Enter database host (e.g., localhost)", key="Host")
    port = st.text_input("Port", placeholder="Enter port (default: 3306)", key="Port")
    user = st.text_input("User", placeholder="Enter username (e.g., root)", key="User")
    password = st.text_input("Password", placeholder="Enter password", type="password", key="Password")
    database = st.text_input("Database", placeholder="Enter database name",key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            try:
                db = init_database(user, password, host, port, database)
                st.session_state.db = db
                st.success("Connected to database!")
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