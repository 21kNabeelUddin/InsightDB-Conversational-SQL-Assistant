from pymongo import MongoClient
import json

# MongoDB connection
MONGO_URI = "your_mongodb_connection_string"
DATABASE_NAME = "sample_mflix"
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]

def execute_query(collection_name, query, projection=None):
    try:
        collection = db[collection_name]
        result = collection.find_one(query, projection)
        return result if result else "No data found."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def generate_query_with_llm(question):
    """
    Use the LLM to generate a MongoDB query from the user's question.
    :param question: User question (e.g., "What is the email of Robert Baratheon?").
    :return: A dictionary with "collection", "query", and "projection".
    """
    # Example LLM output (replace this with an actual LLM API call)
    llm_output = '''
    {
      "collection": "users",
      "query": {"name": "Robert Baratheon"},
      "projection": {"email": 1}
    }
    '''
    return json.loads(llm_output)

def handle_user_question(question):
    """
    Handle user questions by generating a query, executing it, and returning the result.
    :param question: User question (e.g., "What is the email of Robert Baratheon?").
    :return: Direct answer.
    """
    # Step 1: Generate the query using the LLM
    query_data = generate_query_with_llm(question)
    
    # Step 2: Execute the query
    result = execute_query(query_data["collection"], query_data["query"], query_data.get("projection"))
    
    # Step 3: Format the response
    if isinstance(result, dict):
        return f"The result is: {result}"
    else:
        return result

# Example usage
user_question = "What is the email of Robert Baratheon?"
response = handle_user_question(user_question)
print(response)  # Output: The email of Robert Baratheon is robert.baratheon@example.com.
