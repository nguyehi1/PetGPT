def structured_response_template(question, retrieved_info=""):
    return f"""
    You are a precise assistant that always maintains clear formatting and structure in responses. 

QUESTION: "{question}"

CONTEXT INFORMATION:
{retrieved_info}

"""

def generate_system_message():
    return """You are an AI assistant that always maintains precise formatting in responses."""

# Example usage with system message
def create_rag_prompt(question, context):
    system_message = generate_system_message()
    user_prompt = structured_response_template(question, context)
    
    return {
        "system": system_message,
        "user": user_prompt
    }