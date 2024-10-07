import os
import streamlit as st
from mem0 import MemoryClient
from openai import AzureOpenAI

# Set up your Azure OpenAI credentials
st.title("LLM App with Memory 🧠")
st.caption("LLM App with personalized memory layer that remembers ever user's choice and interests")

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2023-05-15",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# Initialize Mem0 with Qdrant
mem0Client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])

user_id = st.text_input("Enter your Username")
print("User ID: ", user_id)

prompt = st.text_input("Ask ChatGPT")
print("Prompt: ", prompt)

if st.button('Chat with LLM'):
    with st.spinner('Searching...'):
        if not prompt:
            prompt = st.text_input("Ask ChatGPT")
            print("Prompt: ", prompt)
        relevant_memories = mem0Client.search(query=prompt, user_id=user_id)
        # Prepare context with relevant memories
        context = "Relevant past information:\n"

        for mem in relevant_memories:
            context += f"- {mem['text']}\n"
            
        # Prepare the full prompt
        full_prompt = f"{context}\nHuman: {prompt}\nAI:"

        # Get response from GPT-4
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant with access to past conversations."},
                {"role": "user", "content": full_prompt}
            ]
        )
        
        answer = response.choices[0].message.content

        st.write("Answer: ", answer)

        # Add AI response to memory
        mem0Client.add(answer, user_id=user_id)


# Sidebar option to show memory
st.sidebar.title("Memory Info")
if st.sidebar.button("View Memory Info"):
    print("View User ID Memory: ", user_id)
    memories = mem0Client.get_all(user_id=user_id)
    if memories:
        st.sidebar.write(f"You are viewing memory for user **{user_id}**")
        for mem in memories:
            st.sidebar.write(f"- {mem['text']}")
    else:
        st.sidebar.info("No learning history found for this user ID.")