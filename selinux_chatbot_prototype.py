import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from qgenie.integrations.langchain import QGenieEmbeddings, QGenieChat
from dotenv import load_dotenv
import os
import yaml
import numpy as np
import re
# Load environment variables
load_dotenv()
api_key = os.getenv("QGENIE_API_KEY")

# Check API key
if not api_key:
    st.error("QGenie API key not found. Please check your .env file.")
    st.stop()

# Load Chroma vector store
vectorstore = Chroma(
    persist_directory="./sepolcy_db/sepolcy_db",
    embedding_function=QGenieEmbeddings()
)

retriever = vectorstore.as_retriever()
llm = QGenieChat(model="qwen2.5-coder-32B-128k", temperature=0)

# Create Conversational QA chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Load prompt library
def load_prompt_library(path="selinux_prompts_all2.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)["prompts"]

prompt_library = load_prompt_library()


# Initialize embedding model
embedding_model = QGenieEmbeddings()

# Precompute embeddings for all prompt descriptions or templates
def embed_prompts(prompt_library):
    prompt_texts = [prompt["description"] for prompt in prompt_library]
    embeddings = embedding_model.embed_documents(prompt_texts)
    return list(zip(prompt_library, embeddings))

embedded_prompts = embed_prompts(prompt_library)

# Semantic match function
def semantic_match_prompt(query):
    query_embedding = embedding_model.embed_query(query)

    # Compute cosine similarity
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    best_match = None
    best_score = -1

    for prompt, embedding in embedded_prompts:
        score = cosine_similarity(query_embedding, embedding)
        if score > best_score:
            best_score = score
            best_match = prompt

    # Optional: set a threshold to avoid poor matches
    if best_score > 0.7:
        return best_match
    else:
        return None

# Match user query to a prompt
def match_prompt(query):
    for prompt in prompt_library:
        for tag in prompt["tags"]:
            if tag in query.lower():
                return prompt
    return None

def fill_prompt(template, context):
    # Replace all {{key}} placeholders with corresponding values from context
    for key, value in context.items():
        placeholder = f"{{{{{key}}}}}"  # creates {{key}}
        template = template.replace(placeholder, value)

    # Optionally warn if any placeholders are left unreplaced
    unreplaced = re.findall(r"\{\{.*?\}\}", template)
    if unreplaced:
        print(f"Warning: Unreplaced placeholders found: {unreplaced}")

    return template

# Streamlit UI setup
st.set_page_config(page_title="SELinux Chatbot", layout="wide")
st.title("🔐 SELinux Policy Chatbot")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "followups" not in st.session_state:
    st.session_state.followups = []

# Input box
user_input = st.chat_input("Ask me anything about SELinux policies:")

# Function to decide whether to use history
def should_use_history(query):
    keywords = ["refer", "above", "previous", "earlier", "context", "same thread"]
    return any(kw in query.lower() for kw in keywords)

# Function to handle a query and update history
def handle_query(query):
    use_history = should_use_history(query)
    history = st.session_state.chat_history if use_history else []

#    matched_prompt = match_prompt(query)
    matched_prompt = semantic_match_prompt(query)
    if matched_prompt:
        # For demo purposes, assume audit_log is the full query
        filled_prompt = fill_prompt(matched_prompt["template"], {"audit_log": query})
        response = qa_chain({
            "question": filled_prompt,
            "chat_history": history
        })
    else:
        response = qa_chain({
            "question": query,
            "chat_history": history
        })

    answer = response["answer"]
    sources = response["source_documents"]

    # Update chat history
    st.session_state.chat_history.append((query, answer))

    # Update follow-up suggestions
    if "SELinux" in query or "policy" in query.lower():
        st.session_state.followups = [
            "What are SELinux types?",
            "How do I write a custom SELinux policy?",
            "What is the difference between enforcing and permissive mode?"
        ]
    else:
        st.session_state.followups = []

    # Show retrieved chunks
    with st.expander("🔍 Retrieved Chunks"):
        for doc in sources:
            st.markdown(doc.page_content)

# Handle user input
if user_input:
    handle_query(user_input)

# Display chat history
for user_q, bot_a in st.session_state.chat_history:
    with st.chat_message("user"):
        st.markdown(user_q)
    with st.chat_message("assistant"):
        st.markdown(bot_a)

# Show follow-up suggestions
if st.session_state.followups:
    st.markdown("**💡 You can also ask:**")
    for suggestion in st.session_state.followups:
        if st.button(suggestion):
            handle_query(suggestion)

# Optional: Reset chat
if st.button("🔄 Reset Chat"):
    st.session_state.chat_history = []

