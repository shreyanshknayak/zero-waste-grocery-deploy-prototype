# langgraph_app.py
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from groq import Groq
import os
from typing import TypedDict, List, Tuple



from langgraph.graph import END, START, MessagesState, StateGraph

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_env = os.getenv("PINECONE_ENV") 
pinecone_host = os.getenv("PINECONE_HOST")
pinecone_index_name = "recipe-index"


from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import pandas as pd




class State(TypedDict):
    query: str
    docs: list
    recipe: str
    result: str
    chat_history: List[Tuple[str, str]]
    step_index: int
    continue_chat: bool  

client = Groq(api_key=api_key)

def build_vectorstore_from_csv(csv_path="recipes.csv"):
    # Step 1: Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key, environment=pinecone_env)


    # Step 2: Create or connect to the index

    if pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name= pinecone_index_name,
            dimension=1024, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region=pinecone_env
            )
        )

    '''if pinecone_index_name not in pinecone.list_indexes():
        pinecone.create_index(pinecone_index_name, dimension=384)  # 384 for MiniLM-L6-v2
    index = pinecone.Index(pinecone_index_name)'''

    index = pc.Index(host = pinecone_host)

    # Step 3: Load top 70 entries, Use only once to upsert entries into the pinecone vectorstore
    '''df = pd.read_csv(csv_path, nrows=70)
    
    # Step 4: Create documents
    docs = [
        Document(
            page_content=f"Title: {row['title']}\nIngredients: {row['ingredients']}\nDirections: {row['directions']}"
        )
        for _, row in df.iterrows()
    ]'''

    # Step 5: Embed texts
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large")  #e5-large --> 1024 dimensions, cosine friendly
    '''texts = [doc.page_content for doc in docs]
    vectors = embeddings.embed_documents(texts)

    # Step 6: Upsert into Pinecone
    pinecone_vectors = [
        (f"recipe-{i}", vec, {"text": texts[i]})
        for i, vec in enumerate(vectors)
    ]
    index.upsert(vectors=pinecone_vectors)
    print("✅ Upserted top 70 recipes to Pinecone")'''

    # Step 7: Return LangChain retriever
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    #vectorstore = Pinecone(index, embeddings.embed_query, "text")
    return vector_store

# Build Vector Store
def build_vectorstore():
    loader = TextLoader("recipes.txt")
    docs = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    return vectorstore

#vectorstore = build_vectorstore()     --- Uses FAISS with a small corpus of custom recipes in the recipes.txt
vectorstore = build_vectorstore_from_csv("recipes.csv")
retriever = vectorstore.as_retriever()

def format_chat_history(history):
    return "\n".join([f"{role}: {msg}" for role, msg in history])

def retrieve_step(state):
    query = state["query"]
    docs = retriever.get_relevant_documents(query)
    return {
        **state,
        "docs": docs,
        "chat_history": state.get("chat_history", []) + [("user", query)]
    }

def generate_step(state):
    docs = state["docs"]
    context = "\n".join([doc.page_content for doc in docs])
    history = state["chat_history"]
    prompt = f"""
You are a zero-waste cooking assistant.
Use only these ingredients: {state['query']}.
Use this context:
{context}

Conversation:
{format_chat_history(history)}

Generate a creative, waste-free recipe and steps.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )

    message = response.choices[0].message.content
    return {
        **state,
        "result": message,
        "chat_history": history + [("assistant", message)],
        "recipe": message,
        "step_index": state["step_index"] + 1
    }

def followup_step(state):
    query = state["query"]
    history = state["chat_history"]
    recipe = state["recipe"]
    prompt = f"""
Continue the cooking conversation about this recipe:
{recipe}

Conversation:
{format_chat_history(history)}

User: {query}
Assistant:
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False
    )

    message = response.choices[0].message.content
    return {
        **state,
        "result": message,
        "chat_history": history + [("assistant", message)],
        "recipe": recipe,
        "continue_chat": False,
        "step_index": state["step_index"] + 1
    }

def entry_condition(state: State) -> str:
    # If it's the first turn, go to retrieve
    if state["step_index"] == 0:
        return "retrieve"
    else:
        return "followup"

builder = StateGraph(State)
builder.add_node("retrieve", retrieve_step)
builder.add_node("generate", generate_step)
builder.add_node("followup", followup_step)

builder.set_conditional_entry_point(entry_condition)
builder.add_edge("retrieve", "generate")
#builder.add_edge("generate", "followup")
#builder.add_edge("followup", "followup")



# 🔁 Conditional routing to either continue or end the chat
def followup_condition(state: State) -> str:
    # Continue followup if flag is True, else end
    if state.get("continue_chat", False):
        return "followup"
    else:
        return "__end__"

builder.add_conditional_edges(
    "followup",
    followup_condition,
    {
        "followup": "followup",
        "__end__": END
    }
)

app_flow = builder.compile()
