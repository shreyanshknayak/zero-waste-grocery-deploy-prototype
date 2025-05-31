# 🥦 Zero-Waste Grocery Helper

**A conversational AI assistant that helps you generate creative, zero-waste recipes using the ingredients you already have.**

Built using [LangGraph](https://github.com/langchain-ai/langgraph), [Streamlit](https://streamlit.io/), and [FAISS](https://github.com/facebookresearch/faiss) or [Pinecone](https://www.pinecone.io/) for a fast, scalable user experience.

---

## 🚀 Features

* ♻️ **Zero-waste cooking assistant** that suggests recipes based on your available ingredients.
* 📚 **Semantic recipe search** using Pinecone vectorstore with HuggingFace sentence embeddings.
* 💬 **Conversational flow** powered by LangGraph state machines.
* 🔁 **Follow-up interaction** to continue chatting about the recipe or ask for variations.
* ⚡️ **Fast and interactive UI** built with Streamlit.
* 🧠 **Enhanced knowledge base** using top 70 recipes from **Recipe1M+** dataset.

---

## 🛠️ How It Works

### Graph Flow

```mermaid
graph TD
    A[Start] --> B[Retrieve Ingredients]
    B --> C[Generate Recipe]
    C --> D[Follow-up Conversation]
    D -->|User continues| D
    D -->|User ends| E[End]
````

1. **Retrieve**: Search relevant recipes using Pinecone vectorstore populated with the top 70 Recipe1M+ entries.
2. **Generate**: Craft a creative zero-waste recipe using an LLM.
3. **Follow-up**: Continue conversation based on recipe context and chat history.

---

## 🧩 Tech Stack

| Component     | Tech Used                                            |
| ------------- | ---------------------------------------------------- |
| UI            | Streamlit                                            |
| State Machine | LangGraph (LangChain)                                |
| LLM           | Groq API using LLaMA 4 Scout 17B                     |
| Embeddings    | HuggingFace `all-MiniLM-L6-v2`                       |
| Vector Search | Pinecone (formerly FAISS)                            |
| Data          | `recipes.txt` + Recipe1M+ (top 70 from `recipe.csv`) |

---

## 🧱 Project Structure

```
.
├── langgraph_app.py         # Main LangGraph app
├── streamlit_app.py         # Streamlit interface
├── index_pinecone.py        # Script to process & push Recipe1M+ entries
├── .gitignore               # recipe.csv is ignored due to size
├── requirements.txt
└── README.md
```

> **Note**: `recipe.csv` (from Recipe1M+) is not included in version control due to size. Ensure it's placed locally for indexing.

---

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/shreyanshknayak/zero-waste-grocery-helper.git
cd zero-waste-grocery-helper
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your environment variables

```bash
export GROQ_API_KEY="your-groq-key-here"
export PINECONE_API_KEY="your-pinecone-key-here"
```

### 4. Start the app

```bash
streamlit run streamlit_app.py
```

---

## 🧪 Example Prompts

> "I have tomatoes, onions, garlic, and stale bread. What can I make?"
-
> "Can you make it gluten-free?"
-
> "Give me a zero-waste variation without using an oven."

---

## ✅ To-Do

* [ ] Add image-based ingredient recognition (CV integration)
* [ ] Save user sessions and recipes
* [ ] Deploy to Streamlit Cloud
* [ ] Add voice interface
* [ ] Expand Pinecone index to include more of Recipe1M+

---

## 🧠 Acknowledgments

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [FAISS](https://github.com/facebookresearch/faiss)
* [Pinecone](https://www.pinecone.io/)
* [Groq LLMs](https://console.groq.com/)
* [Streamlit](https://streamlit.io)
* [Recipe1M+ Dataset](https://www.kaggle.com/datasets/kaggle/recipe-ingredients-dataset)





