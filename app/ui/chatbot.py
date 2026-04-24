# ============================================================
# CONTEXT-AWARE RAG CHATBOT FOR INDIAN UNION BUDGET (2023–2026)
# ============================================================

# -----------------------------
# 1️⃣ Imports
# -----------------------------
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.memory import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain
)
from langchain.chains.combine_documents import create_stuff_documents_chain


# -----------------------------
# 2️⃣ Load Environment Variables
# -----------------------------
load_dotenv()

VECTOR_DB_PATH = "FinancialBills_faiss_index"


# -----------------------------
# 3️⃣ Initialize LLM
# -----------------------------
# This LLM is used BOTH for:
# - Query reformulation
# - Final answer generation
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# -----------------------------
# 4️⃣ Load FAISS Vector Database
# -----------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})


# -----------------------------
# 5️⃣ Create History-Aware Retriever
# -----------------------------
# This prompt rewrites ambiguous follow-up questions
# into standalone queries BEFORE retrieval

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Given a chat history and the latest user question "
     "which might reference context in the chat history, "
     "formulate a standalone question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm,
    retriever,
    contextualize_q_prompt
)


# -----------------------------
# 6️⃣ Create Custom Budget QA Prompt
# -----------------------------
# This prompt is used AFTER retrieval.
# It uses:
# - Retrieved document context
# - Chat history
# - Current user question

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert assistant analyzing Indian Union Budget speeches (2023–2026).

Use ONLY the provided context from the speeches.
If the answer is not found in the context, respond with:
"I’m not sure based on the available budget speech documents."

Guidelines:
- Mention the specific budget year when answering.
- Quote or paraphrase accurately from the context.
- Do not introduce external knowledge.
- If the question compares years, structure the answer year-wise.

Budget Speech Excerpts:
{context}
"""),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])


# -----------------------------
# 7️⃣ Combine Documents Chain
# -----------------------------
# This injects retrieved documents into the QA prompt
combine_docs_chain = create_stuff_documents_chain(
    llm,
    qa_prompt
)


# -----------------------------
# 8️⃣ Final Retrieval Chain (Full RAG)
# -----------------------------
# Flow:
# Question → Rewrite → Retrieve → Answer

rag_chain = create_retrieval_chain(
    history_aware_retriever,
    combine_docs_chain
)


# -----------------------------
# 9️⃣ Chat Loop with Memory
# -----------------------------
def chat():
    print("\n Budget Support Chatbot (type 'exit' to quit)\n")

    chat_history = ChatMessageHistory()

    while True:
        question = input("User: ")

        if question.lower() == "exit":
            print("\n👋 Exiting chatbot.")
            break

        # Invoke RAG pipeline
        response = rag_chain.invoke({
            "input": question,
            "chat_history": chat_history.messages
        })

        answer = response["answer"]

        print("\n🤖 Financial Budget Bot:")
        print(answer)
        print("-" * 60)

        # Store conversation history
        chat_history.add_user_message(question)
        chat_history.add_ai_message(answer)


# -----------------------------
# 🔟 Run Chatbot
# -----------------------------
if __name__ == "__main__":
    chat()