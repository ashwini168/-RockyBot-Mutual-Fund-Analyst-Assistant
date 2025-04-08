# ✅ RockyBot: Unified Financial Assistant and Blog Analyzer

import os
import streamlit as st
import pickle
import feedparser
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import pipeline
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

load_dotenv()

st.set_page_config(page_title="RockyBot - Finance & Blog Assistant", layout="wide")
st.title("🤖 RockyBot: Financial & Blog Analysis Assistant")

# Load Models
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=512)
llm = HuggingFacePipeline(pipeline=qa_pipeline)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# --- Financial Feeds ---
with st.sidebar:
    st.header("🗞️ Market Feeds & Mutual Funds")
    selected_feed = st.selectbox("Choose a financial source", [
        "Moneycontrol", "Economic Times", "Finshots"
    ])
    feed_urls = {
        "Moneycontrol": "https://www.moneycontrol.com/rss/MCtopnews.xml",
        "Economic Times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
        "Finshots": "https://finshots.in/rss"
    }
    selected_url = feed_urls[selected_feed]

@st.cache_data(ttl=600)
def fetch_rss_articles(feed_url):
    feed = feedparser.parse(feed_url)
    articles = []
    for entry in feed.entries[:5]:
        content = entry.get("summary", entry.get("description", ""))
        soup = BeautifulSoup(content, "html.parser")
        text = soup.get_text()
        summary = summarize_text(text)
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "summary": summary
        })
    return articles

def summarize_text(text):
    try:
        if len(text.split()) < 30:
            return text.strip()
        summary = summarizer(text[:1000], max_length=80, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except:
        return text.strip()

def fetch_top_mf_performers():
    return [
        {"name": "ICICI Pru Tech Fund", "return": "32.5%", "type": "Equity"},
        {"name": "Nippon India Growth Fund", "return": "29.3%", "type": "Equity"},
        {"name": "Axis Midcap Fund", "return": "27.9%", "type": "Equity"},
    ]

def compare_funds(fund1, fund2):
    return [
        {"fund": fund1, "1Y": "22.5%", "3Y": "18.4%", "expense": "1.6%", "risk": "Moderate"},
        {"fund": fund2, "1Y": "19.3%", "3Y": "17.1%", "expense": "1.2%", "risk": "Low"},
    ]

# --- Finance Section ---
st.subheader(f"📰 Latest from {selected_feed}")
for article in fetch_rss_articles(selected_url):
    st.markdown(f"**[{article['title']}]({article['link']})**")
    st.caption(article['summary'])

st.markdown("---")
st.subheader("📈 Top Performing Mutual Funds")
for fund in fetch_top_mf_performers():
    st.write(f"**{fund['name']}** - {fund['type']} - Return: {fund['return']}")

st.markdown("---")
st.subheader("🔍 Compare Two Mutual Funds")
f1 = st.text_input("Fund 1", value="ICICI Pru Tech Fund")
f2 = st.text_input("Fund 2", value="Axis Midcap Fund")
if st.button("Compare"):
    for info in compare_funds(f1, f2):
        st.markdown(f"**{info['fund']}**")
        st.write(f"- 1Y Return: {info['1Y']}\n- 3Y Return: {info['3Y']}\n- Expense: {info['expense']}\n- Risk: {info['risk']}")
        st.markdown("---")

# --- Blog Q&A Section ---
st.header("📚 Blog Analyzer & Q&A")
urls = []
for i in range(3):
    url = st.text_input(f"🔗 Blog URL {i+1}")
    if url:
        urls.append(url)

file_path = "faiss_blog_vector.pkl"
if st.button("🚀 Process Blogs"):
    loader = UnstructuredURLLoader(urls=urls)
    with st.spinner("Loading blog content..."):
        data = loader.load()
    if data:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("✅ Blogs processed! Ask away!")

query = st.text_input("💬 Ask about the blogs:")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever(search_type="similarity", k=3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    response = qa_chain.run(query)

    top_docs = retriever.get_relevant_documents(query)
    combined_text = " ".join([doc.page_content for doc in top_docs])
    try:
        summary = summarizer(combined_text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    except:
        summary = "⚠️ Could not summarize."

    st.subheader("📖 Answer")
    st.write(response)

    st.subheader("📝 Blog-Based Summary")
    st.write(summary)
elif query:
    st.warning("⚠️ Please process blog URLs first.")
