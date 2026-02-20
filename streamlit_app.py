import requests
import streamlit as st

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="News Similarity Finder", page_icon="📰", layout="centered")
st.title("📰 News Article Similarity Finder")
st.caption("Search for an article, select one, and get similar articles using FastAPI + Streamlit.")

with st.sidebar:
    st.header("API")
    api = st.text_input("Base URL", value=API_BASE)

st.markdown("### 1) Search for an article")
q = st.text_input("Search keyword", placeholder="e.g., pension")
search_results = []
if st.button("Search"):
    r = requests.get(f"{api}/articles/search", params={"q": q}, timeout=10)
    if r.status_code == 200:
        search_results = r.json()["results"]
        if search_results:
            st.write("Matches:")
            for idx, art in enumerate(search_results):
                st.write(f"{idx + 1}. {art}")
        else:
            st.info("No matches found.")
    else:
        st.error(r.json().get("detail", "Search failed"))

st.markdown("### 2) Select article index and get recommendations")
article_idx = st.number_input("Article index (from search results)", min_value=0, step=1, value=0)
k = st.slider("How many recommendations?", min_value=1, max_value=10, value=3)

if st.button("Recommend 🧠"):
    payload = {"article_idx": int(article_idx), "k": k}
    r = requests.post(f"{api}/recommend", json=payload, timeout=15)
    if r.status_code == 200:
        data = r.json()
        st.subheader(f"Recommendations for article index: {data['input_idx']}")
        for i, rec in enumerate(data["recommendations"], start=1):
            with st.container(border=True):
                st.markdown(f"**{i}. {rec['title']}**")
                st.caption(rec["reason"])
    else:
        st.error(r.json().get("detail", "Recommendation failed"))
