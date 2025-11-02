import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(page_title="FYP Monitoring", layout="wide")
st.title("🚀 Multi-Agent Knowledge System - Monitoring Dashboard")

st.sidebar.header("⚙️ Controls")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

col1, col2, col3, col4 = st.columns(4)

try:
    metrics = requests.get(f"{api_url}/metrics").json()
    
    with col1:
        st.metric("Total Queries", metrics.get("total_queries", 0))
    with col2:
        st.metric("Avg Latency (ms)", f"{metrics.get('avg_latency_ms', 0):.0f}")
    with col3:
        st.metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.0%}")
    with col4:
        st.metric("Cache Hit Rate", f"{metrics.get('cache_hit_rate', 0):.0%}")
        
except Exception as e:
    st.error(f"❌ Cannot connect to API: {e}")

st.divider()

st.header("🧪 Test Queries")
query = st.text_input("Enter a query:", "What is FastAPI?")

if st.button("Send Query", key="main_query"):
    try:
        with st.spinner("Processing..."):
            response = requests.post(f"{api_url}/query", json={"query": query}).json()
        
        st.success("✅ Query processed!")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📝 Answer")
            st.write(response.get("answer", "No answer"))
        with col2:
            st.subheader("📊 Metrics")
            st.metric("Confidence", f"{response['validation']['confidence']}%")
            st.metric("Time", f"{response['processing_time']:.2f}s")
        
        st.subheader("📚 Sources")
        for i, source in enumerate(response.get("sources", []), 1):
            st.write(f"{i}. **{source['source']}** ({source['relevance']:.0%})")
            
    except Exception as e:
        st.error(f"❌ Error: {e}")

st.divider()
st.header("🏥 System Health")

try:
    health = requests.get(f"{api_url}/health").json()
    st.success(f"✅ API Status: {health.get('status', 'unknown')}")
except:
    st.error("❌ API is down")
