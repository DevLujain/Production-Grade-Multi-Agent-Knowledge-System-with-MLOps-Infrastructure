import streamlit as st
import requests
import json
from datetime import datetime

st.set_page_config(page_title="FYP Monitoring", layout="wide")
st.title("🚀 Multi-Agent Knowledge System - Monitoring Dashboard")

st.sidebar.header("⚙️ Controls")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000")

# ====== METRICS SECTION ======
st.header("📊 System Metrics")
col1, col2, col3, col4 = st.columns(4)

try:
    health = requests.get(f"{api_url}/health").json()
    with col1:
        st.metric("API Status", health.get("status", "unknown"))
except Exception as e:
    with col1:
        st.error("API Down")

with col2:
    st.metric("Region", "Singapore")
with col3:
    st.metric("Runtime", "Docker")
with col4:
    st.metric("Model", "Mixtral 8x7B")

st.divider()

# ====== TEST QUERIES SECTION ======
st.header("🧪 Test Queries")
query = st.text_input("Enter a query:", "What is FastAPI?", key="query_input")

if st.button("Send Query", key="send_button_unique"):
    try:
        with st.spinner("⏳ Processing your query..."):
            response = requests.post(
                f"{api_url}/query", 
                json={"query": query},
                timeout=30
            ).json()
        
        st.session_state.last_response = response
        st.success("✅ Query processed!")
        
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Check the URL above.")
    except requests.exceptions.Timeout:
        st.error("❌ Request timed out. API is taking too long.")
    except json.JSONDecodeError:
        st.error("❌ API returned invalid JSON. Check if API is running.")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

# Display response if it exists
if 'last_response' in st.session_state:
    response = st.session_state.last_response
    
    # Display Answer
    st.subheader("📝 Answer")
    st.write(response.get("answer", "No answer available"))
    
    # Display Confidence & Time
    col1, col2, col3 = st.columns(3)
    with col1:
        confidence = response.get("validation", {}).get("confidence", 0)
        st.metric("Confidence", f"{confidence}%")
    with col2:
        status = response.get("validation", {}).get("status", "Unknown")
        st.metric("Status", status)
    with col3:
        st.metric("Sources Found", len(response.get("sources", [])))
    
    # Display Sources
    if response.get("sources"):
        st.subheader("📚 Retrieved Sources")
        for i, source in enumerate(response.get("sources", []), 1):
            st.write(f"**{i}. {source['source']}** - Relevance: {source['relevance']:.0%}")
    
    # Show raw response in expander
    with st.expander("🔍 Show Raw Response"):
        st.json(response)

# ====== SYSTEM HEALTH ======
st.header("🏥 System Health")
col1, col2 = st.columns(2)

with col1:
    try:
        health = requests.get(f"{api_url}/health", timeout=5).json()
        st.success(f"✅ API Status: {health.get('status', 'unknown').upper()}")
        st.json(health)
    except Exception as e:
        st.error(f"❌ API is down: {str(e)}")

with col2:
    st.info("💡 Tips:\n- Change API URL in sidconfidenceebar\n- Check Render logs if API fails\n- Use http://localhost:8000 for local testing")

st.divider()

# ====== QUERY HISTORY ======
st.header("📜 Query History")

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Clear history button
if st.button("Clear History", key="clear_history_button"):
    st.session_state.query_history = []
    st.success("✅ History cleared!")

# Display history
if st.session_state.query_history:
    for i, item in enumerate(reversed(st.session_state.query_history[-10:]), 1):
        st.write(f"{i}. **{item['query']}** - {item['time']}")
else:
    st.write("No queries yet")

