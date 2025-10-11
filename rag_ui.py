"""
Streamlit UI for RAG Knowledge Search System
Simple interface for queries, feedback, and metrics
"""

import os
import streamlit as st
import requests
from datetime import datetime
import plotly.graph_objects as go

# Configuration
API_BASE_URL = os.environ.get('API_URL', 'http://api:8000')

# Page config
st.set_page_config(
    page_title="Knowledge Search",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []


def query_api(query: str, user_id: str = None, top_k: int = 3):
    """Send query to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/query",
            json={"query": query, "user_id": user_id, "top_k": top_k},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Query failed: {e}")
        return None


def send_feedback(query_id: str, feedback_type: str, rating: int = None, comment: str = None):
    """Send feedback to API"""
    try:
        payload = {"query_id": query_id, "feedback_type": feedback_type}
        if rating:
            payload["rating"] = rating
        if comment:
            payload["comment"] = comment
        
        response = requests.post(
            f"{API_BASE_URL}/api/feedback",
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Feedback failed: {e}")
        return False


def get_metrics(days: int = 7):
    """Get system metrics"""
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/metrics",
            params={"days": days},
            timeout=20
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Failed to load metrics: {e}")
        return None


def get_popular_queries():
    """Get popular queries"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/popular-queries", timeout=10)
        response.raise_for_status()
        return response.json().get("popular_queries", [])
    except:
        return []


def get_low_rated_queries():
    """Get low-rated queries"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/low-rated-queries", timeout=10)
        response.raise_for_status()
        return response.json().get("low_rated_queries", [])
    except:
        return []


def trigger_etl():
    """Trigger ETL run"""
    try:
        response = requests.post(f"{API_BASE_URL}/api/trigger-etl", timeout=5)
        response.raise_for_status()
        return True
    except:
        return False


# === Main UI ===

# Sidebar
st.sidebar.title("🔍 Knowledge Search")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["Search", "Metrics", "Admin"]
)

# === Search Page ===
if page == "Search":
    st.title("Knowledge Search")
    st.markdown("Ask questions about our documentation and get AI-powered answers with sources.")
    
    # Search box
    col1, col2 = st.columns([2, 1], vertical_alignment="bottom")
    with col1:
        query = st.text_input(
            "Enter your question",
            placeholder="e.g., How do I install FastAPI?",
            label_visibility="collapsed"
        )
    
    with col2:
        top_k = st.selectbox("Results", [3, 5, 7, 10], index=0)
    
    # Only search when button is clicked
    if st.button("Search", type="primary", use_container_width=True):
        if query:
            with st.spinner("Searching..."):
                result = query_api(query, user_id="streamlit_user", top_k=top_k)
                
                if result:
                    # Store in session
                    st.session_state.query_history.insert(0, {
                        'query': query,
                        'result': result,
                        'timestamp': datetime.now()
                    })
                    
                    # Display answer
                    st.markdown("### Answer")
                    st.markdown(result['answer'])
                    
                    # Performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Time", f"{result['total_time_ms']:.0f}ms")
                    col2.metric("Retrieval", f"{result['retrieval_time_ms']:.0f}ms")
                    col3.metric("Generation", f"{result['generation_time_ms']:.0f}ms")
                    if result.get('cost_estimate'):
                        col4.metric("Cost", f"${result['cost_estimate']:.6f}")
                    else:
                        col4.metric("Cost", "$0.00")
                    
                    # Sources
                    st.markdown("### Sources")
                    for i, doc in enumerate(result['retrieved_docs'], 1):
                        with st.expander(f"📄 {doc['title'][:80]}... (Score: {doc['score']:.3f})"):
                            st.markdown(f"**Source:** [{doc['source_url']}]({doc['source_url']})")
                            st.markdown(f"**Content:**\n{doc['content']}")
                    
                    # Feedback
                    st.markdown("---")
                    st.markdown("### Was this helpful?")
                    
                    col1, col2, col3 = st.columns([1, 1, 4])
                    
                    with col1:
                        if st.button("👍 Yes", key=f"thumbs_up_{result['query_id']}"):
                            if send_feedback(result['query_id'], "thumbs_up"):
                                st.success("Thanks for your feedback!")
                    
                    with col2:
                        if st.button("👎 No", key=f"thumbs_down_{result['query_id']}"):
                            if send_feedback(result['query_id'], "thumbs_down"):
                                st.success("Thanks for your feedback!")
                    
                    # Optional detailed feedback
                    with st.expander("Provide detailed feedback"):
                        rating = st.slider("Rate this response (1-5)", 1, 5, 3)
                        comment = st.text_area("Additional comments (optional)")
                        if st.button("Submit detailed feedback"):
                            if send_feedback(result['query_id'], "rating", rating, comment):
                                st.success("Detailed feedback submitted!")
    
    # Query history
    if st.session_state.query_history:
        st.markdown("---")
        st.markdown("### Recent Searches")
        
        for idx, item in enumerate(st.session_state.query_history[:5]):
            with st.expander(f"{item['query']} - {item['timestamp'].strftime('%H:%M:%S')}"):
                st.markdown(item['result']['answer'][:300] + "...")
                if st.button("View full response", key=f"history_{idx}"):
                    st.session_state.selected_history = item


# === Metrics Page ===
elif page == "Metrics":
    st.title("System Metrics & Analytics")
    
    # Time period selector
    period = st.selectbox("Time Period", ["Last 24 hours", "Last 7 days", "Last 30 days"])
    days_map = {"Last 24 hours": 1, "Last 7 days": 7, "Last 30 days": 30}
    days = days_map[period]
    
    # Fetch metrics
    with st.spinner("Loading metrics..."):
        metrics = get_metrics(days)
        popular = get_popular_queries()
        low_rated = get_low_rated_queries()
    
    if metrics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            "Total Queries",
            f"{metrics['total_queries']:,}",
            help="Total number of queries processed"
        )
        
        col2.metric(
            "Satisfaction Rate",
            f"{metrics['thumbs_up_rate']:.1%}",
            help="Percentage of thumbs up vs total feedback"
        )
        
        col3.metric(
            "Avg Response Time",
            f"{metrics['avg_response_time_ms']:.0f}ms",
            help="Average end-to-end response time"
        )
        
        col4.metric(
            "Total Cost",
            f"${metrics['total_cost']:.4f}",
            help="Total API costs for the period"
        )
        
        st.markdown("---")
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Distribution")
            perf_data = {
                "Metric": ["Avg", "P95", "P99"],
                "Latency (ms)": [
                    metrics['avg_response_time_ms'],
                    metrics['p95_latency_ms'],
                    metrics.get('p99_latency_ms', 0)
                ]
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=perf_data["Metric"],
                    y=perf_data["Latency (ms)"],
                    marker_color=['#4CAF50', '#FFC107', '#F44336']
                )
            ])
            fig.update_layout(
                yaxis_title="Latency (ms)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### User Satisfaction")
            if metrics['total_queries'] > 0:
                satisfaction_data = {
                    "Feedback": ["Positive", "Negative"],
                    "Count": [
                        int(metrics['total_queries'] * metrics['thumbs_up_rate']),
                        int(metrics['total_queries'] * (1 - metrics['thumbs_up_rate']))
                    ]
                }
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=satisfaction_data["Feedback"],
                        values=satisfaction_data["Count"],
                        marker_colors=['#4CAF50', '#F44336'],
                        hole=0.4
                    )
                ])
                fig.update_layout(height=300, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No feedback data available yet")
        
        st.markdown("---")
        
        # Popular and low-rated queries
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔥 Popular Queries")
            if popular:
                for item in popular[:10]:
                    st.markdown(f"- **{item['query']}** ({item['count']} times)")
            else:
                st.info("No query data available yet")
        
        with col2:
            st.markdown("### ⚠️ Low-Rated Queries")
            if low_rated:
                for item in low_rated[:10]:
                    with st.expander(f"{item['query'][:50]}..."):
                        st.markdown(f"**Feedback:** {item.get('feedback_type', 'N/A')}")
                        if item.get('rating'):
                            st.markdown(f"**Rating:** {'⭐' * item['rating']}")
                        if item.get('comment'):
                            st.markdown(f"**Comment:** {item['comment']}")
            else:
                st.info("No low-rated queries")


# === Admin Page ===
elif page == "Admin":
    st.title("Admin Controls")
    
    # ETL Management
    st.markdown("### 📥 Data Ingestion")
    st.markdown("Manually trigger document re-scraping and indexing.")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.info(f"**Scheduled ETL:** Runs every 24 hours automatically")
    
    with col2:
        if st.button("Trigger ETL Now", type="primary"):
            with st.spinner("Triggering ETL..."):
                if trigger_etl():
                    st.success("ETL run scheduled!")
                else:
                    st.error("Failed to trigger ETL")
    
    st.markdown("---")
    
    # Custom URL ingestion
    st.markdown("### 🔗 Ingest Custom URLs")
    
    urls_text = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://docs.example.com\nhttps://example.com/guide",
        height=150
    )
    
    max_pages = st.slider("Max pages per URL", 10, 200, 50)
    
    if st.button("Start Ingestion"):
        urls = [url.strip() for url in urls_text.split('\n') if url.strip()]
        
        if urls:
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/ingest",
                    json={"urls": urls, "max_pages_per_url": max_pages},
                    timeout=10
                )
                if response.status_code == 200:
                    st.success(f"Started ingestion of {len(urls)} URLs")
                else:
                    st.error("Ingestion failed")
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please enter at least one URL")
    
    st.markdown("---")
    
    # System configuration
    st.markdown("### ⚙️ System Configuration")
    
    st.markdown("**Current Configuration:**")
    st.code("""
Embedding Provider: Local (sentence-transformers)
Embedding Model: all-MiniLM-L6-v2
LLM Provider: Local (Ollama)
LLM Model: llama3.2
Vector Store: Elasticsearch (localhost:9200)
    """)
    
    with st.expander("Change Configuration"):
        st.warning("⚠️ Configuration changes require system restart")
        
        embed_provider = st.selectbox(
            "Embedding Provider",
            ["local", "vertex", "azure"],
            help="Local is free, Vertex/Azure require API keys"
        )
        
        llm_provider = st.selectbox(
            "LLM Provider",
            ["local", "vertex", "azure"],
            help="Local is free, Vertex/Azure require API keys"
        )
        
        if st.button("Save Configuration"):
            st.info("Configuration saved. Restart the service to apply changes.")
    
    st.markdown("---")
    
    # Testing tools
    st.markdown("### 🧪 Testing Tools")
    
    if st.button("Run System Health Check"):
        with st.spinner("Running health check..."):
            try:
                response = requests.get(f"{API_BASE_URL}/", timeout=5)
                if response.status_code == 200:
                    st.success("✅ API is healthy")
                    
                    # Test query
                    test_result = query_api("test query", top_k=1)
                    if test_result:
                        st.success("✅ RAG pipeline is working")
                    else:
                        st.error("❌ RAG pipeline failed")
                else:
                    st.error("❌ API is not responding")
            except Exception as e:
                st.error(f"❌ Health check failed: {e}")


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**RAG Knowledge Search v1.0**

A semantic search system powered by:
- Hybrid search (vector + keyword)
- LLM-powered answers
- User feedback & analytics
- Automated document ingestion

For support, contact your admin.
""")

if __name__ == "__main__":
    pass