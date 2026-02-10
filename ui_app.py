# bank_analyzer/ui_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os
from pathlib import Path

# Add parent directory to path to import your modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from src.config import ensure_directories
# Import your existing modules
try:
    
    from src.config import (
        RAW_DATA_PATH, PREPROCESSED_PATH, FINAL_CATEGORIZED_PATH,
        LLM_THRESHOLD, COHERE_API_KEY, GOOGLE_API_KEY
    )
    from src.preprocessing.preprocess import preprocess as preprocess_data
    from src.categorization.categorize import categorize as categorize_data
    from src.categorization.rule_based import rule_based_category
    HAS_BACKEND = True
except ImportError as e:
    st.warning(f"⚠️ Could not import backend modules: {e}")
    HAS_BACKEND = False

# Page config
st.set_page_config(
    page_title="Bank Analyzer AI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        padding: 20px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px;
    }
    
    .insight-card {
        background: white;
        border-radius: 10px;
        padding: 15px;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
        margin: 2px;
    }
    
    .status-complete { background: #d4edda; color: #155724; }
    .status-pending { background: #fff3cd; color: #856404; }
    .status-error { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'raw_uploaded': False,
        'preprocessed': False,
        'categorized': False,
        'df_raw': None,
        'df_processed': None,
        'df_categorized': None,
        'insights': {},
        'visualizations': {}
    }

# Sidebar
with st.sidebar:
    st.title("💸 Bank Analyzer")
    st.markdown("---")
    
    # Pipeline status
    st.subheader("Pipeline Status")
    
    status_items = [
        ("📁 Upload", st.session_state.pipeline_state['raw_uploaded']),
        ("⚙️ Preprocess", st.session_state.pipeline_state['preprocessed']),
        ("🏷️ Categorize", st.session_state.pipeline_state['categorized'])
    ]
    
    for icon, status in status_items:
        badge_class = "status-complete" if status else "status-pending"
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 8px 0;">
            <span style="margin-right: 10px;">{icon}</span>
            <span class="status-badge {badge_class}">
                {'✓ Complete' if status else '○ Pending'}
            </span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    use_llm = st.checkbox("Use AI Categorization", value=True)
    llm_threshold = st.slider("AI Confidence Threshold", 0.0, 1.0, LLM_THRESHOLD)
    
    # API Keys
    with st.expander("🔑 API Keys"):
        if not COHERE_API_KEY:
            st.warning("COHERE_API_KEY not found in config")
        if not GOOGLE_API_KEY:
            st.warning("GOOGLE_API_KEY not found in config")

def save_uploaded_file(uploaded_file):
    """Save uploaded file to the raw data path"""
    try:
        raw_path = Path(RAW_DATA_PATH)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(raw_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return True
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return False

def run_backend_pipeline():
    """Run your existing backend pipeline"""
    try:
        # Step 1: Preprocessing
        with st.spinner("Preprocessing transactions..."):
            df_processed = preprocess_data()
            if df_processed is not None:
                st.session_state.pipeline_state['df_processed'] = df_processed
                st.session_state.pipeline_state['preprocessed'] = True
        
        # Step 2: Categorization
        with st.spinner("Categorizing transactions..."):
            df_categorized = categorize_data()
            if df_categorized is not None:
                st.session_state.pipeline_state['df_categorized'] = df_categorized
                st.session_state.pipeline_state['categorized'] = True
                
                # Generate insights
                generate_insights(df_categorized)
        
        return True
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return False

def generate_insights(df):
    """Generate insights from categorized data"""
    try:
        insights = {}
        
        # Basic metrics
        if 'DIRECTION' in df.columns and 'AMOUNT' in df.columns:
            total_debit = df[df['DIRECTION'] == 'DEBIT']['AMOUNT'].sum()
            total_credit = df[df['DIRECTION'] == 'CREDIT']['AMOUNT'].sum()
            
            insights['metrics'] = {
                'total_transactions': len(df),
                'total_spent': total_debit,
                'total_received': total_credit,
                'net_savings': total_credit - total_debit
            }
        
        # Category distribution
        if 'CATEGORY' in df.columns:
            category_summary = df[df['DIRECTION'] == 'DEBIT'].groupby('CATEGORY')['AMOUNT'].sum().abs().reset_index()
            insights['categories'] = category_summary.to_dict('records')
        
        # Top merchants
        if 'COUNTERPARTY_NAME' in df.columns:
            top_merchants = df[df['DIRECTION'] == 'DEBIT'].groupby('COUNTERPARTY_NAME')['AMOUNT'].sum().abs().nlargest(5).reset_index()
            insights['top_merchants'] = top_merchants.to_dict('records')
        
        st.session_state.pipeline_state['insights'] = insights
        return insights
    except Exception as e:
        st.warning(f"Could not generate insights: {e}")
        return {}

def create_visualizations():
    """Create visualizations from insights"""
    viz = {}
    insights = st.session_state.pipeline_state['insights']
    
    try:
        # Category pie chart
        if 'categories' in insights and insights['categories']:
            categories_df = pd.DataFrame(insights['categories'])
            fig1 = px.pie(categories_df, values='AMOUNT', names='CATEGORY', 
                         title="Spending by Category", hole=0.3)
            viz['categories'] = fig1
        
        # Top merchants bar chart
        if 'top_merchants' in insights and insights['top_merchants']:
            merchants_df = pd.DataFrame(insights['top_merchants'])
            fig2 = px.bar(merchants_df, x='AMOUNT', y='COUNTERPARTY_NAME',
                         orientation='h', title="Top Merchants",
                         labels={'AMOUNT': 'Amount (₹)', 'COUNTERPARTY_NAME': 'Merchant'})
            viz['merchants'] = fig2
        
        st.session_state.pipeline_state['visualizations'] = viz
    except Exception as e:
        st.warning(f"Could not create visualizations: {e}")
    
    return viz

def main():
    st.title("💸 Bank Statement Analyzer")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📁 Upload", 
        "📊 Overview", 
        "📈 Analytics", 
        "🤖 AI Assistant", 
        "📥 Export"
    ])
    
    with tab1:
        render_upload_tab()
    
    with tab2:
        render_overview_tab()
    
    with tab3:
        render_analytics_tab()
    
    with tab4:
        render_ai_assistant_tab()
    
    with tab5:
        render_export_tab()

def render_upload_tab():
    """Upload tab content"""
    st.header("📁 Upload Bank Statement")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your bank statement CSV",
            key="file_uploader"
        )
        
        if uploaded_file is not None:
            # Show preview
            df_preview = pd.read_csv(uploaded_file)
            st.dataframe(df_preview.head(10), use_container_width=True)
            st.caption(f"Preview: {len(df_preview)} transactions found")
            
            # Save and process button
            if st.button("🚀 Process Statement", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Save file
                    if save_uploaded_file(uploaded_file):
                        # Run pipeline
                        success = run_backend_pipeline()
                        st.session_state.pipeline_state['raw_uploaded']=True
                        if success:
                            st.success("✅ Processing complete!")
                            st.rerun()
    
    with col2:
        st.info("**Supported Formats:**")
        st.markdown("""
        - ICICI Bank CSV
        - HDFC Bank CSV
        - SBI Statement
        
        **Required Columns:**
        - `DATE`
        - `PARTICULARS`
        - `DEPOSITS`
        - `WITHDRAWALS`
        - `BALANCE`
        """)

def render_overview_tab():
    """Overview tab content"""
    st.header("📊 Overview")
    
    if not st.session_state.pipeline_state['categorized']:
        st.info("👈 Upload and process a statement to see overview")
        return
    
    df = st.session_state.pipeline_state['df_categorized']
    insights = st.session_state.pipeline_state['insights']
    
    if df is None:
        st.warning("No data available")
        return
    
    # Metrics row
    if 'metrics' in insights:
        metrics = insights['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Transactions</h3>
                <h1>{metrics['total_transactions']}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Spent</h3>
                <h1>₹{abs(metrics['total_spent']):,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Received</h3>
                <h1>₹{metrics['total_received']:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            savings = metrics['net_savings']
            color = "green" if savings > 0 else "red"
            st.markdown(f"""
            <div class="metric-card">
                <h3>Net Savings</h3>
                <h1 style="color: {color};">₹{savings:,.0f}</h1>
            </div>
            """, unsafe_allow_html=True)
    
    # Quick insights
    st.subheader("💡 Quick Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'categories' in insights and insights['categories']:
            categories = pd.DataFrame(insights['categories'])
            top_category = categories.nlargest(1, 'AMOUNT')
            if not top_category.empty:
                st.markdown(f"""
                <div class="insight-card">
                    <strong>Top Spending Category:</strong><br>
                    {top_category.iloc[0]['CATEGORY']} (₹{top_category.iloc[0]['AMOUNT']:,.0f})
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if 'top_merchants' in insights and insights['top_merchants']:
            merchants = pd.DataFrame(insights['top_merchants'])
            top_merchant = merchants.nlargest(1, 'AMOUNT')
            if not top_merchant.empty:
                st.markdown(f"""
                <div class="insight-card">
                    <strong>Top Merchant:</strong><br>
                    {top_merchant.iloc[0]['COUNTERPARTY_NAME']} (₹{top_merchant.iloc[0]['AMOUNT']:,.0f})
                </div>
                """, unsafe_allow_html=True)
    
    # Data preview
    st.subheader("📋 Transaction Preview")
    st.dataframe(
        df[['DATE', 'COUNTERPARTY_NAME', 'CATEGORY', 'AMOUNT', 'DIRECTION']].head(20),
        use_container_width=True
    )

def render_analytics_tab():
    """Analytics tab content"""
    st.header("📈 Analytics")
    
    if not st.session_state.pipeline_state['categorized']:
        st.info("👈 Upload and process a statement to see analytics")
        return
    
    # Create visualizations if not already created
    if not st.session_state.pipeline_state['visualizations']:
        create_visualizations()
    
    viz = st.session_state.pipeline_state['visualizations']
    
    # Display charts
    col1, col2 = st.columns(2)
    
    with col1:
        if 'categories' in viz:
            st.plotly_chart(viz['categories'], use_container_width=True)
        else:
            st.info("No category data to visualize")
    
    with col2:
        if 'merchants' in viz:
            st.plotly_chart(viz['merchants'], use_container_width=True)
        else:
            st.info("No merchant data to visualize")
    
    # Advanced analytics
    st.subheader("🔍 Advanced Analysis")
    
    df = st.session_state.pipeline_state['df_categorized']
    
    if df is not None and 'DATE' in df.columns and 'AMOUNT' in df.columns:
        try:
            # df['DATE'] = pd.to_datetime(df['DATE'])
            df['Month'] = df['DATE'].dt.strftime('%Y-%m')
            
            monthly_summary = df.groupby(['Month', 'DIRECTION'])['AMOUNT'].sum().abs().unstack().fillna(0)
            monthly_summary['Net'] = monthly_summary.get('CREDIT', 0) - monthly_summary.get('DEBIT', 0)
            
            st.dataframe(monthly_summary, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate monthly summary: {e}")

def render_ai_assistant_tab():
    """AI Assistant tab content"""
    st.header("🤖 AI Assistant")
    
    if not st.session_state.pipeline_state['categorized']:
        st.info("👈 Upload and process a statement to use AI Assistant")
        return
    
    # Check if we have the required dependencies
    try:
        from langchain_cohere import ChatCohere
        from langchain.agents import create_agent
        AGENT_AVAILABLE = True
    except ImportError as e:
        st.warning(f"🤖 Agent dependencies not available: {e}")
        st.info("Install required packages: `pip install langchain langchain-cohere`")
        AGENT_AVAILABLE = False
        return
    
    # Initialize agent in session state
    if 'agent' not in st.session_state:
        try:
            from src.agent.tools import python_code_executor_tool
            from src.agent.system_prompt import SYSTEM_PROMPT
            from src.config import COHERE_API_KEY
            
            # Load the categorized data into a variable accessible by the tool
            df = st.session_state.pipeline_state['df_categorized']
            
            # Initialize LLM
            llm = ChatCohere(
                model="command-a-03-2025",
                temperature=0.4,
                cohere_api_key=COHERE_API_KEY
            )
            
            # Create agent
            agent = create_agent(
                model=llm,
                tools=[python_code_executor_tool],
                system_prompt=SYSTEM_PROMPT,
            )
            
            st.session_state.agent = agent
            st.session_state.agent_initialized = True
            st.session_state.chat_history = []
            
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {e}")
            st.session_state.agent_initialized = False
            return
    
    # Pre-built questions
    st.markdown("""
    ### Ask questions about your transactions
    
    **Example questions:**
    """)
    
    questions = [
        "How much total did I spend?",
        "What are my top 5 expenses?",
        "What's my average transaction amount?",
        "What's my total income vs expenses?"
    ]
    
    # Display questions in a grid
    cols = st.columns(3)
    for idx, question in enumerate(questions):
        with cols[idx % 3]:
            if st.button(f"❔ {question}", key=f"q_{idx}", use_container_width=True):
                st.session_state.user_query = question
                st.rerun()
    
    # Chat interface
    st.markdown("---")
    
    # Query input
    user_query = st.text_input(
        "Or type your own question:",
        value=st.session_state.get('user_query', ''),
        placeholder="e.g., How much did I spend on food last week?",
        key="query_input"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat history
    if 'chat_history' in st.session_state and st.session_state.chat_history:
        st.markdown("### 💬 Conversation")
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(f"**You:** {chat['question']}")
            
            with st.chat_message("assistant"):
                st.markdown(f"**Assistant:** {chat['answer']}")
    
    # Process query
    if ask_button and user_query:
        with st.spinner("🤔 Thinking..."):
            try:
                # Prepare the agent input
                agent_input = {
                    "messages": [{"role": "user", "content": user_query}],
                    "configurable": {"thread_id": "streamlit_session"}
                }
                
                # Get response from agent
                response = st.session_state.agent.invoke(agent_input)
                
                # Extract the response content
                if response and 'messages' in response:
                    last_message = response['messages'][-1]
                    answer = last_message.content
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_query,
                        'answer': answer
                    })
                    
                    # Clear the query input
                    st.session_state.user_query = ""
                    st.rerun()
                else:
                    st.error("No response from agent")
                    
            except Exception as e:
                st.error(f"❌ Error running agent: {str(e)}")
    
    # Explanation of capabilities
    with st.expander("ℹ️ What can I ask?", expanded=False):
        st.markdown("""
        **The AI Assistant can analyze your transaction data to answer questions like:**
        
        - **Totals:** "How much did I spend in total?", "What's my total income?"
        - **Categories:** "How much did I spend on food/transport/shopping?"
        - **Time periods:** "Show me last week's expenses", "What was my spending in January?"
        - **Merchants:** "How much did I spend at Amazon/Zomato?"
        - **Patterns:** "What are my recurring payments?", "Show me my spending trends"
        - **Comparisons:** "Compare food vs transport spending", "Month over month comparison"
        
        **Behind the scenes**, the agent runs Python pandas code on your transaction data to calculate the answers.
        """)
    
    # Debug info (hidden by default)
    # with st.expander("🔧 Debug Info", expanded=False):
    #     if st.session_state.pipeline_state['df_categorized'] is not None:
    #         st.write(f"DataFrame shape: {st.session_state.pipeline_state['df_categorized'].shape}")
    #         st.write(f"Columns: {list(st.session_state.pipeline_state['df_categorized'].columns)}")
            
    #         # Show sample of data that agent will work with
    #         st.write("Sample data (first 5 rows):")
    #         st.dataframe(st.session_state.pipeline_state['df_categorized'][['DATE', 'COUNTERPARTY_NAME', 'CATEGORY', 'AMOUNT', 'DIRECTION']].head())
def render_export_tab():
    """Export tab content"""
    st.header("📥 Export Data")
    
    if not st.session_state.pipeline_state['categorized']:
        st.info("👈 Upload and process a statement to export data")
        return
    
    df = st.session_state.pipeline_state['df_categorized']
    
    if df is None:
        st.warning("No data to export")
        return
    
    # Export options
    st.subheader("Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV Export
        csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download CSV",
            data=csv,
            file_name=f"bank_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Processed data only
        processed_df = st.session_state.pipeline_state.get('df_processed')
        if processed_df is not None:
            csv_processed = processed_df.to_csv(index=False)
            st.download_button(
                label="⚙️ Download Processed",
                data=csv_processed,
                file_name=f"processed_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        # Insights summary
        insights = st.session_state.pipeline_state['insights']
        if insights:
            # Create a simple text summary
            summary_lines = ["Bank Analysis Summary", "=" * 50, ""]
            
            if 'metrics' in insights:
                m = insights['metrics']
                summary_lines.extend([
                    f"Total Transactions: {m['total_transactions']}",
                    f"Total Spent: ₹{abs(m['total_spent']):,.0f}",
                    f"Total Received: ₹{m['total_received']:,.0f}",
                    f"Net Savings: ₹{m['net_savings']:,.0f}",
                    ""
                ])
            
            if 'categories' in insights and insights['categories']:
                summary_lines.append("Spending by Category:")
                for cat in insights['categories'][:5]:
                    summary_lines.append(f"  - {cat['CATEGORY']}: ₹{cat['AMOUNT']:,.0f}")
            
            summary_text = "\n".join(summary_lines)
            
            st.download_button(
                label="📄 Download Summary",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    # Data preview for export
    st.subheader("Export Preview")
    
    export_cols = st.multiselect(
        "Select columns to export:",
        df.columns.tolist(),
        default=['DATE', 'COUNTERPARTY_NAME', 'CATEGORY', 'AMOUNT', 'DIRECTION']
    )
    
    if export_cols:
        st.dataframe(df[export_cols].head(10), use_container_width=True)

if __name__ == "__main__":
    main()