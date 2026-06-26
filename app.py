import streamlit as st
import os
from dotenv import load_dotenv
import requests
from datetime import datetime

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from tavily import TavilyClient
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call

# ==========================================
# PAGE CONFIG & STYLING
# ==========================================

st.set_page_config(
    page_title="🏙️ City Assistant",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    
    .chat-message.user {
        background-color: #1e3a5f;
        color: white;
        justify-content: flex-end;
        border-left: 4px solid #ff6b35;
    }
    
    .chat-message.bot {
        background-color: white;
        color: #1e293b;
        border-left: 4px solid #2c5aa0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .chat-message.system {
        background-color: #f0f4f8;
        color: #64748b;
        border-left: 4px solid #64748b;
    }
    
    .message-avatar {
        font-size: 1.5rem;
        line-height: 1;
    }
    
    .message-content {
        flex: 1;
        line-height: 1.6;
    }
    
    /* Tool approval card */
    .approval-card {
        background: linear-gradient(135deg, #fff5f0 0%, #ffe8dc 100%);
        border: 2px solid #ff6b35;
        border-radius: 0.75rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    .approval-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #ff6b35;
        margin-bottom: 0.5rem;
    }
    
    .approval-details {
        background-color: white;
        border-left: 3px solid #ff6b35;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
    }
    
    .tool-action {
        background-color: #f8fafc;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        color: #1e293b;
    }
    
    .tool-action strong {
        color: #ff6b35;
    }
    
    /* Buttons */
    .approval-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    /* Success and error messages */
    .success-message {
        background-color: #f0fdf4;
        border-left: 4px solid #4ade80;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #166534;
    }
    
    .error-message {
        background-color: #fef2f2;
        border-left: 4px solid #ef4444;
        padding: 1rem;
        border-radius: 0.5rem;
        color: #991b1b;
    }
    
    /* Info section */
    .info-section {
        background-color: white;
        border-left: 4px solid #2c5aa0;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    .info-section h3 {
        color: #1e3a5f;
        margin-bottom: 1rem;
    }
    
    .example-box {
        background-color: #f8fafc;
        border-left: 3px solid #ff6b35;
        padding: 0.75rem 1rem;
        margin: 0.75rem 0;
        border-radius: 0.4rem;
        font-size: 0.95rem;
        color: #1e293b;
    }
    
    /* Header styling */
    h1 {
        color: #1e3a5f;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.05);
    }
    
    h2 {
        color: #2c5aa0;
        border-bottom: 2px solid #ff6b35;
        padding-bottom: 0.5rem;
    }
    
    /* Loading animation */
    .loading {
        color: #2c5aa0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# AGENT SETUP
# ==========================================

@tool
def get_weather(city: str) -> str:
    """Get current weather of a city"""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city},IN&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if str(data.get("cod")) != "200":
            return f"Error: {data.get('message', 'Could not fetch weather')}"
        
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"].title()
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        
        return f"🌦️ Weather in {city.title()}: {desc}\n📍 Temperature: {temp}°C\n💨 Wind: {wind_speed} m/s\n💧 Humidity: {humidity}%"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"


@tool
def get_news(city: str) -> str:
    """Get latest news about a city"""
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    try:
        response = tavily_client.search(
            query=f"latest news {city}",
            search_depth="basic",
            max_results=3
        )
        
        results = response.get("results", [])
        
        if not results:
            return f"No news found for {city}"
        
        news_text = f"📰 Latest news about {city.title()}:\n\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("content", "")[:150]
            news_text += f"{i}. {title}\n   {snippet}...\n   🔗 {url}\n\n"
        
        return news_text
    except Exception as e:
        return f"Error fetching news: {str(e)}"


# Initialize LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Tool call tracking
class ApprovalMiddleware:
    def __init__(self):
        self.pending_tool_call = None
    
    @wrap_tool_call
    def handle_tool_call(self, request, handler):
        tool_name = request.tool_call["name"]
        tool_input = request.tool_call.get("args", {})
        
        self.pending_tool_call = {
            "id": request.tool_call["id"],
            "name": tool_name,
            "input": tool_input,
            "timestamp": datetime.now().isoformat()
        }
        
        return ToolMessage(
            content="Tool call pending user approval",
            tool_call_id=request.tool_call["id"]
        )

approval_middleware = ApprovalMiddleware()

memory = MemorySaver()
agent = create_agent(
    llm,
    tools=[get_weather, get_news],
    system_prompt="You are a helpful city assistant. Provide weather and news information about Indian cities. Be conversational and friendly.",
    middleware=[approval_middleware.handle_tool_call],
    checkpointer=memory
)

# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! 👋 I'm your city assistant. I can help you find weather information and latest news about any city in India. What would you like to know?",
            "avatar": "🤖"
        }
    ]

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "streamlit_city_assistant"

if "pending_approval" not in st.session_state:
    st.session_state.pending_approval = None

if "tool_result" not in st.session_state:
    st.session_state.tool_result = None

# ==========================================
# MAIN LAYOUT
# ==========================================

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("# 🏙️ City Assistant")
    st.markdown("*Get weather and news about Indian cities with real-time approval*")

# Create two columns: Chat (left) and Info (right)
chat_col, info_col = st.columns([2, 1], gap="large")

with chat_col:
    st.markdown("## Chat")
    
    # Display chat messages
    messages_container = st.container(height=400, border=False)
    
    with messages_container:
        for message in st.session_state.messages:
            role = message.get("role", "assistant")
            avatar = message.get("avatar", "🤖" if role == "assistant" else "👤")
            
            if role == "assistant":
                st.markdown(f"""
                <div class="chat-message bot">
                    <div class="message-avatar">{avatar}</div>
                    <div class="message-content">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message-avatar">{avatar}</div>
                    <div class="message-content">{message['content']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Tool approval section
    if st.session_state.pending_approval:
        tool_call = st.session_state.pending_approval
        
        st.markdown(f"""
        <div class="approval-card">
            <div class="approval-header">⚡ Tool Approval Required</div>
            <p>The assistant wants to <strong>{tool_call['name'].replace('_', ' ').title()}</strong> for <strong>{tool_call['input'].get('city', 'unknown').title()}</strong></p>
            <div class="approval-details">
                <strong>Action:</strong> {tool_call['name']}<br>
                <strong>City:</strong> {tool_call['input'].get('city', 'N/A').title()}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_approve, col_deny = st.columns(2)
        with col_approve:
            if st.button("✅ Approve", key="approve_btn", use_container_width=True):
                # Execute the tool
                try:
                    if tool_call['name'] == 'get_weather':
                        result = get_weather(tool_call['input']['city'])
                    elif tool_call['name'] == 'get_news':
                        result = get_news(tool_call['input']['city'])
                    else:
                        result = "Unknown tool"
                    
                    st.session_state.tool_result = result
                    st.session_state.pending_approval = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Error executing tool: {str(e)}")
        
        with col_deny:
            if st.button("❌ Deny", key="deny_btn", use_container_width=True):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "I understand. I won't execute that action. Is there anything else I can help you with?",
                    "avatar": "🤖"
                })
                st.session_state.pending_approval = None
                st.rerun()
    
    # Display tool result if available
    if st.session_state.tool_result:
        st.markdown(f"""
        <div class="success-message">
            <strong>Tool Executed Successfully:</strong><br>{st.session_state.tool_result}
        </div>
        """, unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant",
            "content": st.session_state.tool_result,
            "avatar": "🤖"
        })
        st.session_state.tool_result = None
    
    # Input area
    st.markdown("---")
    user_input = st.text_input(
        "Your message",
        placeholder="Ask about weather, news, or anything about a city...",
        label_visibility="collapsed"
    )
    
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "avatar": "👤"
        })
        
        # Show loading state
        with st.spinner("🤔 Agent is thinking..."):
            try:
                config = {
                    "configurable": {
                        "thread_id": st.session_state.thread_id
                    }
                }
                
                result = agent.invoke({
                    "messages": [{"role": "user", "content": user_input}]
                }, config=config)
                
                bot_response = result['messages'][-1].content
                
                # Add bot message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response,
                    "avatar": "🤖"
                })
                
                # Check for pending tool call
                if approval_middleware.pending_tool_call:
                    st.session_state.pending_approval = approval_middleware.pending_tool_call
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        st.rerun()

with info_col:
    st.markdown("## 💡 Examples")
    st.markdown("""
    Try asking these questions:
    """)
    
    examples = [
        "What's the weather in Mumbai?",
        "Tell me the latest news about Ahmedabad",
        "How's the weather in Bangalore right now?",
        "What's happening in Delhi today?",
        "Show me news about Hyderabad",
        "Get the weather in Pune"
    ]
    
    for example in examples:
        st.markdown(f"""
        <div class="example-box">
        {example}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("## ℹ️ About")
    st.markdown("""
    This is your **City Assistant** - powered by:
    
    - 🤖 **Mistral AI** for conversation
    - 🌦️ **OpenWeather API** for live weather
    - 📰 **Tavily Search** for news
    
    All tool calls require your approval before execution.
    """)
    
    # Settings
    with st.expander("⚙️ Settings"):
        st.markdown("### Thread ID")
        new_thread = st.text_input("Chat Thread ID", value=st.session_state.thread_id)
        if new_thread != st.session_state.thread_id:
            st.session_state.thread_id = new_thread
            st.success("Thread ID updated!")
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! 👋 I'm your city assistant. I can help you find weather information and latest news about any city in India. What would you like to know?",
                    "avatar": "🤖"
                }
            ]
            st.success("Chat history cleared!")
            st.rerun()