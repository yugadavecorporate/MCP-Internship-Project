import os
import sys
import json
import asyncio
import streamlit as st
from groq import AsyncGroq
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from context_manager import ConversationBufferMemory, inject_context_to_system_prompt

# Load environment variables from .env file
load_dotenv()

# Check if API Key is available
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("Missing API Key! Please create a .env file and add GROQ_API_KEY='your_key'.")
    st.stop()

# Initialize async Groq client
client = AsyncGroq(api_key=api_key)

# Minimalist UI configuration
st.set_page_config(page_title="Chat", page_icon="💬", layout="centered")

# Hide default Streamlit elements for a cleaner, app-like look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# App Title & Subtitle (Minimalist)
st.title("💬 Intelligent MCP Assistant")
st.caption("A smart chatbot fully integrated with local databases, file systems and the internet via MCP.")

# Initialize ConversationBufferMemory in Streamlit session state
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(max_turns=20)

# Display existing messages from memory buffer
for msg in st.session_state["memory"].get_history():
    if msg["role"] not in ["system", "tool"]:
        # Hide invisible/blank LLM tool-calling requests visually from user chat view
        if msg["role"] == "assistant" and not msg.get("content"):
            continue
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

async def process_user_input():
    """Asynchronous pipeline to establish the MCP context and bridge it with Groq."""
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
        env=os.environ.copy()
    )
    
    try:
        # Spawn background Stdio Server instance dynamically containing our SQLite and File Tooling logic
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # Step 1: Detect available LLM tools
                tools_response = await session.list_tools()
                groq_tools = [
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema
                        }
                    }
                    for tool in tools_response.tools
                ]
                
                # Step 2: Implement robust multi-iteration loop to support N-hops of Tool executing chains!
                MAX_ITERATIONS = 5
                
                with st.status("Analyzing Request Pipeline...", expanded=True) as status:
                    for iteration in range(MAX_ITERATIONS):
                        # Build full context payload via context injector (includes system prompt + history)
                        messages_payload = inject_context_to_system_prompt(memory=st.session_state["memory"])
                        
                        response = await client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=messages_payload,
                            temperature=0.2,  # Reduce temperature to greatly limit hallucination inaccuracies!
                            tools=groq_tools,
                            tool_choice="auto",
                        )
                        
                        response_msg = response.choices[0].message
                        
                        # Parse Groq response decision matrix (Tools required??)
                        if response_msg.tool_calls:
                            status.update(label=f"Iteration {iteration+1}: Executing {len(response_msg.tool_calls)} local tools...")
                            
                            # Stage into ConversationBufferMemory with tool_calls list
                            st.session_state["memory"].add_assistant_message(
                                content=response_msg.content,
                                tool_calls=[
                                    {
                                        "id": tc.id,
                                        "type": "function",
                                        "function": {
                                            "name": tc.function.name,
                                            "arguments": tc.function.arguments
                                        }
                                    }
                                    for tc in response_msg.tool_calls
                                ]
                            )
                            
                            for tool_call in response_msg.tool_calls:
                                tool_name = tool_call.function.name
                                tool_args = json.loads(tool_call.function.arguments)
                                
                                st.write(f"⚙️ Running `{tool_name}` ...")
                                
                                try:
                                    # Direct request securely to underlying stdio background child Python server
                                    mcp_result = await session.call_tool(tool_name, arguments=tool_args)
                                    
                                    tool_result_content = ""
                                    if hasattr(mcp_result, 'content') and mcp_result.content:
                                        tool_result_content = "\n".join([c.text for c in mcp_result.content if getattr(c, 'type', '') == "text"])
                                    
                                    # Standard error checking block    
                                    if getattr(mcp_result, 'isError', False):
                                        tool_result_content = f"Tool Failure Error: {tool_result_content}"
                                        
                                except Exception as e:
                                    tool_result_content = f"Python SDK Execution Error: {str(e)}"
                                    
                                # Persist tool result into ConversationBufferMemory
                                st.session_state["memory"].add_tool_result(
                                    tool_call_id=tool_call.id,
                                    name=tool_name,
                                    content=str(tool_result_content)
                                )
                                
                            # Cycle back upwards in the loop to re-feed contexts to the LLM agent!
                        else:
                            # Finished! No tool calls returned. Standard response generation!
                            status.update(label="Information Retrieval Completed!", state="complete", expanded=False)
                            
                            text_content = response_msg.content or "Sorry, I couldn't generate a fully certain response. Try being more specific."
                            
                            with st.chat_message("assistant"):
                                placeholder = st.empty()
                                displayed_text = ""
                                chunk_size = 4
                                # Custom Typewriter Micro-Interaction logic natively feeding chunks asynchronously!
                                for i in range(0, len(text_content), chunk_size):
                                    displayed_text += text_content[i:i+chunk_size]
                                    placeholder.markdown(displayed_text + "▌")
                                    await asyncio.sleep(0.01)
                                placeholder.markdown(displayed_text)
                                st.session_state["memory"].add_assistant_message(content=text_content)
                            
                            # Escape recursive iteration
                            break
                    else:
                        status.update(label="Iteration Count Exceeded", state="error", expanded=True)
                        st.error("Too many tools executed sequentially without a definitive answer. Context terminated to protect API quotas.")

                # Flag successful pipeline completion
                st.session_state["response_completed"] = True
                        
    except BaseException as e:
        # Silently swallow AnyIO/anyio TaskGroup teardown noise on Windows.
        if type(e).__name__ in ("ExceptionGroup", "BaseExceptionGroup") and st.session_state.get("response_completed", False):
            pass
        else:
            import traceback
            st.error(f"Unexpected Error: {e}")
            traceback.print_exc()

# Check User input field trigger via synchronous Streamlit run
if user_input := st.chat_input("Say something..."):
    # Append user message to memory and render it visually
    st.session_state["memory"].add_user_message(user_input)
    st.session_state["response_completed"] = False  # Reset state tracking
    with st.chat_message("user"):
        st.write(user_input)

    # Add a loading micro-interaction so the user knows something is happening
    with st.spinner("AI is connecting securely to Local Services..."):
        # Hand off standard application run sequence to newly spun standard Event Loop thread architecture
        try:
            asyncio.run(process_user_input())
        except BaseException as e:
            # We conditionally swallow TaskGroup errors ONLY if the whole pipeline succeeded.
            if type(e).__name__ in ("ExceptionGroup", "BaseExceptionGroup") and st.session_state.get("response_completed", False):
                pass
            else:
                st.error(f"Critical Pipeline Failure: {e}")
                if hasattr(e, 'exceptions'):
                    for sub in e.exceptions:
                        st.error(f"Details: {sub}")