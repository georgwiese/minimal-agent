import os
import secrets
import gradio as gr
from dotenv import load_dotenv
from minimal_agent.agent import Agent
from minimal_agent.tools import VisitWebpageTool, TavilySearchTool

load_dotenv()

# Global agent instance to maintain state
agent_instance = None

def create_agent():
    """Create an agent with Tavily search tool."""
    tools = [
        TavilySearchTool(max_results=10),
        VisitWebpageTool(max_output_length=1000)
    ]
    
    return Agent(
        model=os.environ.get("MODEL"),
        tools=tools,
    )

def format_reasoning_steps(steps):
    """Format reasoning steps as markdown with separate expandable sections."""
    markdown = "### Reasoning Steps:\n\n"
    
    if not steps:
        markdown += "*Reasoning steps will appear here as the agent works through your question...*\n\n"
        return markdown
    
    for i, step in enumerate(steps, 1):
        summary = step.summary
        
        # Step header with summary
        markdown += f"**Step {i}:** {summary}\n\n"
        
        # Expandable thought section
        if step.thought:
            markdown += "<details>\n"
            markdown += "<summary>💭 Thought</summary>\n\n"
            markdown += step.thought + "\n\n"
            markdown += "</details>\n\n"
        
        # Expandable code section
        if step.code:
            markdown += "<details>\n"
            markdown += "<summary>💻 Code</summary>\n\n"
            markdown += "```python\n"
            markdown += step.code
            markdown += "\n```\n\n"
            markdown += "</details>\n\n"
        
        # Expandable observation section
        if step.observation:
            markdown += "<details>\n"
            markdown += "<summary>👁️ Observation</summary>\n\n"
            markdown += '<div style="max-height: 300px; overflow-y: auto; background-color: #f6f8fa; padding: 10px; border-radius: 5px;">\n\n'
            markdown += "```\n"
            markdown += step.observation
            markdown += "\n```\n\n"
            markdown += "</div>\n"
            markdown += "</details>\n\n"
    
    return markdown

def clear_conversation():
    """Clear the conversation and reset the agent."""
    global agent_instance
    agent_instance = None
    return gr.update(visible=False), format_reasoning_steps([]), [], "", format_conversation_history([])

def format_conversation_history(history):
    """Format conversation history for display in chat style."""
    if not history:
        return "_Start a conversation by asking a question below._"
    
    markdown = ""
    for i in range(0, len(history), 2):
        if i < len(history):
            # User message
            markdown += f'<div style="text-align: right; margin: 10px 0;">\n'
            markdown += f'<div style="display: inline-block; background-color: #007bff; color: white; padding: 10px 15px; border-radius: 15px; max-width: 70%;">\n'
            markdown += f'{history[i]["content"]}\n'
            markdown += '</div>\n</div>\n\n'
        
        if i + 1 < len(history):
            # Agent message
            markdown += f'<div style="text-align: left; margin: 10px 0;">\n'
            markdown += f'<div style="display: inline-block; background-color: #f1f3f5; color: black; padding: 10px 15px; border-radius: 15px; max-width: 70%;">\n'
            markdown += f'{history[i+1]["content"]}\n'
            markdown += '</div>\n</div>\n\n'
    
    return markdown

def run_agent_query_streaming(query, conversation_history):
    """Run the agent with the given query, streaming results."""
    global agent_instance
    
    if not query.strip():
        yield "", "", conversation_history
        return
    
    if not os.environ.get("MODEL"):
        yield "Error: MODEL environment variable not set. Please configure your .env file.", "", conversation_history
        return
    
    if not os.environ.get("TAVILY_API_KEY"):
        yield "Error: TAVILY_API_KEY environment variable not set. Please configure your .env file.", "", conversation_history
        return
    
    try:
        # Create agent if it doesn't exist or if starting new conversation
        if agent_instance is None:
            agent_instance = create_agent()
        
        final_answer = ""
        
        # Determine if this is a new conversation or continuation
        is_new_conversation = len(conversation_history) == 0
        
        # Stream results as they come
        for update in agent_instance.run_streaming(query, reset_history=is_new_conversation):
            steps_markdown = format_reasoning_steps(update["steps"])
            if update["final_answer"]:
                final_answer = update["final_answer"]
                # Update conversation history
                new_history = conversation_history + [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": final_answer}
                ]
                yield final_answer, steps_markdown, new_history
            else:
                yield final_answer, steps_markdown, conversation_history
            
    except Exception as e:
        yield f"Error: {str(e)}", "", conversation_history

# Create Gradio interface
with gr.Blocks(title="Minimal Agent Web UI") as demo:
    gr.Markdown(
        """
        # Minimal Agent Web Interface
        
        Ask questions and the agent will search the web and reason through the answer.
        """
    )
    
    # State to store conversation history
    conversation_state = gr.State([])
    
    with gr.Row():
        # Left column for chat interface
        with gr.Column(scale=1):
            # Chat history area (top)
            conversation_display = gr.Markdown(
                label="Chat",
                value=format_conversation_history([]),
                elem_id="chat-display",
                height=400
            )
            
            # Current answer area
            output = gr.Markdown(
                label="Current Response",
                value="",
                visible=False  # Only show when there's a response
            )
            
            # Input area (bottom)
            with gr.Row():
                query_input = gr.Textbox(
                    label="",
                    placeholder="Ask a question...",
                    lines=1,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("Clear Conversation", variant="secondary", size="sm")
            
        # Right column for reasoning steps
        with gr.Column(scale=1):
            reasoning_steps = gr.Markdown(
                label="Reasoning Steps",
                value=format_reasoning_steps([]),
                height=500
            )
    
    # Examples
    gr.Examples(
        examples=[
            ["What was the hottest day in 2024 and how much was the Dow Jones on that day?"],
            ["Who won the Nobel Prize in Physics in 2023?"],
            ["What is the current population of Tokyo?"],
            ["What are the latest developments in AI?"]
        ],
        inputs=query_input
    )
    
    # Event handlers
    def handle_submit(query, history):
        # Clear the query input after submission
        for result in run_agent_query_streaming(query, history):
            final_answer, steps, new_history = result
            history_display = format_conversation_history(new_history)
            # Hide the output area since we show everything in chat
            yield gr.update(visible=False), steps, new_history, history_display, ""
    
    submit_btn.click(
        fn=handle_submit,
        inputs=[query_input, conversation_state],
        outputs=[output, reasoning_steps, conversation_state, conversation_display, query_input]
    )
    
    query_input.submit(
        fn=handle_submit,
        inputs=[query_input, conversation_state],
        outputs=[output, reasoning_steps, conversation_state, conversation_display, query_input]
    )
    
    clear_btn.click(
        fn=clear_conversation,
        inputs=[],
        outputs=[output, reasoning_steps, conversation_state, query_input, conversation_display]
    )

def create_gradio_interface():
    """Create and return the Gradio interface."""
    return demo


if __name__ == "__main__":
    # Generate or use existing token
    token = os.environ.get("ACCESS_TOKEN")
    if not token:
        # Use hex token to avoid base64 special characters
        token = secrets.token_hex(32)
    
    # Try to get public IP
    import socket
    import urllib.request
    
    public_ip = None
    try:
        # Try to get public IP from external service
        response = urllib.request.urlopen('https://api.ipify.org', timeout=2)
        public_ip = response.read().decode('utf-8')
    except:
        try:
            # Fallback to local hostname
            hostname = socket.gethostname()
            public_ip = socket.gethostbyname(hostname)
        except:
            public_ip = "your-server-ip"
    
    # Print the access URL
    print(f"\n{'='*60}")
    print(f"Access the web app at:")
    print(f"  http://{public_ip}:7860/")
    print(f"")
    print(f"Login with:")
    print(f"  Username: (leave empty)")
    print(f"  Password: {token}")
    print(f"{'='*60}\n")
    

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        auth=lambda u, p: p == token,
        auth_message="Enter the access token"
    )