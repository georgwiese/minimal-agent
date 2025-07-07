import os
import gradio as gr
from dotenv import load_dotenv
from minimal_agent.agent import Agent
from minimal_agent.tools import VisitWebpageTool, TavilySearchTool

load_dotenv()

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

def run_agent_query_streaming(query):
    """Run the agent with the given query, streaming results."""
    if not query.strip():
        yield "", ""
        return
    
    if not os.environ.get("MODEL"):
        yield "Error: MODEL environment variable not set. Please configure your .env file.", ""
        return
    
    if not os.environ.get("TAVILY_API_KEY"):
        yield "Error: TAVILY_API_KEY environment variable not set. Please configure your .env file.", ""
        return
    
    try:
        agent = create_agent()
        final_answer = ""
        
        # Stream results as they come
        for update in agent.run_streaming(query):
            steps_markdown = format_reasoning_steps(update["steps"])
            if update["final_answer"]:
                final_answer = update["final_answer"]
            yield final_answer, steps_markdown
            
    except Exception as e:
        yield f"Error: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(title="Minimal Agent Web UI") as demo:
    gr.Markdown(
        """
        # Minimal Agent Web Interface
        
        Ask questions and the agent will search the web and reason through the answer.
        """
    )
    
    with gr.Row():
        # Left column for input and output
        with gr.Column(scale=1):
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What was the hottest day in 2024?",
                lines=3
            )
            
            submit_btn = gr.Button("Ask Agent", variant="primary")
            
            output = gr.Textbox(
                label="Final Answer",
                lines=10,
                max_lines=20
            )
            
        # Right column for reasoning steps (takes more space)
        with gr.Column(scale=3):
            reasoning_steps = gr.Markdown(
                label="Reasoning Steps",
                value=format_reasoning_steps([])  # Initialize with empty steps to show placeholder
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
    submit_btn.click(
        fn=run_agent_query_streaming,
        inputs=query_input,
        outputs=[output, reasoning_steps]
    )
    
    query_input.submit(
        fn=run_agent_query_streaming,
        inputs=query_input,
        outputs=[output, reasoning_steps]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )