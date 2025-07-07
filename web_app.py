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
    """Format reasoning steps as markdown."""
    if not steps:
        return "No reasoning steps recorded."
    
    markdown = "### Reasoning Steps:\n\n"
    for i, step in enumerate(steps, 1):
        summary = step.summary
        markdown += f"{i}. {summary}\n"
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
        with gr.Column():
            query_input = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What was the hottest day in 2024?",
                lines=3
            )
            
            submit_btn = gr.Button("Ask Agent", variant="primary")
            
    with gr.Row():
        with gr.Column(scale=1):
            reasoning_steps = gr.Markdown(
                label="Reasoning Steps",
                value=""
            )
        with gr.Column(scale=2):
            output = gr.Textbox(
                label="Final Answer",
                lines=10,
                max_lines=20
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