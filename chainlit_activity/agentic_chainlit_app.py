import os
from io import StringIO
from dotenv import load_dotenv
import chainlit as cl
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from google import genai
from crewai import LLM
import time
import asyncio
from contextlib import redirect_stdout

# Load environment variables from .env file
load_dotenv()

# Configure the API keys
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

# Set up the LLM
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
llm = LLM(model="gemini/gemini-2.0-flash")

# Set up SerperDevTool for web search
search_tool = SerperDevTool()

class OutputCapture:
    """
    Captures stdout output for monitoring CrewAI's verbose output.
    Enhanced to preserve all output formatting and details.
    """
    def __init__(self):
        self.buffer = StringIO()
        self.output = []
        self.last_position = 0
        
    def capture_output(self):
        """Start capturing stdout."""
        self.buffer = StringIO()
        return redirect_stdout(self.buffer)
    
    def get_new_output(self):
        """Get any new output from stdout and store it."""
        value = self.buffer.getvalue()
        if value and len(value) > self.last_position:
            new_output = value[self.last_position:]
            self.output.append(new_output)
            self.last_position = len(value)
            return new_output
        return ""
    
    def get_all_output(self):
        """Get all captured output including any remaining content in buffer."""
        final_value = self.buffer.getvalue()
        if final_value and len(final_value) > self.last_position:
            self.output.append(final_value[self.last_position:])
        return "".join(self.output)
    
    def clear(self):
        """Clear all captured output."""
        self.output = []
        self.buffer = StringIO()
        self.last_position = 0

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the chat session and display welcome message with instructions.
    """
    # Initialize session data
    cl.user_session.set("update_tasks", {})
    
    # Send initial welcome message
    await cl.Message(
        content="# 👋 Welcome to the AI Research Assistant!\n\n"
                "This assistant uses a team of AI agents with web search capabilities to research topics for you.\n\n"
                "**How to use:**\n"
                "1. Enter a topic or question you want to research\n"
                "2. Watch as the Research Agent gathers information and the Proofreader Agent enhances it\n"
                "3. Review the comprehensive analysis and click on the collapsible sections to see the full agent thought process\n\n"
                "Let's get started!",
    ).send()

def create_research_agent() -> Agent:
    """
    Create a research agent with web search capabilities.
    
    Returns:
        Agent: The configured research agent
    """
    research_agent = Agent(
        role="Web Research Specialist",
        goal="Find the most up-to-date information on any topic using web search",
        backstory="You are an expert researcher with a talent for finding accurate information online. "
                "You have a background in information science and know how to identify reliable sources. "
                "You're meticulous and always cite your sources.",
        verbose=True,
        allow_delegation=False,
        tools=[search_tool],
        llm=llm,
        max_iterations=8,  # Allow more iterations for thorough research
        max_rpm=10,  # Limit rate for API calls
    )
    return research_agent

def execute_research(query: str, output_capture: OutputCapture) -> tuple:
    """
    Execute research and proofreading tasks using a single CrewAI crew and capture verbose output.
    
    Args:
        query (str): The user's research query
        output_capture (OutputCapture): Object to capture stdout
        
    Returns:
        tuple: A tuple containing (research_result, proofread_result, output_text)
    """
    # Create the research agent
    research_agent = create_research_agent()
    
    # Create the proofreader agent
    proofreader_agent = create_proofreader_agent()
    
    # Create the research task
    research_task = Task(
        description=f"Research and provide a comprehensive analysis on '{query}'. "
                   f"Focus on recent developments, key facts, various perspectives, and credible sources. "
                   f"Use web search to find the most up-to-date and accurate information.",
        expected_output="A comprehensive analysis with key points, insights, and source references.",
        agent=research_agent,
    )
    
    # Create the proofreading task that will use the output from the research task
    proofread_task = Task(
        description=f"Review and enhance the research results on '{query}' provided by the researcher. "
                   f"Check for accuracy, clarity, coherence, and completeness. "
                   f"Improve the language and organization where needed. "
                   f"Ensure the content directly addresses the original query.",
        expected_output="An enhanced version of the research results with improved clarity, "
                       "organization, and accuracy, along with a brief summary of improvements made.",
        agent=proofreader_agent,
        context=[research_task]  # This links the proofreading task to the research task
    )
    
    # Create a single crew with both agents and both tasks
    crew = Crew(
        agents=[research_agent, proofreader_agent],
        tasks=[research_task, proofread_task],
        verbose=True,
        process=Process.sequential  # Tasks will be executed in sequence
    )
    
    # Capture the output during execution
    with output_capture.capture_output():
        # Execute the crew with both tasks
        results = crew.kickoff()
        
    # Get any remaining output
    output_capture.get_new_output()
    
    # Parse the results - with sequential tasks, results will be the output of the last task
    # We need to extract both the research and proofread results
    research_result = research_task.output  # This gets the output of the research task
    proofread_result = results  # This is the output of the last task (proofreading)
    
    # Return both results and the full output
    return research_result, proofread_result, output_capture.get_all_output()

def create_proofreader_agent() -> Agent:
    """
    Create a proofreader agent to review and improve research results.
    
    Returns:
        Agent: The configured proofreader agent
    """
    proofreader_agent = Agent(
        role="Content Proofreader and Enhancer",
        goal="Improve the quality, clarity, and accuracy of research content",
        backstory="You are an expert editor with decades of experience improving research documents. "
                "You have a keen eye for detail, excellent language skills, and know how to make "
                "complex information accessible without oversimplifying. You can identify gaps "
                "in research and suggest improvements.",
        verbose=True,
        allow_delegation=True,  # Allow the proofreader to delegate back to researcher if needed
        tools=[search_tool],  # Give proofreader search capability for fact-checking
        llm=llm,
        max_iterations=5,
    )
    return proofreader_agent

def create_proofreader_task(query: str, research_result: str) -> Task:
    """
    Create a proofreading task based on the research results.
    
    Args:
        query (str): The original research query
        research_result (str): The results from the research agent
        
    Returns:
        Task: The configured proofreading task
    """
    proofreader_agent = create_proofreader_agent()
    
    proofreader_task = Task(
        description=f"Review and enhance the research results on '{query}'. "
                    f"Check for accuracy, clarity, coherence, and completeness. "
                    f"Improve the language and organization where needed. "
                    f"Ensure the content directly addresses the original query. "
                    f"Research result to review: {research_result}",
        expected_output="An enhanced version of the research results with improved clarity, "
                       "organization, and accuracy, along with a brief summary of improvements made.",
        agent=proofreader_agent,
    )
    
    return proofreader_task

@cl.on_message
async def main(message: cl.Message):
    """
    Process user messages and respond with research results.
    Display verbose output in real-time.
    
    Args:
        message (cl.Message): The user's message
    """
    query = message.content
    
    # Create initial progress message
    progress_msg = cl.Message(
        content="🔍 Starting research on your topic...\n\n```\nInitializing AI research agent...\n```"
    )
    await progress_msg.send()
    
    try:
        # Create output capture object
        output_capture = OutputCapture()
        
        # Set up periodic updates
        async def update_progress():
            last_update = ""
            while True:
                new_output = output_capture.get_new_output()
                if new_output:
                    # Format and clean the output for better readability
                    formatted_output = new_output.replace("\r", "\n")
                    
                    # Enhance formatting for better readability with agent identification
                    if "web research specialist" in formatted_output.lower() or "researcher" in formatted_output.lower():
                        formatted_output = f"🔎 [RESEARCHER] {formatted_output}"
                    elif "proofreader" in formatted_output.lower() or "editor" in formatted_output.lower():
                        formatted_output = f"✏️ [PROOFREADER] {formatted_output}"
                    elif "task:" in formatted_output.lower():
                        formatted_output = f"🔄 {formatted_output}"
                    elif "agent:" in formatted_output.lower():
                        formatted_output = f"🤖 {formatted_output}"
                    elif "thinking:" in formatted_output.lower():
                        formatted_output = f"💭 {formatted_output}"
                    elif "searching" in formatted_output.lower():
                        formatted_output = f"🔍 {formatted_output}"
                    
                    last_update += formatted_output
                    
                    # Keep the output log to a reasonable length but preserve more content
                    if len(last_update) > 8000:
                        last_update = "...\n" + last_update[-8000:]
                    
                    # Extract and highlight important parts of the output
                    highlighted_output = last_update
                    
                    # Update the progress message with the latest output
                    # Include context about which agent is working and what they're doing
                    
                    # Determine which agent is currently active
                    current_agent = "researcher"
                    if "proofreader" in highlighted_output.lower() or "content proofreader and enhancer" in highlighted_output.lower():
                        current_agent = "proofreader"
                    
                    # Set status based on the current activity and agent
                    if current_agent == "researcher":
                        if "thinking:" in highlighted_output.lower():
                            status = "💭 Research Agent is analyzing the query..."
                        elif "searching" in highlighted_output.lower():
                            status = "🔍 Research Agent is searching the web for information..."
                        elif "task:" in highlighted_output.lower():
                            status = "🔄 Research Agent is working on gathering information..."
                        elif "agent:" in highlighted_output.lower():
                            status = "🔎 Research Agent is processing search results..."
                        else:
                            status = "⚙️ Research Agent is working..."
                    else:  # proofreader
                        if "thinking:" in highlighted_output.lower():
                            status = "💭 Proofreading Agent is reviewing the research..."
                        elif "searching" in highlighted_output.lower():
                            status = "🔍 Proofreading Agent is fact-checking information..."
                        elif "task:" in highlighted_output.lower():
                            status = "🔄 Proofreading Agent is enhancing the content..."
                        elif "agent:" in highlighted_output.lower():
                            status = "✏️ Proofreading Agent is improving the research..."
                        else:
                            status = "📝 Proofreading Agent is working..."
                        
                    progress_msg.content = f"**{status}**\n\nResearching: {query}\n\n```\n{highlighted_output}\n```"
                    await progress_msg.send()
                
                await cl.sleep(0.5)
        
        # Start the update task
        update_task = asyncio.create_task(update_progress())
        task_id = f"update_task_{int(time.time())}"
        
        # Store the task in user session
        update_tasks = cl.user_session.get("update_tasks", {})
        update_tasks[task_id] = update_task
        cl.user_session.set("update_tasks", update_tasks)
        
        # Execute the research with proofreading
        research_result, proofread_result, _ = await cl.make_async(execute_research)(query, output_capture)
        
        # Cancel and remove the update task
        update_tasks = cl.user_session.get("update_tasks", {})
        if task_id in update_tasks:
            update_task = update_tasks[task_id]
            update_task.cancel()
            try:
                await update_task
            except asyncio.CancelledError:
                pass
            
            # Remove the task from the dictionary
            del update_tasks[task_id]
            cl.user_session.set("update_tasks", update_tasks)
        
        # Show final result
        all_output = output_capture.get_all_output()
        
        # Format the output for better readability
        formatted_output = all_output.replace("\r", "\n")
        
        # Create a completion message
        completion_msg = cl.Message(
            content="✅ **Research and Proofreading Completed!**\n\nAgent thought processes are available below (click to expand):"
        )
        await completion_msg.send()
        
        # Split the logs into sections by agent for better organization
        # Check for the transition marker between research and proofreading
        transition_marker = "==== RESEARCH COMPLETED. PASSING TO PROOFREADER ===="
        
        # Split by the transition marker if present
        if transition_marker in formatted_output:
            parts = formatted_output.split(transition_marker)
            researcher_log = parts[0]
            proofreader_log = parts[1] if len(parts) > 1 else ""
            
            # Researcher logs with collapsible element
            researcher_msg = cl.Message(
                content="🔎 **Research Agent Process**",
                elements=[
                    cl.Text(name="researcher_log", content=researcher_log, display="inline"),
                ]
            )
            await researcher_msg.send()
            
            # Proofreader logs with collapsible element
            proofreader_msg = cl.Message(
                content="✏️ **Proofreader Agent Process**",
                elements=[
                    cl.Text(name="proofreader_log", content=proofreader_log, display="inline"),
                ]
            )
            await proofreader_msg.send()
        else:
            # If we can't split cleanly, just show the entire log
            # Try to identify agent sections based on common patterns in the output
            enhanced_log = formatted_output
            # Add visual separators for agent transitions we can detect
            enhanced_log = enhanced_log.replace("Starting task for Web Research Specialist", 
                                              "🔍 ====== RESEARCH AGENT STARTING ======\nStarting task for Web Research Specialist")
            enhanced_log = enhanced_log.replace("Starting task for Content Proofreader and Enhancer", 
                                              "✏️ ====== PROOFREADING AGENT STARTING ======\nStarting task for Content Proofreader and Enhancer")
            
            # Create a collapsible element with the full log
            full_log_msg = cl.Message(
                content="� **Full Agent Process Log**",
                elements=[
                    cl.Text(name="full_agent_log", content=enhanced_log, display="inline"),
                ]
            )
            await full_log_msg.send()
        
        # Format the research result
        formatted_research = f"# Initial Research Results: {query}\n\n{research_result}"
        
        # Send the research result
        await cl.Message(content=formatted_research).send()
        
        # Format the research result
        formatted_research = f"# Initial Research Results: {query}\n\n{research_result}"
        
        # Send the research result
        await cl.Message(content=formatted_research).send()
        
        # Format the proofread result
        formatted_proofread = f"# Enhanced Research by Proofreader Agent: {query}\n\n{proofread_result}"
        
        # Send the proofread result as the final message
        await cl.Message(content=formatted_proofread).send()
        
        # Comparison message
        comparison_msg = cl.Message(
            content="✅ **Process Complete!**\n\nYou can see above how the agents worked in a single crew: first the researcher collected information, then passed it to the proofreader who enhanced and improved it."
        )
        await comparison_msg.send()
        
    except Exception as exc:
        # Handle any errors with detailed error message
        error_message = f"❌ An error occurred: {str(exc)}"
        progress_msg.content = error_message
        await progress_msg.send()
        await cl.Message(content="Please try again with a different query or check your API keys.").send()
        
        # Clean up any running tasks
        update_tasks = cl.user_session.get("update_tasks", {})
        for tid, task in list(update_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del update_tasks[tid]
        cl.user_session.set("update_tasks", {})

@cl.on_chat_end
async def on_chat_end():
    """
    Clean up resources when the chat session ends.
    """
    # Cancel any running update tasks
    update_tasks = cl.user_session.get("update_tasks", {})
    for task_id, task in update_tasks.items():
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    # Clear the tasks dictionary
    cl.user_session.set("update_tasks", {})

if __name__ == "__main__":
    # This is used when running locally
    pass
