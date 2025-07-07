import logging
import re

from jinja2 import StrictUndefined, Template
from litellm import completion
from smolagents.local_python_executor import LocalPythonExecutor

from .prompts import SYSTEM_PROMPT
from .tools import FinalAnswerTool

BASE_BUILTIN_MODULES = [
    "collections",
    "datetime",
    "itertools",
    "math",
    "queue",
    "random",
    "re",
    "stat",
    "statistics",
    "time",
    "unicodedata",
]


# Import and define logger using standard library
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        model,
        tools=None,
        authorized_imports=None,
        max_steps=10,
    ):
        self.model = model
        self.max_steps = max_steps

        # 1. Initialize Python executor:

        # 1.1. Initialize tools: The agent always needs the FinalAnswerTool.
        # This tool is used by the agent to signal that it has found the final
        # answer, as described in it's system prompt (see below).
        self.tools = {tool.name: tool for tool in tools + [FinalAnswerTool()]}

        # 1.2. Described the imports that the agent is allowed to use in it's code
        # execution. This is to ensure that the agent doesn't execute potentially
        # dangerous code - e.g. code that deletes all files.
        self.authorized_imports = authorized_imports or BASE_BUILTIN_MODULES

        # 1.3 Initialize the Local Python Executor
        self.python_executor = LocalPythonExecutor(
            additional_authorized_imports=[],
        )
        self.python_executor.send_tools(self.tools)

        # 2. Initialize system prompt and history management

        # Initialize the system prompt to describe how the agent should solve approach
        # solving the task and how it can use tools. It's important to read this
        # system prompt to understand how the agent works.
        self.system_prompt = self.initialize_system_prompt(SYSTEM_PROMPT)

        # Track all the thoughts and observations by the agent in a list.
        # Initially it only contains the system prompt.
        self.history = [{"role": "system", "content": self.system_prompt}]
        
        # Track reasoning steps (summaries) for display
        self.reasoning_steps = []

    def _extract_python_code(self, text: str) -> None | str:
        pattern = r"```py([\s\S]*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    
    def _extract_summary(self, text: str) -> str:
        """Extract the summary from the thought text."""
        pattern = r"Summary:\s*(.+?)(?:\n|$)"
        match = re.search(pattern, text)
        assert match, "Summary not found in thought text"
        return match.group(1).strip()

    def initialize_system_prompt(self, system_prompt_template: str) -> str:
        """Initialize the system prompt from a template. Most of the system
        prompt is just general instructions that are the same irrespective of the agent.
        But the system prompt also needs to include the tools and authorized imports that
        the agent can use in its code. For this, a template using the 'jinja2' library
        is helpful.

        Args:
            system_prompt_template (str): System prompt template with placeholders for tools and authorized imports.

        Returns:
            str: system prompt for the agent
        """
        compiled_template = Template(system_prompt_template, undefined=StrictUndefined)
        variables = {
            "tools": self.tools,
            "authorized_imports": str(self.authorized_imports),
        }
        return compiled_template.render(**variables)

    def run(self, task: str) -> str:
        """Run the task and return the result."""
        # Use run_streaming and just get the final result
        for update in self.run_streaming(task):
            if update["final_answer"]:
                return update["final_answer"]
        
        # This should never be reached, but just in case
        return "Could not solve task: Unknown error."
    
    def run_streaming(self, task: str):
        """Run the task and yield steps as they happen."""
        # Reset reasoning steps for new task
        self.reasoning_steps = []
        
        # Append the task to the history so that the agent can
        # pick it up.
        self.history.append({"role": "user", "content": f"Task: {task}"})

        # Create a loop of thoughts (with code creation), code execution and
        # observation until the task is solved.
        task_completed = False
        nr_steps = 0
        while not task_completed or nr_steps <= self.max_steps:
            logger.info(f"!STEP!: {nr_steps}")
            is_final_answer, observation, output = self.step(self.history)
            logger.info(f"!Observation!: {observation['content']}")
            self.history.append(observation)
            logging.debug(f"!Last History entry! f{self.history[-1]}")
            nr_steps += 1
            
            # Yield the current state
            if is_final_answer:
                yield {"final_answer": output, "steps": self.reasoning_steps}
                return
            else:
                yield {"final_answer": None, "steps": self.reasoning_steps}
                
        yield {"final_answer": "Could not solve task: Maximum number of steps exceeded.", "steps": self.reasoning_steps}

    def step(self, history: list) -> list:
        """Implement the logic for each step of the agent's decision-making process.

        Args:
            history (list): History of the agent so far in message format.

        Returns:
            is_final_answer (bool): Whether or not this is the final answer of the agent, i.e.
            if the executed code contained the 'final_answer_tool'.
            observation (dict): The observation that the agent made in this step in message
            format such that it can be added to the history
            output (str, None): Contains the final answer if is_final_answer is True, otherwise None.
        """

        # 1. Generate thought
        response = completion(
            model=self.model, messages=history, stream=False, stop="<end_code>"
        )
        thought = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": thought})
        logging.info(f"!Thought!: {thought}")
        
        # Extract and store the summary for this step
        summary = self._extract_summary(thought)
        self.reasoning_steps.append(summary)

        code_action = self._extract_python_code(thought)
        logging.info(f"!Code action!: {code_action}")

        # 2. Execute code
        output, execution_logs, is_final_answer = self.python_executor(
            code_action=code_action
        )

        # 3. Create observation that can be added to the history.
        # Note that the observation has role 'human'. This ensures that
        # the LLM reacts to it in the next step. Think of it as follows:
        # The agent asked the user to execute some code. This is now done and the
        # resulting observation is now handed back to the LLM by the user.
        observation = {"role": "user", "content": "Observation:\n" + execution_logs}
        return is_final_answer, observation, output
