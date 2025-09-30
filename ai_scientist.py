from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class AIScientist:
    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

    def create_agents(self, researcher_prompt, analyst_prompt, writer_prompt):
        """Create AI scientist agents with customizable prompts"""

        # Research Agent
        researcher = Agent(
            role="Research Scientist",
            goal=researcher_prompt["goal"],
            backstory=researcher_prompt["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Data Analyst Agent
        analyst = Agent(
            role="Data Analyst",
            goal=analyst_prompt["goal"],
            backstory=analyst_prompt["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        # Scientific Writer Agent
        writer = Agent(
            role="Scientific Writer",
            goal=writer_prompt["goal"],
            backstory=writer_prompt["backstory"],
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )

        return researcher, analyst, writer

    def create_tasks(self, researcher, analyst, writer, research_topic):
        """Create tasks for the agents"""

        task1 = Task(
            description=f"Conduct comprehensive research on: {research_topic}. "
                       f"Gather relevant information, identify key concepts, and summarize findings.",
            agent=researcher,
            expected_output="A detailed research summary with key findings and insights"
        )

        task2 = Task(
            description=f"Analyze the research findings from the previous task. "
                       f"Identify patterns, trends, and draw meaningful conclusions.",
            agent=analyst,
            expected_output="An analytical report with data-driven insights and conclusions"
        )

        task3 = Task(
            description=f"Write a comprehensive scientific report based on the research and analysis. "
                       f"Include introduction, methodology, findings, and conclusions.",
            agent=writer,
            expected_output="A well-structured scientific report in professional format"
        )

        return [task1, task2, task3]

    def run(self, researcher_prompt, analyst_prompt, writer_prompt, research_topic):
        """Execute the AI scientist crew"""

        # Create agents
        researcher, analyst, writer = self.create_agents(
            researcher_prompt,
            analyst_prompt,
            writer_prompt
        )

        # Create tasks
        tasks = self.create_tasks(researcher, analyst, writer, research_topic)

        # Create crew
        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Execute
        result = crew.kickoff()

        return result


# Default prompts - intentionally very poor quality to demonstrate optimization
DEFAULT_PROMPTS = {
    "researcher": {
        "goal": "Do research",
        "backstory": "You research stuff."
    },
    "analyst": {
        "goal": "Analyze",
        "backstory": "You analyze."
    },
    "writer": {
        "goal": "Write",
        "backstory": "You write."
    }
}

# High-quality prompts for comparison
GOOD_PROMPTS = {
    "researcher": {
        "goal": "Conduct thorough research and gather comprehensive information on assigned topics",
        "backstory": "You are an experienced research scientist with expertise in literature review "
                    "and information synthesis. You excel at finding relevant sources and "
                    "extracting key insights from complex information."
    },
    "analyst": {
        "goal": "Analyze data and research findings to extract meaningful patterns and insights",
        "backstory": "You are a skilled data analyst with strong analytical thinking. "
                    "You can identify trends, patterns, and draw evidence-based conclusions "
                    "from research data."
    },
    "writer": {
        "goal": "Produce clear, well-structured scientific reports and documentation",
        "backstory": "You are a professional scientific writer with years of experience in "
                    "publishing research papers. You excel at communicating complex ideas "
                    "in a clear and engaging manner."
    }
}