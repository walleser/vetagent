import os
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI


from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

tools = [SemanticScholarQueryRun()]


instructions = """You are an expert researcher."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOpenAI(temperature=0,openai_api_key=os.environ["OPENAI_API_KEY"])


from langchain_community.tools.semanticscholar.tool import SemanticScholarQueryRun

from langchain_community.tools import BraveSearch


brave_api = os.environ["BRAVE_SEARCH_API_KEY"]


brave_search = BraveSearch.from_api_key(api_key=brave_api, search_kwargs={"count": 5})

tools = [SemanticScholarQueryRun(),brave_search]

agent = create_openai_functions_agent(llm, tools, prompt)


agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

agent_executor.invoke(
    {
        "input": "What is the correct dosing for pimobendan in a dog that ways 30 lbs? "
        "Break down the task into subtasks for search."
    }
)