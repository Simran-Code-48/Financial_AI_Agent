from langchain_cohere import ChatCohere
from langchain.agents import create_agent
from rich import print as rprint
from .tools import python_code_executor_tool
from .system_prompt import SYSTEM_PROMPT
from src.config import COHERE_API_KEY

def run_agent():
    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0.4,
        cohere_api_key=COHERE_API_KEY
    )

    agent = create_agent(
        model=llm,
        tools=[python_code_executor_tool],
        system_prompt=SYSTEM_PROMPT,
    )

    while True:
        print("="*20+"User"+'='*20)
        user_input = input('Enter query or q to quit : ')
        if user_input == 'q':
            break
        print("="*20+"AI"+'='*20)

        response = agent.invoke(
            {"messages":[{"role":"user","content":user_input}]},
            {"configurable":{"thread_id":"session1"}},
        )

        rprint(response['messages'][-1].content)

if __name__ == "__main__":
    run_agent()