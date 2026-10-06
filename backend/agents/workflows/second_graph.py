from langgraph.graph import StateGraph, START, END
from backend.agents.profiler import profiler_node
from backend.agents.eda_planner import eda_planner_node
from backend.agents.eda_code import eda_code_node
from backend.agents.docker_executer import docker_executor_node
from backend.agents.workflows.state import second_state

builder = StateGraph(second_state)

builder.add_node('profiler', profiler_node)
builder.add_node('planner', eda_planner_node)
builder.add_node('code_generator', eda_code_node)
builder.add_node('docker', docker_executor_node)

builder.add_edge(START, 'profiler')
builder.add_edge('profiler', 'planner')
builder.add_edge('planner', 'code_generator')
builder.add_edge('code_generator', 'docker')
builder.add_edge('docker', END)

step_2 = builder.compile()


