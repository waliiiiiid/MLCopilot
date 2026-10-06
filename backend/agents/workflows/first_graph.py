from langgraph.graph import StateGraph, START, END
from backend.agents.profiler import profiler_node
from backend.agents.visualizationplanner import visual_plan_node
from backend.agents.visual_code import visual_code_node
from backend.agents.docker_executer import docker_executor_node
from backend.agents.workflows.state import first_state

builder = StateGraph(first_state)

builder.add_node('profiler', profiler_node)
builder.add_node('planner', visual_plan_node)
builder.add_node('code_generator', visual_code_node)
builder.add_node('docker', docker_executor_node)

builder.add_edge(START, 'profiler')
builder.add_edge('profiler', 'planner')
builder.add_edge('planner', 'code_generator')
builder.add_edge('code_generator', 'docker')
builder.add_edge('docker', END)


step_1 = builder.compile()



