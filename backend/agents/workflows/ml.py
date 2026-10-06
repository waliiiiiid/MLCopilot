from langgraph.graph import StateGraph, START, END
from backend.agents.workflows.ml_state import ml_state

from backend.agents.data_splitter import splitter
from backend.agents.preprocessing import preprocess_node
from backend.agents.model_selection import model_selection_node
from backend.agents.trainer import train_model
from backend.agents.evaluate import evaluate_model
from backend.agents.tools.real_predict import real_vs_predicted
from backend.agents.tools.confusion import confusion_matrix_node
from backend.agents.tools.roc_auc import roc_auc_node
import pandas as pd



def computaion_route(stat:ml_state):
    if stat['task_type'].lower()=='classification':
        return 'classification'
    return 'regression'



builder = StateGraph(ml_state)

builder.add_node("splitter", splitter)
builder.add_node("preprocess", preprocess_node)
builder.add_node("select", model_selection_node)
builder.add_node("train_model", train_model)
builder.add_node("test_model", evaluate_model)
builder.add_node("real_vs_predicted", real_vs_predicted)
builder.add_node("confusion_matrix", confusion_matrix_node)
builder.add_node("roc_auc", roc_auc_node)





builder.add_edge(START, "splitter")
builder.add_edge("splitter", "preprocess")
builder.add_edge("preprocess", "select")
builder.add_edge("select", "train_model")
builder.add_edge("train_model", "test_model")

builder.add_conditional_edges("test_model",
         computaion_route, {
             'classification': "confusion_matrix",
                'regression': "real_vs_predicted"
}  )

builder.add_edge("confusion_matrix", "roc_auc")
builder.add_edge("roc_auc", END)
builder.add_edge("real_vs_predicted", END)



pipeline = builder.compile()




