from typing import TypedDict

class ml_state(TypedDict):
    dataset_path:str
    train_path:str
    test_path:str
    train_shape:tuple
    test_shape:tuple
    pre_process:str
    base_model:str
    metric:str
    metric_value:float
    task_type:str
    target_column:str
    report:str
    r2:float
    y_proba:list
    y_pred:list
    y_real:list
    