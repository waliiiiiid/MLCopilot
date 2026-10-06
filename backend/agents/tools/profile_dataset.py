from langchain_core.tools import tool
import pandas as pd

@tool
def profile_dataset(file_path: str):
    """
    Analyze a tabular dataset and return information
    about its structure, columns, data types, missing
    values, duplicates, and feature types.
    """

    df = pd.read_csv(file_path.lstrip("/"))

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "numerical_columns": df.select_dtypes(include="number").columns.tolist(),
        "categorical_columns": df.select_dtypes(
            include=["str", "object"]
        ).columns.tolist(),
        'description':df.describe().to_dict()
    }

