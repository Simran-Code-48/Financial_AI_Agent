import pandas as pd
from langchain_core.tools import Tool
from src.config import FINAL_CATEGORIZED_PATH

def python_code_executor(code: str) -> str:
    """
    Executes Python code with access to pandas as pd and dataframe df.
    The code MUST assign the final output to a variable named `result`.
    """
    df = pd.read_csv(FINAL_CATEGORIZED_PATH)
    
    local_env = {
        "df": df,
        "pd": pd
    }

    try:
        exec(code, {}, local_env)
        if "result" not in local_env:
            return "Error: code did not set a `result` variable."
        return str(local_env["result"])
    except Exception as e:
        return f"Execution error: {e}"

python_code_executor_tool = Tool(
    name="python_code_executor_tool",
    func=python_code_executor,
    description=(
        "Executes Python pandas code on a DataFrame named `df`. "
        "Use this tool for any calculations, filtering, grouping, or aggregation. "
        "The code MUST assign the final value to a variable named `result`."
    ),
)