import os
import subprocess
import tempfile

from langchain_core.tools import tool


@tool
def docker_executor(code: str):
    """
    Execute Python code inside the Docker sandbox.
    The cleaned dataset is available at:
    /outputs/cleaned_dataset.csv

    Generated files should be saved to:
    /outputs/
    """

    output_dir = os.path.abspath("outputs")
    os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:

        workspace_dir = os.path.join(tmpdir, "workspace")
        os.makedirs(workspace_dir, exist_ok=True)

        code_path = os.path.join(workspace_dir, "main.py")

        with open(code_path, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",

                    # Generated Python code
                    "-v",
                    f"{workspace_dir}:/workspace",

                    # Cleaned dataset + output files
                    "-v",
                    f"{output_dir}:/outputs",

                    # Run generated code
                    "ml-agent-sandbox",
                    "python",
                    "/workspace/main.py",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            return {
                "execution_success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "execution_success": False,
                "stdout": "",
                "stderr": "Execution timed out after 60 seconds.",
                "return_code": -1,
            }

        except Exception as e:
            return {
                "execution_success": False,
                "stdout": "",
                "stderr": str(e),
                "return_code": -1,
            }