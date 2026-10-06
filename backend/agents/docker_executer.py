import os
import subprocess
import tempfile


def _strip_code_fences(code: str) -> str:
    """LLMs sometimes wrap code in ```python ... ``` fences despite being told
    not to. Strip that off defensively so the sandbox always gets valid Python."""
    text = code.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        elif lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def docker_executor_node(state):

    code = _strip_code_fences(state["code"])
    dataset_path = state["file_path"]

    dataset_ext = os.path.splitext(dataset_path)[1] or ".csv"

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

                    "-v",
                    f"{workspace_dir}:/workspace",

                    "-v",
                    f"{os.path.abspath(dataset_path)}:/data/dataset",

                    "-v",
                    f"{output_dir}:/outputs",

                    "-e",
                    f"DATASET_EXT={dataset_ext}",

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
                "execution_stdout": result.stdout,
                "execution_stderr": result.stderr,
                "execution_return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:

            return {
                "execution_success": False,
                "execution_stdout": "",
                "execution_stderr": "Execution timed out after 60 seconds.",
                "execution_return_code": -1,
            }

        except Exception as e:

            return {
                "execution_success": False,
                "execution_stdout": "",
                "execution_stderr": str(e),
                "execution_return_code": -1,
            }
