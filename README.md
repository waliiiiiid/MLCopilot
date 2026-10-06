# ML Agent Pipeline (Titanic EDA/Visualization Agents)

An agentic pipeline (LangChain + LangGraph, Groq-hosted LLMs) that profiles a
dataset and then either plans+generates+executes **visualization** code or
**cleaning** code inside a Docker sandbox.

The visualization graph (`step_1`, in `first_graph.py`) and the cleaning graph
(`step_2`, in `second_graph.py`) are kept as two separate LangGraph graphs on
purpose, since combining both planning stages in a single agent context
exceeded the LLM's message length limit.

## Setup

1. Install dependencies (uv or pip):
   ```
   uv sync
   # or
   pip install -r requirements.txt
   ```
2. Copy your real API keys into `.env` (placeholders are checked in — do not
   commit real keys):
   ```
   GROQ_API_KEY='...'
   GOOGLE_API_KEY='...'   # currently unused by the code, safe to leave blank
   ```
3. Build the sandbox image the code executor runs generated scripts in:
   ```
   docker build -t ml-agent-sandbox -f docker/dockerfile .
   ```
4. Run the pipeline from the project root:
   ```
   python -m backend.main
   ```
   Output figures / cleaned dataset / cleaning report land in `outputs/`.

## What was fixed

- `backend/agents/workflows/state.py` — removed a dangling, unused
  `third_state` class, and added `execution_stdout` / `execution_stderr` /
  `execution_return_code` to the state schemas. LangGraph silently drops any
  key a node returns that isn't declared in the state's TypedDict, so
  stdout/stderr from the sandbox were being thrown away, making failures
  impossible to debug.
- `backend/agents/workflows/third_graph.py` — deleted. It imported a `State`
  class that doesn't exist in `state.py`, and its node/edge wiring didn't
  match either the visualization or cleaning pipeline (missing a `START`
  edge, missing the planner/code-generator nodes), so it could never compile.
  It wasn't referenced anywhere else in the project. If you want a single
  combined pipeline (clean, then visualize the cleaned data) let me know and
  I can build that as a proper third graph.
- `backend/agents/docker_executer.py`:
  - Now passes `DATASET_EXT` (derived from the input file's extension) into
    the container as an environment variable, since the generated cleaning
    script (`eda_code.py`'s prompt) reads `os.environ["DATASET_EXT"]` but the
    old `docker run` command never set it — every cleaning run would crash
    with a `KeyError` inside the sandbox.
  - Strips accidental ```` ```python ... ``` ```` fences from LLM-generated
    code before writing it to disk, since models occasionally ignore the
    "no markdown" instruction and that alone was enough to break execution
    with a `SyntaxError`.
- `backend/agents/visual_code.py` — the prompt told the model to detect
  CSV vs. Parquet from the file extension of `/data/dataset`, but that mount
  path has no extension. It now uses the same `DATASET_EXT` environment
  variable as the cleaning pipeline for reliable format detection.
- `backend/main.py` — now prints `execution_stdout`/`execution_stderr` when a
  run fails, and runs both pipelines (previously step_2 was commented out).
- Removed the committed `.venv/` and `uv.lock` from the archive (regenerate
  with `uv sync`) and stale `__pycache__` files that referenced modules that
  no longer exist (`planner.py`, `workflow.py`, `data_agent.py`, `tools.py`).

## ⚠️ Rotate your API keys

The uploaded project's `.env` contained **live** `GROQ_API_KEY` and
`GOOGLE_API_KEY` values. I removed them from this archive and replaced them
with placeholders, but since those keys were shared in this conversation you
should treat them as compromised and rotate/revoke them from the Groq and
Google AI consoles now.
