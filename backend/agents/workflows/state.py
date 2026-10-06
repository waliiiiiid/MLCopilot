from typing import TypedDict, NotRequired


class first_state(TypedDict):
    file_path: str
    profile: str
    plan: str
    code: str
    execution_success: bool
    execution_stdout: NotRequired[str]
    execution_stderr: NotRequired[str]
    execution_return_code: NotRequired[int]


class second_state(TypedDict):
    file_path: str
    profile: str
    plan: str
    code: str
    execution_success: bool
    execution_stdout: NotRequired[str]
    execution_stderr: NotRequired[str]
    execution_return_code: NotRequired[int]

