from typing import Any


def config_as_token(in_dict: dict[Any, Any], curr_tabs: int = 0) -> str:
    token = "\n"
    for k, v in in_dict.items():
        token += "\t" * curr_tabs
        token += f"{k}: "
        if isinstance(v, dict):
            token += config_as_token(v, curr_tabs + 1)
        else:
            token += f"{v}"
        token += "\n"
    return token
