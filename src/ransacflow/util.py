from pathlib import Path


def get_project_root() -> Path:
    #                 ransacflow    src      /
    return Path(__file__).parent.parent.parent


def get_model_root() -> Path:
    return get_project_root() / "models"


def get_data_root() -> Path:
    return get_project_root() / "data"

