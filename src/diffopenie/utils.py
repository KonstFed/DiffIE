from pathlib import Path

import yaml
from pydantic import BaseModel


def load_config(cls: type[BaseModel], cfg_p: Path | str) -> BaseModel:
    cfg_p = Path(cfg_p)
    with cfg_p.open("r") as f:
        data = yaml.safe_load(f)

    return cls.model_validate(data)


def save_config(model: BaseModel, cfg_p: Path | str) -> None:
    """Save a Pydantic BaseModel instance to a YAML file.

    Args:
        model: The Pydantic model instance to save.
        cfg_p: The path to the YAML file where the model will be saved.

    """
    cfg_p = Path(cfg_p)
    # Convert the model to a dictionary
    model_dict = model.model_dump(mode="json")

    # Write the dictionary to a YAML file
    with cfg_p.open("w") as f:
        yaml.safe_dump(model_dict, f, default_flow_style=False, sort_keys=False)