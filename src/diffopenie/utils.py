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


# ------------------------------ Pretty print functions ------------------------------

RELATION_COLOR = "\033[95m"
SUBJECT_COLOR = "\033[96m"
OBJECT_COLOR = "\033[92m"
RESET = "\033[0m"


def hprint(
    words: list[str],
    subject_ind: list[int],
    object_ind: list[int],
    relation_ind: list[int],
    legend: bool = False,
) -> None:
    """Colourful print for triplets given by word indices"""

    if legend:
        print(f"{SUBJECT_COLOR}Subject{RESET}: {SUBJECT_COLOR}A0{RESET}")
        print(f"{RELATION_COLOR}Relation{RESET}: {RELATION_COLOR}P{RESET}")
        print(f"{OBJECT_COLOR}Object{RESET}: {OBJECT_COLOR}A1{RESET}")

    for i, word in enumerate(words):
        if i in subject_ind:
            print(SUBJECT_COLOR + word + RESET, end=" ")
        elif i in object_ind:
            print(OBJECT_COLOR + word + RESET, end=" ")
        elif i in relation_ind:
            print(RELATION_COLOR + word + RESET, end=" ")
        else:
            print(word, end=" ")
    print()


def hprint_s(
    words: list[str],
    subject_span: tuple[int, int],
    object_span: tuple[int, int],
    relation_span: tuple[int, int],
    legend: bool = False,
) -> None:
    """Colourful print for subject, object, and relation spans"""
    subject_ind = [i for i in range(subject_span[0], subject_span[1] + 1)]
    object_ind = [i for i in range(object_span[0], object_span[1] + 1)]
    relation_ind = [i for i in range(relation_span[0], relation_span[1] + 1)]

    hprint(words, subject_ind, object_ind, relation_ind, legend)
