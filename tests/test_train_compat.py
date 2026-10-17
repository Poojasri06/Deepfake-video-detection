"""src/model.py must stay identical to the model class in the original src/train.py,
because train.py is the code that produced the checkpoint."""
import ast
from pathlib import Path

import torch.nn as nn
from torchvision import models

from src.model import DeepfakeDetector

TRAIN = Path(__file__).resolve().parents[1] / "src" / "train.py"


def _class_from_train_py():
    tree = ast.parse(TRAIN.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "DeepfakeDetector")
    ns = {"nn": nn, "models": type("M", (), {"efficientnet_b0": staticmethod(
        lambda weights=None: models.efficientnet_b0(weights=None))})}   # avoid the ImageNet download
    exec(compile(ast.Module([cls], []), str(TRAIN), "exec"), ns)
    return ns["DeepfakeDetector"]


def test_same_parameter_names_and_shapes_as_train_py():
    old, new = _class_from_train_py()().state_dict(), DeepfakeDetector(pretrained=False).state_dict()
    assert list(old) == list(new)
    assert all(old[k].shape == new[k].shape for k in old)


def test_train_py_state_dict_loads_strictly():
    DeepfakeDetector(pretrained=False).load_state_dict(_class_from_train_py()().state_dict(), strict=True)


def test_training_preprocessing_constants_match_inference_defaults():
    """Defaults in train.py CONFIG that inference must mirror."""
    from src.config import InferenceConfig
    tree = ast.parse(TRAIN.read_text())
    cfg = next(n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CONFIG")
    conf = ast.literal_eval(cfg.value)
    d = InferenceConfig(checkpoint_path="x")
    assert (conf["frames_per_clip"], conf["stride"], conf["image_size"]) == (
        d.frames_per_clip, d.stride, d.image_size)
