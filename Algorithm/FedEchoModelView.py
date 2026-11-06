"""Client-local reversible view augmentation around unchanged local SGD.

The caller constructs FedEcho's canonical start first. The returned state is
canonical again, so response measurement is always y_i - x_i in one coordinate
system. No reset, extra local training, or predictor modification is introduced.
"""
from Algorithm.SymmetryView import apply_horizontal_view_, map_back_state


def train_in_model_view(model, local_training, view):
    if view not in {"I", "F"}:
        raise ValueError("Unknown model view")
    if view == "F":
        apply_horizontal_view_(model)
    result = local_training.train(model)
    return map_back_state(result, model) if view == "F" else result
