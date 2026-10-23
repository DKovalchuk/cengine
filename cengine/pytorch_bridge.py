"""Rebuild torch modules from cengine ``.pt`` checkpoints.

The bridge closes the CPU <-> CUDA loop:

1. train with cengine on the CPU;
2. ``trainer.save_model_pytorch(path)`` — a ``.pt`` with a ``state_dict``
   plus ``metadata["architecture"]`` (layer types and sizes);
3. ``load_model_torch(path)`` — rebuilds the equivalent ``torch.nn.Module``
   and loads the weights, ready for ``model.to("cuda")`` and regular
   torch training;
4. ``trainer.load_model_pytorch(path)`` — hand the (possibly updated)
   weights back to cengine.

Requires ``torch`` (imported lazily so cengine itself stays torch-free).
"""
from __future__ import annotations

from typing import Any

__all__ = ["build_torch_model", "load_model_torch"]


def _build_layer(spec: dict[str, Any]):
    import torch.nn as nn

    kind = spec.get("type")
    if kind == "linear":
        return nn.Linear(spec["in_features"], spec["out_features"])
    if kind == "elu":
        return nn.ELU(spec.get("alpha", 1.0))
    if kind == "sigmoid":
        return nn.Sigmoid()
    if kind == "tanh":
        return nn.Tanh()
    if kind == "softmax":
        return nn.Softmax(dim=-1)
    if kind == "gumbel_softmax":
        return nn.GumbelSoftmax(temperature=spec.get("temperature", 1.0), dim=-1)
    if kind == "recurrent_expert":
        raise NotImplementedError(
            "recurrent_expert has no generic torch equivalent — build your "
            "recurrent module manually (the parameter keys w1/b1/... match "
            "the C registration order) and load the state_dict into it"
        )
    raise ValueError(f"unknown layer type in architecture: {kind!r}")


def build_torch_model(architecture: Any):
    """Build a torch module from an ``architecture`` description.

    ``architecture`` is what ``Module.architecture()`` returns and what
    ``save_model_pytorch`` stores in ``metadata["architecture"]``: a list
    of layer dicts for a Sequential stack, a single dict for one layer.

    Raises ``NotImplementedError`` for layer kinds that have no faithful
    torch equivalent and ``ValueError`` for unknown kinds — never a
    silently different model.
    """
    import torch.nn as nn

    if isinstance(architecture, list):
        return nn.Sequential(*[_build_layer(spec) for spec in architecture])
    return _build_layer(architecture)


def load_model_torch(path: str):
    """Rebuild the torch module from a cengine ``.pt`` and load its weights.

    ``load_state_dict`` runs in strict mode: a key or shape mismatch raises
    with nothing partially applied.  Checkpoints written by older cengine
    versions (no ``metadata["architecture"]``) are rejected with a clear
    message instead of a guessed layout.
    """
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    meta = checkpoint.get("metadata", {})
    architecture = meta.get("architecture")
    if architecture is None:
        raise ValueError(
            f"{path}: checkpoint has no metadata['architecture'] — re-save "
            "it with save_model_pytorch() (older cengine versions did not "
            "record the layer layout)"
        )

    model = build_torch_model(architecture)
    model.load_state_dict(state_dict)
    return model
