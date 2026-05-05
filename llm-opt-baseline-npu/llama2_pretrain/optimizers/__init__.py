_OPTIMIZER_REGISTRY = {}


def register_optimizer(name, cls):
    """Register an optimizer class by name."""
    _OPTIMIZER_REGISTRY[name.lower()] = cls


def get_optimizer(name):
    """Get an optimizer class by name (case-insensitive)."""
    cls = _OPTIMIZER_REGISTRY.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Optimizer '{name}' not found. "
            f"Available: {list(_OPTIMIZER_REGISTRY.keys())}"
        )
    return cls


def list_optimizers():
    """List all registered optimizer names."""
    return sorted(_OPTIMIZER_REGISTRY.keys())
