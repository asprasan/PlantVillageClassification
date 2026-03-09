
MODEL_REGISTRY = {}

def register_model(key: str):
    def decorator(cls):
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator
