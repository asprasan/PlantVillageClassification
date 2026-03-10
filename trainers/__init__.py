
TRAINER_REGISTRY = {}

def register_trainer(key: str):
    def decorator(cls):
        TRAINER_REGISTRY[key] = cls
        return cls
    return decorator
