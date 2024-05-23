from inspect import isfunction
MODEL_REGISTRY = {}


def register(name):
    def decorator(model_class):
        MODEL_REGISTRY[name] = model_class
        return model_class

    return decorator

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d