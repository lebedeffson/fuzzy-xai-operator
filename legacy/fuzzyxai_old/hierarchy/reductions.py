from .base import FuzzyRepresentation

def reduce_to_f0(obj: FuzzyRepresentation):
    return obj.reduce_to_f0()

def reduction_loss(obj: FuzzyRepresentation) -> float:
    _, delta = obj.reduce_to_f0()
    return float(delta)
