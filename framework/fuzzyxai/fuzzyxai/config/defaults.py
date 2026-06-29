"""Default settings for the reproducible doctoral demo layer."""

DEFAULT_NUMERIC_TERMS = ("low", "medium", "high")
DEFAULT_ACTIVATION_THRESHOLD = 0.05
DEFAULT_I_MIN = 0.50
DEFAULT_INTERVAL_EPSILON = 0.05
DEFAULT_RANDOM_STATE = 42

DEFAULT_BETA = {
    "repr": 0.30,
    "rules": 0.25,
    "activations": 0.15,
    "uncertainty": 0.20,
    "trace": 0.10,
}

DEFAULT_LAMBDA = {
    "H": 0.20,
    "C": 0.20,
    "O": 0.20,
    "K": 0.20,
    "U": 0.20,
}

DEFAULT_ETA = {
    "model": 0.50,
    "rules": 0.30,
    "trace": 0.20,
}
