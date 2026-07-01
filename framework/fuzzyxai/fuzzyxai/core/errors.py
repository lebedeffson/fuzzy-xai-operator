class FuzzyXAIError(Exception):
    """Base framework error."""


class AdapterValidationError(FuzzyXAIError):
    """Raised when an external model payload cannot be adapted."""


class ProofVerificationError(FuzzyXAIError):
    """Raised when a proof trace is internally inconsistent."""
