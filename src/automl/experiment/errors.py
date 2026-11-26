class ExperimentError(Exception):
    """A rejected experiment operation with a machine-readable reason."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def as_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}
