"""Errors the program can explain to the user without a stack trace."""


class LeadAgentError(RuntimeError):
    """Base class for all expected errors."""


class ConfigError(LeadAgentError):
    """A required setting (usually an API key) is missing."""


class MissingInputError(LeadAgentError):
    """A stage needs output from an earlier stage that does not exist yet."""


class ApiError(LeadAgentError):
    """An external API (Exa, OpenAI, Anthropic) answered with an error."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LLMError(LeadAgentError):
    """The LLM did not return usable JSON."""
