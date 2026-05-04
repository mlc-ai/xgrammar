"""Minimal OpenAI-compatible protocol models used by the benchmark."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict


class ChatCompletionMessage(BaseModel):
    """A chat message in an OpenAI-compatible chat completion request."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Optional[str] = None


class DebugConfig(BaseModel):
    """Local container for endpoint-specific debug options."""

    model_config = ConfigDict(extra="allow")

    grammar_execution_mode: Optional[str] = None
    ignore_eos: bool = False


class ChatToolCall(BaseModel):
    """A tool-call container kept for compatibility with stored records."""

    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    type: str = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request.

    The model intentionally allows extra fields so endpoint-specific options can
    pass through without depending on a provider SDK.
    """

    model_config = ConfigDict(extra="allow")

    messages: List[Union[ChatCompletionMessage, Dict[str, Any]]]
    model: str
    max_tokens: Optional[int] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    tool_choice: Optional[Any] = None
    response_format: Optional[Dict[str, Any]] = None
    debug_config: Optional[DebugConfig] = None
    stream_options: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
