from __future__ import annotations

from controller.controller import log
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         CastOllama - v1.0.0                                 ║
║          World-Class Ollama Model Management Framework for Python           ║
║                                                                              ║
║  Features:                                                                   ║
║    ✦ Singleton Pattern  — Only one instance per application                 ║
║    ✦ Builder Pattern    — Fluent API for clean configuration                ║
║    ✦ Request Queue      — Async queue to prevent server overload             ║
║    ✦ Tool Calling       — Register & dispatch custom tools automatically     ║
║    ✦ Web Search         — Native Ollama web search with API key             ║
║    ✦ Streaming          — Token-by-token output with thinking support        ║
║    ✦ Model Switching    — Hot-swap models at runtime                        ║
║    ✦ Server Control     — Start / restart Ollama server on demand           ║
║    ✦ Full Options API   — Temperature, context length, tokens, and more     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Usage (Builder pattern):
    from cast_ollama import CastOllama

    llm = (
        CastOllama.builder()
        .set_model("qwen3:4b")
        .set_host("http://localhost:11434")
        .set_temperature(0.7)
        .set_context_length(32768)
        .set_max_tokens(2048)
        .enable_streaming(True)
        .enable_thinking(True)
        .build()
    )

    # Register a custom tool
    @llm.tool(description="Get the weather for a city")
    def get_weather(city: str) -> str:
        return f"Sunny, 22°C in {city}"

    # Chat
    response = llm.chat("What is the weather in Dhaka?")
    log(response)
"""

import asyncio
import inspect
import json
import logging
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Union
import urllib.request
import urllib.error


# ─────────────────────────────────────────────────────────────────────────────
#  Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[CastOllama] %(levelname)s — %(message)s",
)
logger = logging.getLogger("CastOllama")


# ─────────────────────────────────────────────────────────────────────────────
#  Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOptions:
    """Encapsulates all Ollama model inference options."""
    temperature:     float          = 0.8
    num_ctx:         int            = 4096       # context length
    num_predict:     int            = -1         # max tokens (-1 = unlimited)
    top_k:          int            = 40
    top_p:          float          = 0.9
    min_p:          float          = 0.0
    repeat_penalty: float          = 1.1
    seed:           int            = -1          # -1 = random
    stop:           List[str]      = field(default_factory=list)
    tfs_z:          float          = 1.0
    typical_p:      float          = 1.0
    presence_penalty: float        = 0.0
    frequency_penalty: float       = 0.0
    mirostat:       int            = 0
    mirostat_tau:   float          = 5.0
    mirostat_eta:   float          = 0.1
    num_thread:     Optional[int]  = None
    num_gpu:        int            = -1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to Ollama API options dict, excluding sentinel values."""
        data: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            if v is None:
                continue
            if k == "seed" and v == -1:
                continue
            if k == "num_predict" and v == -1:
                continue
            if k == "num_gpu" and v == -1:
                continue
            if k == "stop" and not v:
                continue
            data[k] = v
        return data


@dataclass
class ToolDefinition:
    """Represents a registered tool (function) for Ollama tool calling."""
    name:        str
    description: str
    parameters:  Dict[str, Any]
    handler:     Callable


@dataclass
class ChatMessage:
    """A single conversation message."""
    role:      str          # "user" | "assistant" | "tool" | "system"
    content:   str
    tool_name: Optional[str] = None   # populated for tool-result messages
    thinking:  Optional[str] = None   # populated when thinking is enabled

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_name:
            d["tool_name"] = self.tool_name
        return d


@dataclass
class StreamChunk:
    """Yielded during a streaming response."""
    thinking:   Optional[str] = None
    content:    Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    done:       bool = False
    stats:      Optional[Dict] = None


@dataclass
class ChatResponse:
    """Final response after a complete chat round-trip."""
    model:      str
    content:    str
    thinking:   Optional[str]
    tool_calls: Optional[List[Dict]]
    messages:   List[ChatMessage]
    stats:      Optional[Dict]
    raw:        Dict


# ─────────────────────────────────────────────────────────────────────────────
#  Exception Hierarchy
# ─────────────────────────────────────────────────────────────────────────────

class CastOllamaError(Exception):
    """Base exception for CastOllama."""

class ServerNotRunningError(CastOllamaError):
    """Raised when Ollama server is unreachable."""

class ModelNotFoundError(CastOllamaError):
    """Raised when the specified model is not available."""

class ToolExecutionError(CastOllamaError):
    """Raised when a registered tool fails to execute."""

class WebSearchError(CastOllamaError):
    """Raised when the Ollama web-search API request fails."""

class BuilderError(CastOllamaError):
    """Raised for invalid builder configurations."""


# ─────────────────────────────────────────────────────────────────────────────
#  Internal HTTP helper  (stdlib-only, no third-party deps required)
# ─────────────────────────────────────────────────────────────────────────────

class _HttpClient:
    """
    Minimal HTTP client built on urllib so that CastOllama has
    zero mandatory dependencies.  Install `requests` for nicer errors;
    it is used automatically when present.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: int = 120):
        self.base_url  = base_url.rstrip("/")
        self.api_key   = api_key
        self.timeout   = timeout
        self._session  = None

        try:
            import requests
            self._requests = requests
        except ImportError:
            self._requests = None

    def _headers(self, extra: Optional[Dict] = None) -> Dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        if extra:
            h.update(extra)
        return h

    # ── Non-streaming POST ──────────────────────────────────────────────────

    def post_json(self, path: str, payload: Dict,
                  cloud: bool = False) -> Dict:
        url = f"{'https://ollama.com' if cloud else self.base_url}{path}"
        body = json.dumps(payload).encode()

        if self._requests:
            resp = self._requests.post(
                url,
                headers=self._headers(),
                data=body,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return resp.json()

        # Fallback: urllib
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise CastOllamaError(
                f"HTTP {exc.code}: {exc.read().decode()}"
            ) from exc

    # ── Streaming POST  ─────────────────────────────────────────────────────

    def post_stream(self, path: str, payload: Dict) -> Iterator[Dict]:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode()

        if self._requests:
            with self._requests.post(
                url,
                headers=self._headers(),
                data=body,
                stream=True,
                timeout=self.timeout,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        chunk = line if isinstance(line, str) else line.decode()
                        yield json.loads(chunk)
            return

        # Fallback: urllib
        req = urllib.request.Request(
            url,
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    line = raw_line.decode().strip()
                    if line:
                        yield json.loads(line)
        except urllib.error.HTTPError as exc:
            raise CastOllamaError(
                f"HTTP {exc.code}: {exc.read().decode()}"
            ) from exc

    # ── GET ─────────────────────────────────────────────────────────────────

    def get_json(self, path: str) -> Dict:
        url = f"{self.base_url}{path}"
        if self._requests:
            resp = self._requests.get(
                url, headers=self._headers(), timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()

        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise CastOllamaError(
                f"HTTP {exc.code}: {exc.read().decode()}"
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
#  Builder
# ─────────────────────────────────────────────────────────────────────────────

class CastOllamaBuilder:
    """
    Fluent builder to configure and create/retrieve the CastOllama singleton.

    Example:
        llm = (
            CastOllama.builder()
            .set_model("llama3.2")
            .set_temperature(0.5)
            .enable_streaming(True)
            .build()
        )
    """

    def __init__(self):
        self._model:          str            = "llama3.2"
        self._host:           str            = "http://localhost:11434"
        self._system_prompt:  Optional[str]  = None
        self._api_key:        Optional[str]  = None
        self._options:        ModelOptions   = ModelOptions()
        self._streaming:      bool           = False
        self._thinking:       bool           = False
        self._web_search:     bool           = False
        self._web_search_key: Optional[str]  = None
        self._max_queue_size: int            = 50
        self._timeout:        int            = 120
        self._log_level:      str            = "INFO"
        self._keep_history:   bool           = True

    # ── Model / host ─────────────────────────────────────────────────────────

    def set_model(self, model: str) -> "CastOllamaBuilder":
        """Set the default model (e.g. 'llama3.2', 'qwen3:4b', 'deepseek-r1')."""
        self._model = model
        return self

    def set_host(self, host: str) -> "CastOllamaBuilder":
        """Set Ollama server URL. Default: 'http://localhost:11434'."""
        self._host = host
        return self

    def set_system_prompt(self, prompt: str) -> "CastOllamaBuilder":
        """Set a default system prompt injected at the start of every chat."""
        self._system_prompt = prompt
        return self

    # ── Auth ─────────────────────────────────────────────────────────────────

    def set_api_key(self, key: str) -> "CastOllamaBuilder":
        """API key for Ollama Cloud features (required for web search quota)."""
        self._api_key = key
        return self

    # ── Inference options ────────────────────────────────────────────────────

    def set_temperature(self, value: float) -> "CastOllamaBuilder":
        """Sampling temperature 0.0–2.0. Lower = more deterministic."""
        if not 0.0 <= value <= 2.0:
            raise BuilderError("Temperature must be between 0.0 and 2.0")
        self._options.temperature = value
        return self

    def set_context_length(self, tokens: int) -> "CastOllamaBuilder":
        """Maximum context window in tokens. Recommended ≥ 32768 for tools."""
        self._options.num_ctx = tokens
        return self

    def set_max_tokens(self, tokens: int) -> "CastOllamaBuilder":
        """Maximum tokens to generate (-1 = unlimited)."""
        self._options.num_predict = tokens
        return self

    def set_top_k(self, k: int) -> "CastOllamaBuilder":
        self._options.top_k = k
        return self

    def set_top_p(self, p: float) -> "CastOllamaBuilder":
        self._options.top_p = p
        return self

    def set_min_p(self, p: float) -> "CastOllamaBuilder":
        self._options.min_p = p
        return self

    def set_repeat_penalty(self, penalty: float) -> "CastOllamaBuilder":
        self._options.repeat_penalty = penalty
        return self

    def set_seed(self, seed: int) -> "CastOllamaBuilder":
        """Set a fixed seed for reproducible outputs."""
        self._options.seed = seed
        return self

    def set_stop_tokens(self, tokens: List[str]) -> "CastOllamaBuilder":
        """List of token strings that stop generation."""
        self._options.stop = tokens
        return self

    def set_num_threads(self, threads: int) -> "CastOllamaBuilder":
        """Number of CPU threads for inference."""
        self._options.num_thread = threads
        return self

    def set_num_gpu(self, layers: int) -> "CastOllamaBuilder":
        """Number of GPU layers to offload (-1 = auto)."""
        self._options.num_gpu = layers
        return self

    def set_mirostat(self, mode: int, tau: float = 5.0,
                     eta: float = 0.1) -> "CastOllamaBuilder":
        """Enable Mirostat sampling (0=off, 1=Mirostat, 2=Mirostat 2.0)."""
        self._options.mirostat     = mode
        self._options.mirostat_tau = tau
        self._options.mirostat_eta = eta
        return self

    # ── Feature toggles ──────────────────────────────────────────────────────

    def enable_streaming(self, enabled: bool = True) -> "CastOllamaBuilder":
        """Enable token-by-token streaming responses."""
        self._streaming = enabled
        return self

    def enable_thinking(self, enabled: bool = True) -> "CastOllamaBuilder":
        """Enable extended thinking output (supported models only, e.g. qwen3)."""
        self._thinking = enabled
        return self

    def enable_web_search(
        self, enabled: bool = True,
        api_key: Optional[str] = None
    ) -> "CastOllamaBuilder":
        """
        Enable Ollama's native web search.
        api_key is required for higher rate limits (from ollama.com account).
        """
        self._web_search = enabled
        if api_key:
            self._web_search_key = api_key
        return self

    def keep_history(self, enabled: bool = True) -> "CastOllamaBuilder":
        """Keep conversation history across multiple chat() calls."""
        self._keep_history = enabled
        return self

    # ── Queue / perf ─────────────────────────────────────────────────────────

    def set_max_queue_size(self, size: int) -> "CastOllamaBuilder":
        """Maximum number of tasks in the request queue (default 50)."""
        self._max_queue_size = size
        return self

    def set_timeout(self, seconds: int) -> "CastOllamaBuilder":
        """HTTP request timeout in seconds."""
        self._timeout = seconds
        return self

    # ── Logging ──────────────────────────────────────────────────────────────

    def set_log_level(
        self, level: str = "INFO"
    ) -> "CastOllamaBuilder":
        """Logging verbosity: DEBUG | INFO | WARNING | ERROR."""
        self._log_level = level.upper()
        return self

    # ── Build ────────────────────────────────────────────────────────────────

    def build(self) -> "CastOllama":
        """
        Construct (or reconfigure) the CastOllama singleton and return it.
        Safe to call multiple times — always returns the single instance.
        """
        return CastOllama._create(self)


# ─────────────────────────────────────────────────────────────────────────────
#  CastOllama — Main Singleton Class
# ─────────────────────────────────────────────────────────────────────────────

class CastOllama:
    """
    CastOllama — World-class Ollama model management class.

    ⚠  Do NOT instantiate directly.  Use the builder:
        instance = CastOllama.builder().set_model("llama3.2").build()
    """

    # ── Singleton state ──────────────────────────────────────────────────────
    _instance: Optional["CastOllama"]   = None
    _lock:     threading.Lock           = threading.Lock()

    # ── Constructor / Singleton guard ────────────────────────────────────────

    def __new__(cls, *args, **kwargs):
        raise RuntimeError(
            "Use CastOllama.builder()...build() to obtain the singleton."
        )

    @classmethod
    def _create(cls, cfg: "CastOllamaBuilder") -> "CastOllama":
        with cls._lock:
            if cls._instance is None:
                instance = object.__new__(cls)
                instance._init(cfg)
                cls._instance = instance
            else:
                # Reconfigure the existing singleton (safe re-apply)
                cls._instance._reconfigure(cfg)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "CastOllama":
        """Return the existing singleton or raise if not yet built."""
        if cls._instance is None:
            raise CastOllamaError(
                "CastOllama has not been built yet. "
                "Call CastOllama.builder()...build() first."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """
        Destroy the singleton.  Useful in tests.
        After this call, builder().build() will create a fresh instance.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._stop_worker()
            cls._instance = None
        logger.info("Singleton reset. A fresh instance can now be built.")

    @classmethod
    def builder(cls) -> CastOllamaBuilder:
        """Entry point — returns a new builder."""
        return CastOllamaBuilder()

    # ── Private init ─────────────────────────────────────────────────────────

    def _init(self, cfg: CastOllamaBuilder) -> None:
        """Called once when the singleton is first created."""
        self._cfg = cfg
        self._apply_config(cfg)

        # Request queue + worker thread
        self._queue:  queue.Queue = queue.Queue(maxsize=cfg._max_queue_size)
        self._worker: threading.Thread = threading.Thread(
            target=self._queue_worker,
            daemon=True,
            name="CastOllama-Worker",
        )
        self._running = True
        self._worker.start()

        logger.info(
            f"CastOllama initialized — model={self._model}, "
            f"host={self._host}, streaming={self._streaming}"
        )

    def _reconfigure(self, cfg: CastOllamaBuilder) -> None:
        """Hot-reconfigure without destroying history or tools."""
        self._cfg = cfg
        self._apply_config(cfg)
        logger.info(
            f"CastOllama reconfigured — model={self._model}, "
            f"streaming={self._streaming}"
        )

    def _apply_config(self, cfg: CastOllamaBuilder) -> None:
        """Extract values from builder config."""
        self._model:          str           = cfg._model
        self._host:           str           = cfg._host
        self._system_prompt:  Optional[str] = cfg._system_prompt
        self._options:        ModelOptions  = cfg._options
        self._streaming:      bool          = cfg._streaming
        self._thinking:       bool          = cfg._thinking
        self._web_search:     bool          = cfg._web_search
        self._web_search_key: Optional[str] = cfg._web_search_key or cfg._api_key
        self._keep_history:   bool          = cfg._keep_history
        self._timeout:        int           = cfg._timeout

        # Set log level
        logger.setLevel(getattr(logging, cfg._log_level, logging.INFO))

        # HTTP client
        self._http = _HttpClient(
            base_url=cfg._host,
            api_key=cfg._api_key,
            timeout=cfg._timeout,
        )

        # Tool registry  (preserved on reconfigure)
        if not hasattr(self, "_tools"):
            self._tools:    Dict[str, ToolDefinition] = {}

        # Conversation history (preserved on reconfigure)
        if not hasattr(self, "_history"):
            self._history:  List[ChatMessage] = []

        # Web-search tool auto-register
        if self._web_search:
            self._register_web_search_tool()

    # ─────────────────────────────────────────────────────────────────────────
    #  Queue Worker
    # ─────────────────────────────────────────────────────────────────────────

    def _queue_worker(self) -> None:
        """Background worker — processes queued requests sequentially."""
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            fn, args, kwargs, result_holder, event = task
            try:
                result_holder["result"] = fn(*args, **kwargs)
            except Exception as exc:
                result_holder["error"] = exc
            finally:
                event.set()
                self._queue.task_done()

    def _stop_worker(self) -> None:
        self._running = False
        if hasattr(self, "_worker") and self._worker.is_alive():
            self._worker.join(timeout=3)

    def _enqueue(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Submit a callable to the request queue and block until it completes.
        Raises queue.Full if the queue is at capacity.
        """
        holder: Dict[str, Any] = {}
        event  = threading.Event()
        task   = (fn, args, kwargs, holder, event)
        try:
            self._queue.put(task, timeout=5)
        except queue.Full as exc:
            raise CastOllamaError(
                "CastOllama request queue is full. "
                "Increase max_queue_size or slow down requests."
            ) from exc
        event.wait()
        if "error" in holder:
            raise holder["error"]
        return holder["result"]

    # ─────────────────────────────────────────────────────────────────────────
    #  Model Management
    # ─────────────────────────────────────────────────────────────────────────

    def switch_model(self, model: str) -> "CastOllama":
        """
        Hot-swap the active model at runtime.

        Example:
            llm.switch_model("deepseek-r1:14b")
        """
        previous      = self._model
        self._model   = model
        logger.info(f"Model switched: {previous} → {model}")
        return self

    def get_model(self) -> str:
        """Return the currently active model name."""
        return self._model

    def list_local_models(self) -> List[Dict]:
        """Return all models available on the local Ollama server."""
        try:
            data = self._http.get_json("/api/tags")
            return data.get("models", [])
        except Exception as exc:
            raise CastOllamaError(f"Failed to list models: {exc}") from exc

    def pull_model(self, model: str) -> None:
        """
        Pull a model from the Ollama library.
        Blocks until the download is complete.
        """
        logger.info(f"Pulling model '{model}' …")
        for chunk in self._http.post_stream(
            "/api/pull", {"model": model, "stream": True}
        ):
            status = chunk.get("status", "")
            if status:
                logger.debug(f"  pull: {status}")
        logger.info(f"Model '{model}' is ready.")

    def delete_model(self, model: str) -> None:
        """Delete a local model."""
        self._http.post_json("/api/delete", {"model": model})
        logger.info(f"Model '{model}' deleted.")

    def model_info(self, model: Optional[str] = None) -> Dict:
        """Return detailed info about a model."""
        return self._http.post_json(
            "/api/show", {"model": model or self._model}
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  Tool Registration
    # ─────────────────────────────────────────────────────────────────────────

    def tool(
        self,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict] = None,
    ) -> Callable:
        """
        Decorator to register a Python function as an Ollama tool.

        The parameter schema is auto-inferred from type hints if not provided.

        Example:
            @llm.tool(description="Get the current stock price for a ticker")
            def get_stock_price(ticker: str) -> str:
                return f"${ticker}: $142.50"
        """
        def decorator(fn: Callable) -> Callable:
            schema = parameters_schema or _infer_schema(fn)
            desc   = description or (fn.__doc__ or fn.__name__).strip()

            tool_def = ToolDefinition(
                name        = fn.__name__,
                description = desc,
                parameters  = schema,
                handler     = fn,
            )
            self._tools[fn.__name__] = tool_def
            logger.info(f"Tool registered: '{fn.__name__}'")

            @wraps(fn)
            def wrapper(*args, **kwargs):
                return fn(*args, **kwargs)
            return wrapper
        return decorator

    def register_tool(
        self,
        fn: Callable,
        description: Optional[str] = None,
        parameters_schema: Optional[Dict] = None,
    ) -> "CastOllama":
        """
        Programmatically register a tool function (non-decorator form).

        Example:
            def search_docs(query: str) -> str:
                ...
            llm.register_tool(search_docs, description="Search docs")
        """
        schema = parameters_schema or _infer_schema(fn)
        desc   = description or (fn.__doc__ or fn.__name__).strip()
        self._tools[fn.__name__] = ToolDefinition(
            name        = fn.__name__,
            description = desc,
            parameters  = schema,
            handler     = fn,
        )
        logger.info(f"Tool registered: '{fn.__name__}'")
        return self

    def unregister_tool(self, name: str) -> "CastOllama":
        """Remove a registered tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Tool unregistered: '{name}'")
        return self

    def list_tools(self) -> List[str]:
        """Return names of all registered tools."""
        return list(self._tools.keys())

    def _build_tools_payload(self) -> List[Dict]:
        """Convert registered tools to Ollama API format."""
        return [
            {
                "type": "function",
                "function": {
                    "name":        td.name,
                    "description": td.description,
                    "parameters":  td.parameters,
                },
            }
            for td in self._tools.values()
        ]

    def _dispatch_tool(self, name: str, args: Dict) -> str:
        """Execute a tool and return its result as a string."""
        if name not in self._tools:
            raise ToolExecutionError(f"Unknown tool: '{name}'")
        try:
            result = self._tools[name].handler(**args)
            return str(result)
        except Exception as exc:
            logger.error(f"Tool '{name}' raised: {exc}")
            raise ToolExecutionError(
                f"Tool '{name}' failed: {exc}"
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    #  Web Search
    # ─────────────────────────────────────────────────────────────────────────

    def enable_web_search(self, api_key: Optional[str] = None) -> "CastOllama":
        """Enable Ollama native web search. Optionally supply/update API key."""
        self._web_search = True
        if api_key:
            self._web_search_key = api_key
        self._register_web_search_tool()
        logger.info("Web search enabled.")
        return self

    def disable_web_search(self) -> "CastOllama":
        """Disable web search and remove the web_search tool."""
        self._web_search = False
        if "web_search" in self._tools:
            del self._tools["web_search"]
        logger.info("Web search disabled.")
        return self

    def _register_web_search_tool(self) -> None:
        """Register the built-in Ollama web_search tool."""
        def web_search(query: str, max_results: int = 5) -> str:
            return self._call_web_search_api(query, max_results)

        web_search.__doc__ = "Search the web for up-to-date information."
        self._tools["web_search"] = ToolDefinition(
            name        = "web_search",
            description = "Search the web for up-to-date information.",
            parameters  = {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type":        "string",
                        "description": "The search query.",
                    },
                    "max_results": {
                        "type":        "integer",
                        "description": "Number of results to return (default 5).",
                    },
                },
            },
            handler = web_search,
        )

    def _call_web_search_api(self, query: str, max_results: int = 5) -> str:
        """
        Calls the official Ollama Cloud web_search API.
        Falls back to a local summary if no API key is set.
        """
        key = self._web_search_key or os.getenv("OLLAMA_API_KEY")
        if not key:
            # Without an API key we can still try the free-tier endpoint
            logger.warning(
                "No OLLAMA_API_KEY set. Free-tier web search rate limits apply."
            )

        try:
            payload = {"query": query, "max_results": max_results}
            http = _HttpClient(
                base_url="https://ollama.com",
                api_key=key,
                timeout=self._timeout,
            )
            data = http.post_json("/api/web_search", payload, cloud=True)
            results = data.get("results", [])
            if not results:
                return "No results found."
            snippets = []
            for r in results:
                snippets.append(
                    f"[{r.get('title', 'No title')}]({r.get('url', '')})\n"
                    f"{r.get('content', '')}"
                )
            return "\n\n---\n\n".join(snippets)
        except Exception as exc:
            raise WebSearchError(f"Web search failed: {exc}") from exc

    def web_search(self, query: str, max_results: int = 5) -> str:
        """
        Perform a web search directly (without triggering a chat).

        Returns raw search result text.
        """
        return self._call_web_search_api(query, max_results)

    # ─────────────────────────────────────────────────────────────────────────
    #  Conversation History
    # ─────────────────────────────────────────────────────────────────────────

    def clear_history(self) -> "CastOllama":
        """Clear the conversation history."""
        self._history = []
        logger.info("Conversation history cleared.")
        return self

    def get_history(self) -> List[ChatMessage]:
        """Return a copy of the current conversation history."""
        return list(self._history)

    def add_system_message(self, content: str) -> "CastOllama":
        """Prepend or replace the system message in history."""
        self._history = [
            m for m in self._history if m.role != "system"
        ]
        self._history.insert(0, ChatMessage(role="system", content=content))
        return self

    def load_history(self, messages: List[Dict]) -> "CastOllama":
        """
        Restore conversation history from a list of dicts.

        Each dict should have 'role' and 'content' keys.
        """
        self._history = [
            ChatMessage(role=m["role"], content=m.get("content", ""))
            for m in messages
        ]
        return self

    def export_history(self) -> List[Dict]:
        """Export conversation history as a list of dicts (JSON-serialisable)."""
        return [m.to_dict() for m in self._history]

    def _build_messages_payload(
        self,
        user_content: str,
        extra: Optional[List[ChatMessage]] = None,
    ) -> List[Dict]:
        """Build the full messages list for an API request."""
        messages: List[Dict] = []

        # System prompt
        sys_in_history = any(m.role == "system" for m in self._history)
        if self._system_prompt and not sys_in_history:
            messages.append({"role": "system", "content": self._system_prompt})

        # History
        if self._keep_history:
            messages.extend(m.to_dict() for m in self._history)

        # Extra tool-result messages
        if extra:
            messages.extend(m.to_dict() for m in extra)

        # Current user turn
        messages.append({"role": "user", "content": user_content})
        return messages

    # ─────────────────────────────────────────────────────────────────────────
    #  Core Chat  (non-streaming)
    # ─────────────────────────────────────────────────────────────────────────

    def _chat_once(
        self,
        messages:  List[Dict],
        model:     str,
        tools:     Optional[List[Dict]] = None,
        stream:    bool = False,
        think:     bool = False,
    ) -> Dict:
        """Single API round-trip.  Returns raw JSON."""
        payload: Dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "stream":   stream,
            "options":  self._options.to_dict(),
        }
        if tools:
            payload["tools"] = tools
        if think:
            payload["think"] = True

        return self._http.post_json("/api/chat", payload)

    def _chat_once_stream(
        self,
        messages:  List[Dict],
        model:     str,
        tools:     Optional[List[Dict]] = None,
        think:     bool = False,
    ) -> Iterator[StreamChunk]:
        """Streaming API call. Yields StreamChunk objects."""
        payload: Dict[str, Any] = {
            "model":    model,
            "messages": messages,
            "stream":   True,
            "options":  self._options.to_dict(),
        }
        if tools:
            payload["tools"] = tools
        if think:
            payload["think"] = True

        accumulated_thinking   = ""
        accumulated_content    = ""
        accumulated_tool_calls: List[Dict] = []

        for chunk in self._http.post_stream("/api/chat", payload):
            msg  = chunk.get("message", {})
            done = chunk.get("done", False)

            thinking_token = msg.get("thinking", "")
            content_token  = msg.get("content",  "")
            tc             = msg.get("tool_calls")

            if thinking_token:
                accumulated_thinking += thinking_token
            if content_token:
                accumulated_content  += content_token
            if tc:
                accumulated_tool_calls.extend(tc)

            sc = StreamChunk(
                thinking   = thinking_token or None,
                content    = content_token  or None,
                tool_calls = tc,
                done       = done,
                stats      = {k: v for k, v in chunk.items()
                               if k not in ("message", "done")} if done else None,
            )
            yield sc

            if done:
                break

    # ─────────────────────────────────────────────────────────────────────────
    #  Public Chat Interface
    # ─────────────────────────────────────────────────────────────────────────

    def chat(
        self,
        message:  str,
        model:    Optional[str] = None,
        stream:   Optional[bool] = None,
        think:    Optional[bool] = None,
        tools:    Optional[bool] = None,
    ) -> Union[ChatResponse, Iterator[StreamChunk]]:
        """
        Send a message and receive a response.

        Args:
            message: The user's input text.
            model:   Override the active model for this request only.
            stream:  Override the streaming setting for this request only.
            think:   Override the thinking setting for this request only.
            tools:   Set to False to suppress tool calling for this request.

        Returns:
            ChatResponse      — when streaming is off (default).
            Iterator[StreamChunk] — when streaming is on.

        Example:
            # Non-streaming
            resp = llm.chat("What is 2+2?")
            log(resp.content)

            # Streaming
            llm.enable_streaming(True)
            for chunk in llm.chat("Tell me a story"):
                if chunk.content:
                    log(chunk.content, end="", flush=True)
        """
        use_stream = self._streaming if stream is None else stream
        use_think  = self._thinking  if think  is None else think
        use_model  = model           or self._model
        use_tools  = self._build_tools_payload() if (
            tools is not False and self._tools
        ) else None

        if use_stream:
            return self._enqueue(
                self._chat_stream_with_tools,
                message, use_model, use_tools, use_think,
            )
        else:
            return self._enqueue(
                self._chat_sync_with_tools,
                message, use_model, use_tools, use_think,
            )

    def _chat_sync_with_tools(
        self,
        message:   str,
        model:     str,
        tools:     Optional[List[Dict]],
        think:     bool,
    ) -> ChatResponse:
        """Non-streaming chat with automatic tool-call loop."""
        messages = self._build_messages_payload(message)
        extra_msgs: List[ChatMessage] = []
        iterations = 0

        while iterations < 10:   # safety cap on tool-call loops
            iterations += 1
            raw = self._chat_once(messages, model, tools=tools, think=think)
            msg = raw.get("message", {})
            assistant_content  = msg.get("content",  "")
            assistant_thinking = msg.get("thinking", "")
            tc                 = msg.get("tool_calls")

            if tc:
                # Append assistant message with tool_calls
                messages.append({
                    "role":       "assistant",
                    "content":    assistant_content,
                    "tool_calls": tc,
                })

                # Execute each tool and collect results
                for call in tc:
                    fn   = call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    logger.debug(f"Tool call: {name}({args})")
                    try:
                        result = self._dispatch_tool(name, args)
                    except ToolExecutionError as exc:
                        result = f"ERROR: {exc}"
                    logger.debug(f"Tool result: {result[:120]}…")
                    messages.append({
                        "role":      "tool",
                        "tool_name": name,
                        "content":   result,
                    })
                    extra_msgs.append(ChatMessage(
                        role="tool",
                        content=result,
                        tool_name=name,
                    ))
                # Continue the loop — feed results back to model
                continue

            # No more tool calls — final answer
            break

        # Update history
        if self._keep_history:
            self._history.append(ChatMessage(role="user",      content=message))
            self._history.extend(extra_msgs)
            self._history.append(ChatMessage(
                role      = "assistant",
                content   = assistant_content,
                thinking  = assistant_thinking or None,
            ))

        return ChatResponse(
            model      = raw.get("model", model),
            content    = assistant_content,
            thinking   = assistant_thinking or None,
            tool_calls = tc,
            messages   = [ChatMessage(role=m["role"], content=m.get("content",""))
                          for m in messages],
            stats      = {k: v for k, v in raw.items() if k != "message"},
            raw        = raw,
        )

    def _chat_stream_with_tools(
        self,
        message:   str,
        model:     str,
        tools:     Optional[List[Dict]],
        think:     bool,
    ) -> Iterator[StreamChunk]:
        """
        Streaming chat with automatic tool-call agentic loop.
        Yields StreamChunk objects.
        """
        messages = self._build_messages_payload(message)
        iterations = 0

        while iterations < 10:
            iterations += 1
            accumulated_thinking   = ""
            accumulated_content    = ""
            accumulated_tool_calls: List[Dict] = []

            for chunk in self._chat_once_stream(messages, model, tools, think):
                if chunk.thinking:
                    accumulated_thinking += chunk.thinking
                if chunk.content:
                    accumulated_content  += chunk.content
                if chunk.tool_calls:
                    accumulated_tool_calls.extend(chunk.tool_calls)
                yield chunk

            if accumulated_tool_calls:
                messages.append({
                    "role":       "assistant",
                    "content":    accumulated_content,
                    "tool_calls": accumulated_tool_calls,
                })
                for call in accumulated_tool_calls:
                    fn   = call.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", {})
                    try:
                        result = self._dispatch_tool(name, args)
                    except ToolExecutionError as exc:
                        result = f"ERROR: {exc}"
                    messages.append({
                        "role":      "tool",
                        "tool_name": name,
                        "content":   result,
                    })
                continue   # feed tool results back, keep streaming

            # No tool calls in this round — we're done
            break

        # Update history
        if self._keep_history:
            self._history.append(ChatMessage(role="user",     content=message))
            self._history.append(ChatMessage(
                role     = "assistant",
                content  = accumulated_content,
                thinking = accumulated_thinking or None,
            ))

    # ─────────────────────────────────────────────────────────────────────────
    #  Generate  (single-turn, no history)
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt:        str,
        model:         Optional[str] = None,
        system:        Optional[str] = None,
        stream:        Optional[bool] = None,
        format:        Optional[str] = None,   # "json"
        raw:           bool = False,
    ) -> Union[str, Iterator[str]]:
        """
        Lower-level text generation (no chat history, no tools).

        Args:
            prompt:  The raw prompt string.
            model:   Override model for this call.
            system:  Override system prompt for this call.
            stream:  Override streaming for this call.
            format:  Output format — pass "json" to force JSON mode.
            raw:     If True, use the raw prompt without template.

        Returns:
            str             — full response when streaming=False.
            Iterator[str]   — token stream when streaming=True.
        """
        use_model  = model or self._model
        use_stream = self._streaming if stream is None else stream

        payload: Dict[str, Any] = {
            "model":   use_model,
            "prompt":  prompt,
            "stream":  use_stream,
            "options": self._options.to_dict(),
            "raw":     raw,
        }
        if system:
            payload["system"] = system
        if format:
            payload["format"] = format

        def _sync():
            data = self._http.post_json("/api/generate", payload)
            return data.get("response", "")

        def _stream():
            for chunk in self._http.post_stream("/api/generate", payload):
                if token := chunk.get("response"):
                    yield token
                if chunk.get("done"):
                    break

        if use_stream:
            return self._enqueue(_stream)
        else:
            return self._enqueue(_sync)

    # ─────────────────────────────────────────────────────────────────────────
    #  Embeddings
    # ─────────────────────────────────────────────────────────────────────────

    def embed(
        self,
        text:  Union[str, List[str]],
        model: Optional[str] = None,
    ) -> Union[List[float], List[List[float]]]:
        """
        Generate embeddings for text or a list of texts.

        Returns a single vector or list of vectors.
        """
        use_model = model or self._model
        payload   = {
            "model": use_model,
            "input": text,
        }
        data = self._enqueue(
            lambda: self._http.post_json("/api/embed", payload)
        )
        embeddings = data.get("embeddings", [])
        if isinstance(text, str):
            return embeddings[0] if embeddings else []
        return embeddings

    # ─────────────────────────────────────────────────────────────────────────
    #  Options  (runtime adjustments)
    # ─────────────────────────────────────────────────────────────────────────

    def set_temperature(self, value: float) -> "CastOllama":
        """Change temperature at runtime."""
        self._options.temperature = value
        return self

    def set_context_length(self, tokens: int) -> "CastOllama":
        """Change context window length at runtime."""
        self._options.num_ctx = tokens
        return self

    def set_max_tokens(self, tokens: int) -> "CastOllama":
        """Change max output tokens at runtime."""
        self._options.num_predict = tokens
        return self

    def set_top_p(self, value: float) -> "CastOllama":
        self._options.top_p = value
        return self

    def set_top_k(self, value: int) -> "CastOllama":
        self._options.top_k = value
        return self

    def set_seed(self, seed: int) -> "CastOllama":
        self._options.seed = seed
        return self

    def set_stop_tokens(self, tokens: List[str]) -> "CastOllama":
        self._options.stop = tokens
        return self

    def get_options(self) -> Dict:
        """Return the current inference options as a dict."""
        return self._options.to_dict()

    # ─────────────────────────────────────────────────────────────────────────
    #  Streaming toggle
    # ─────────────────────────────────────────────────────────────────────────

    def enable_streaming(self, enabled: bool = True) -> "CastOllama":
        """Toggle streaming on/off at runtime."""
        self._streaming = enabled
        logger.info(f"Streaming {'enabled' if enabled else 'disabled'}.")
        return self

    def enable_thinking(self, enabled: bool = True) -> "CastOllama":
        """Toggle extended thinking output at runtime."""
        self._thinking = enabled
        logger.info(f"Thinking {'enabled' if enabled else 'disabled'}.")
        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  System Prompt
    # ─────────────────────────────────────────────────────────────────────────

    def set_system_prompt(self, prompt: str) -> "CastOllama":
        """Update the system prompt without rebuilding the singleton."""
        self._system_prompt = prompt
        return self

    # ─────────────────────────────────────────────────────────────────────────
    #  Server Management
    # ─────────────────────────────────────────────────────────────────────────

    def is_server_running(self) -> bool:
        """Return True if the Ollama server is reachable."""
        try:
            self._http.get_json("/api/tags")
            return True
        except Exception:
            return False

    def restart_server(self, wait_seconds: int = 5) -> bool:
        """
        Attempt to restart the Ollama server process.

        Works on macOS (launchctl), Linux (systemctl / direct exec),
        and Windows (taskkill + start).

        Returns True if the server responds after restart.
        """
        logger.info("Restarting Ollama server …")
        system = platform.system()

        try:
            if system == "Darwin":
                subprocess.run(
                    ["pkill", "-x", "ollama"], check=False
                )
                time.sleep(1)
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            elif system == "Linux":
                # Try systemd first
                result = subprocess.run(
                    ["systemctl", "is-active", "ollama"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    subprocess.run(
                        ["systemctl", "restart", "ollama"], check=False
                    )
                else:
                    # Direct process
                    subprocess.run(
                        ["pkill", "-x", "ollama"], check=False
                    )
                    time.sleep(1)
                    subprocess.Popen(
                        ["ollama", "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )

            elif system == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    check=False, capture_output=True
                )
                time.sleep(1)
                subprocess.Popen(
                    ["ollama", "serve"],
                    creationflags=subprocess.CREATE_NEW_CONSOLE,
                )

            else:
                logger.error(f"Unsupported OS for server restart: {system}")
                return False

        except FileNotFoundError:
            logger.error(
                "'ollama' binary not found. "
                "Make sure Ollama is installed and on PATH."
            )
            return False

        # Wait for server to come back
        logger.info(f"Waiting {wait_seconds}s for server to restart …")
        for i in range(wait_seconds * 2):
            time.sleep(0.5)
            if self.is_server_running():
                logger.info("Ollama server is back online.")
                return True

        logger.warning("Ollama server did not respond after restart.")
        return False

    def start_server(self) -> bool:
        """
        Start the Ollama server if it is not already running.
        Returns True if the server is available after the call.
        """
        if self.is_server_running():
            logger.info("Ollama server is already running.")
            return True
        return self.restart_server()

    def stop_server(self) -> bool:
        """
        Stop the Ollama server process.
        Returns True if the process was terminated.
        """
        logger.info("Stopping Ollama server …")
        system = platform.system()
        try:
            if system in ("Darwin", "Linux"):
                subprocess.run(["pkill", "-x", "ollama"], check=False)
            elif system == "Windows":
                subprocess.run(
                    ["taskkill", "/F", "/IM", "ollama.exe"],
                    check=False, capture_output=True
                )
            logger.info("Ollama server stopped.")
            return True
        except FileNotFoundError:
            logger.error("'ollama' binary not found.")
            return False

    def server_health(self) -> Dict:
        """
        Return a health summary of the Ollama server.
        """
        running = self.is_server_running()
        info: Dict[str, Any] = {
            "running":      running,
            "host":         self._host,
            "active_model": self._model,
        }
        if running:
            models = self.list_local_models()
            info["local_models"]  = [m["name"] for m in models]
            info["model_count"]   = len(models)
        return info

    # ─────────────────────────────────────────────────────────────────────────
    #  Convenience wrappers
    # ─────────────────────────────────────────────────────────────────────────

    def ask(self, question: str, **kwargs) -> str:
        """
        Convenience shortcut: chat() and return only the text content.

        Example:
            answer = llm.ask("What is the capital of Bangladesh?")
        """
        resp = self.chat(question, **kwargs)
        if isinstance(resp, ChatResponse):
            return resp.content
        # streaming — collect
        content = ""
        for chunk in resp:
            if chunk.content:
                content += chunk.content
        return content

    def think(self, message: str, **kwargs) -> tuple[str, str]:
        """
        Force thinking mode and return (thinking, content) tuple.

        Example:
            reasoning, answer = llm.think("Solve this math puzzle …")
        """
        resp = self.chat(message, think=True, stream=False, **kwargs)
        return (resp.thinking or "", resp.content)

    def json_chat(self, message: str, **kwargs) -> Dict:
        """
        Request a JSON-mode response and parse it automatically.

        Adds format=json to the options; raises ValueError if parse fails.
        """
        orig_format = None
        # Temporarily force JSON format via generate
        result = self.generate(
            prompt=message,
            stream=False,
            format="json",
            **kwargs,
        )
        try:
            return json.loads(result)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Model did not return valid JSON: {result[:200]}"
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    #  Dunder methods
    # ─────────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<CastOllama model={self._model!r} "
            f"host={self._host!r} "
            f"streaming={self._streaming} "
            f"thinking={self._thinking} "
            f"web_search={self._web_search} "
            f"tools={list(self._tools.keys())} "
            f"queue_size={self._queue.qsize()}>"
        )

    def __str__(self) -> str:
        return self.__repr__()


# ─────────────────────────────────────────────────────────────────────────────
#  Helper:  Auto-infer JSON schema from Python function type hints
# ─────────────────────────────────────────────────────────────────────────────

_PY_TO_JSON: Dict[Any, str] = {
    str:   "string",
    int:   "integer",
    float: "number",
    bool:  "boolean",
    list:  "array",
    dict:  "object",
}


def _infer_schema(fn: Callable) -> Dict:
    """
    Build an Ollama-compatible JSON schema from a function's type hints
    and docstring.
    """
    sig    = inspect.signature(fn)
    hints  = fn.__annotations__
    doc    = inspect.getdoc(fn) or ""

    # Parse "Args:" section of docstring for per-parameter descriptions
    arg_docs: Dict[str, str] = {}
    in_args   = False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("args:") or stripped.lower().startswith("arguments:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped[0].isspace() and ":" in stripped:
                in_args = False
            else:
                m = re.match(r"(\w+)\s*(?:\(.*?\))?\s*:\s*(.*)", stripped)
                if m:
                    arg_docs[m.group(1)] = m.group(2).strip()

    properties: Dict[str, Any] = {}
    required:   List[str]      = []

    for param_name, param in sig.parameters.items():
        if param_name == "self":
            continue

        py_type  = hints.get(param_name, str)
        # Strip Optional[X]
        origin = getattr(py_type, "__origin__", None)
        args   = getattr(py_type, "__args__", None)
        if origin is Union and type(None) in args:
            py_type = next(a for a in args if a is not type(None))

        json_type = _PY_TO_JSON.get(py_type, "string")
        prop: Dict[str, Any] = {"type": json_type}
        if param_name in arg_docs:
            prop["description"] = arg_docs[param_name]

        properties[param_name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    return {
        "type":       "object",
        "required":   required,
        "properties": properties,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Re-export Union for _infer_schema
# ─────────────────────────────────────────────────────────────────────────────

from typing import Union   # noqa: E402 (needed by _infer_schema)


# ─────────────────────────────────────────────────────────────────────────────
#  Quick demo  (run:  python cast_ollama.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── 1. Build the singleton ───────────────────────────────────────────────
    llm = (
        CastOllama.builder()
        .set_model("qwen3:4b")
        .set_host("http://localhost:11434")
        .set_temperature(0.7)
        .set_context_length(32768)
        .set_max_tokens(1024)
        .set_system_prompt("You are a helpful assistant.")
        .enable_streaming(False)
        .enable_thinking(False)
        .enable_web_search(False)   # set api_key= for live search
        .set_log_level("INFO")
        .build()
    )

    log(llm)

    # ── 2. Register a custom tool ────────────────────────────────────────────
    @llm.tool(description="Get the current temperature for a city")
    def get_temperature(city: str) -> str:
        """
        Returns simulated temperature for a city.

        Args:
            city (str): The city name.
        """
        fake = {"Dhaka": "34°C", "London": "12°C", "Tokyo": "21°C"}
        return fake.get(city, "22°C")

    # ── 3. Simple Q&A ────────────────────────────────────────────────────────
    if llm.is_server_running():
        log("\n─── Simple chat ───")
        resp = llm.chat("What is the temperature in Dhaka?")
        log(f"Content:  {resp.content}")
        log(f"Thinking: {resp.thinking}")

        # ── 4. Switch model on the fly ───────────────────────────────────────
        llm.switch_model("llama3.2")
        log(f"\nSwitched to: {llm.get_model()}")

        # ── 5. Streaming demo ────────────────────────────────────────────────
        llm.enable_streaming(True)
        log("\n─── Streaming chat ───")
        for chunk in llm.chat("Write one sentence about Bangladesh."):
            if chunk.content:
                log(chunk.content, end="", flush=True)
        log()

        # ── 6. Server health ─────────────────────────────────────────────────
        log("\n─── Server health ───")
        log(json.dumps(llm.server_health(), indent=2))

    else:
        log(
            "\n⚠  Ollama server not running.  "
            "Start it with: llm.start_server()"
        )