# llm.py

from typing import TypedDict, Any, Dict, Generator
import requests
import json
import base64
from pathlib import Path
from typing import Dict, Generator, Optional, Any, Union

class ModelList(TypedDict):
    DEEP_SEEK_R1_8B= 'deepseek-r1:8b',
    GEMMA3_12B= 'gemma3:12b' # ? Overall Good
    QWEN3_8B= 'qwen3-vl:8b' # ! LONG THINKER
    LLAMA_3B= 'llama3.2:3b' # ? Smart One
    QWEN25_CODER_7= 'qwen2.5-coder:7b' # ? MR.CODER


def chat_with_ollama(config: Dict[str, Any]) -> Any:
    """
    Chat with local Ollama LLM following OpenAI-style API patterns.

    Args:
        config (dict): Configuration dictionary with the following parameters:
            Required:
                - system_prompt (str): System message/instructions
                - user_prompt (str): User's message/question
                - model (str): Model name (e.g., 'llama3.2', 'mistral', 'llava')

            Optional:
                - response_format (dict|str): Structure for response formatting
                  Options:
                    1. String: "json" for basic JSON mode
                    2. Dict with OpenAI style: {"type": "json_object"}
                    3. Dict with JSON schema: {"type": "object", "properties": {...}}
                - file_path (str): Path to image/file/video/audio to include with prompt
                - stream_reasoning_response (bool): Enable streaming with reasoning
                - thinking (bool): Enable reasoning/thinking mode (default: False)
                - enable_web_search (bool): Enable web search tool capability
                - search_api (str): Search API to use ('duckduckgo', 'google', 'serper')
                - search_api_key (str): API key for search service (if required)
                - temperature (float): Sampling temperature (default: 0.7)
                - max_tokens (int): Maximum tokens to generate
                - top_p (float): Nucleus sampling parameter

    Returns:
        str or Generator: Response text or streaming generator

    Raises:
        ValueError: If required parameters are missing
        FileNotFoundError: If file_path doesn't exist
        requests.RequestException: If API request fails
    """

    # Validate required parameters
    required_params = ['system_prompt', 'user_prompt', 'model']
    for param in required_params:
        if param not in config:
            raise ValueError(f"Missing required parameter: '{param}'")

    # Ollama API endpoint
    ollama_url = "http://localhost:11434/api/chat"

    # Build messages array (OpenAI format)
    messages = [
        {
            "role": "system",
            "content": config['system_prompt']
        },
        {
            "role": "user",
            "content": config['user_prompt']
        }
    ]

    # Handle file attachment (image/video/audio for vision models)
    if 'file_path' in config and config['file_path']:
        file_path = Path(config['file_path'])

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {config['file_path']}")

        # Read and encode file as base64
        with open(file_path, 'rb') as f:
            file_data = base64.b64encode(f.read()).decode('utf-8')

        # Update user message to include image (Ollama format)
        messages[-1] = {
            "role": "user",
            "content": config['user_prompt'],
            "images": [file_data]
        }

    # Build request payload
    payload = {
        "model": config['model'],
        "messages": messages,
        "stream": config.get('stream_reasoning_response', False)
    }

    # Add optional parameters
    options = {}
    if 'temperature' in config:
        options['temperature'] = config['temperature']
    if 'top_p' in config:
        options['top_p'] = config['top_p']
    if 'max_tokens' in config:
        options['num_predict'] = config['max_tokens']

    if options:
        payload['options'] = options

    # Handle thinking/reasoning mode
    if 'thinking' in config:
        payload['thinking'] = config['thinking']

    # Handle web search tools
    if config.get('enable_web_search', False):
        tools = _build_search_tools(config)
        if tools:
            payload['tools'] = tools

    # Handle response_format - convert to Ollama's 'format' parameter
    if 'response_format' in config:
        payload['format'] = _convert_response_format(config['response_format'])

    # Handle streaming with reasoning
    if config.get('stream_reasoning_response', False):
        return _stream_response(ollama_url, payload)

    # Non-streaming request
    try:
        response = requests.post(ollama_url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()
        return result['message']['content']

    except requests.RequestException as e:
        raise requests.RequestException(f"Ollama API request failed: {str(e)}")


def _convert_response_format(response_format: Union[str, Dict]) -> Union[str, Dict]:
    """
    Convert response_format to Ollama's format parameter.

    Ollama format accepts:
    1. String: "json" for basic JSON mode
    2. JSON Schema object: {"type": "object", "properties": {...}}
    3. JSON Schema array: {"type": "array", "items": {...}}

    Args:
        response_format: Can be:
            - str: "json"
            - dict: {"type": "json_object"} (OpenAI style)
            - dict: JSON schema object or array

    Returns:
        Ollama-compatible format parameter
    """
    # If it's already a string, return as is
    if isinstance(response_format, str):
        return response_format

    # If it's a dict
    if isinstance(response_format, dict):
        # OpenAI style: {"type": "json_object"}
        if response_format.get('type') == 'json_object':
            return 'json'

        # OpenAI style with schema: {"type": "json_schema", "json_schema": {...}}
        if response_format.get('type') == 'json_schema':
            return response_format.get('json_schema', 'json')

        # Direct JSON schema - object: {"type": "object", "properties": {...}}
        if response_format.get('type') == 'object' and 'properties' in response_format:
            return response_format

        # Direct JSON schema - array: {"type": "array", "items": {...}}
        if response_format.get('type') == 'array' and 'items' in response_format:
            # Convert Python False to JSON false for additionalProperties
            return _sanitize_json_schema(response_format)

        # Fallback to json
        return 'json'

    return 'json'


def _sanitize_json_schema(schema: Dict) -> Dict:
    """
    Recursively convert Python booleans to JSON-compatible values.
    Converts False -> false, True -> true in the schema.

    Args:
        schema: JSON schema dict potentially containing Python booleans

    Returns:
        Sanitized schema with JSON-compatible values
    """
    if isinstance(schema, dict):
        return {k: _sanitize_json_schema(v) for k, v in schema.items()}
    elif isinstance(schema, list):
        return [_sanitize_json_schema(item) for item in schema]
    elif isinstance(schema, bool):
        # Keep as is - requests library will handle JSON serialization
        return schema
    else:
        return schema


def _stream_response(url: str, payload: Dict) -> Generator[str, None, None]:
    """
    Internal method to handle streaming responses character by character.

    Args:
        url (str): Ollama API endpoint
        payload (dict): Request payload

    Yields:
        str: Individual characters from the response stream
    """
    try:
        with requests.post(url, json=payload, stream=True, timeout=300) as response:
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line.decode('utf-8'))

                    # Extract content from the chunk
                    if 'message' in chunk and 'content' in chunk['message']:
                        content = chunk['message']['content']

                        # Yield character by character
                        for char in content:
                            yield char

                    # Check if streaming is done
                    if chunk.get('done', False):
                        break

    except requests.RequestException as e:
        raise requests.RequestException(f"Streaming request failed: {str(e)}")


def _build_search_tools(config: Dict) -> list[Dict]:
    """
    Build web search tool definitions for Ollama.

    Args:
        config: Configuration dictionary

    Returns:
        List of tool definitions
    """
    search_api = config.get('search_api', 'duckduckgo')

    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information and real-time data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to find information"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    return tools


def web_search(query: str, search_api: str = 'duckduckgo', api_key: str = None) -> str:
    """
    Perform web search using specified search API.

    Args:
        query: Search query string
        search_api: Search API to use ('duckduckgo', 'google', 'serper')
        api_key: API key for the search service (if required)

    Returns:
        Search results as string
    """
    try:
        if search_api == 'duckduckgo':
            # DuckDuckGo Instant Answer API (no key required)
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
                timeout=10
            )
            data = response.json()

            # Combine abstract and related topics
            result = data.get("AbstractText", "")
            if not result and data.get("RelatedTopics"):
                topics = [t.get("Text", "") for t in data.get("RelatedTopics", [])[:3]]
                result = " ".join(topics)

            return result if result else "No results found."

        elif search_api == 'google' and api_key:
            # Google Custom Search API (requires API key)
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": api_key,
                    "cx": config.get('google_cx'),  # Custom search engine ID
                    "q": query
                },
                timeout=10
            )
            data = response.json()
            items = data.get("items", [])[:3]
            results = [f"{item['title']}: {item['snippet']}" for item in items]
            return "\n".join(results) if results else "No results found."

        elif search_api == 'serper' and api_key:
            # Serper.dev API (requires API key)
            response = requests.post(
                "https://google.serper.dev/search",
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json"
                },
                json={"q": query},
                timeout=10
            )
            data = response.json()
            organic = data.get("organic", [])[:3]
            results = [f"{item['title']}: {item['snippet']}" for item in organic]
            return "\n".join(results) if results else "No results found."

        else:
            return "Search API not configured or API key missing."

    except Exception as e:
        return f"Search failed: {str(e)}"

