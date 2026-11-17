"""Talk to Gemma 3 running in Ollama (on the college server, via SSH tunnel)."""
import ollama

import config

_client = ollama.Client(host=config.OLLAMA_HOST)


def ask_llm(system: str, prompt: str) -> str:
    try:
        response = _client.chat(
            model=config.OLLAMA_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            options={"temperature": config.TEMPERATURE, "num_ctx": config.CONTEXT_WINDOW},
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {config.OLLAMA_HOST} (model '{config.OLLAMA_MODEL}'). "
            "Is the SSH tunnel running and is the model name correct? "
            f"Details: {exc}"
        ) from exc
    return response.message.content.strip()
