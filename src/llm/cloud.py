from __future__ import annotations

from config import SETTINGS
from llm.client import GenResult, LLMClient


class CloudGPU:
    def __init__(self) -> None:
        self.is_mock = not SETTINGS.models.cloud_url
        url = SETTINGS.models.cloud_url or SETTINGS.models.edge_url
        api_key = SETTINGS.models.cloud_api_key if SETTINGS.models.cloud_url else None
        self._client = LLMClient(url, api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 384,
        reasoning: str | None = None,
    ) -> GenResult:
        return self._client.chat(
            SETTINGS.models.cloud_model,
            messages,
            max_tokens=max_tokens,
            reasoning=reasoning,
            logprobs=False,
        )

    def close(self) -> None:
        self._client.close()
