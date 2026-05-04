"""Benchmark backends."""

import argparse
import json
import logging
import os
import time
import traceback
from typing import Literal, Optional

from request_record import Metrics, RequestRecord, ServerMetrics
from typing_extensions import Self

logger = logging.getLogger(__name__)


class APIEndPoint:
    """Manages the sending of requests to a specified API endpoint and gathers
    inference statistics.
    """

    def __init__(self, include_server_metrics: bool = False) -> None:
        self.include_server_metrics = include_server_metrics

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        pass

    async def __call__(self, request: RequestRecord) -> RequestRecord:
        raise NotImplementedError()


class OpenAIChatEndPoint(APIEndPoint):
    """The backend of sending HTTP requests in OpenAI API through "v1/chat/completions"."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        host: str,
        port: int,
        api_type: Literal["sglang"],
        timeout: Optional[float] = None,
        include_server_metrics: bool = False,
    ) -> None:
        super().__init__(include_server_metrics=include_server_metrics)

        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.timeout = timeout
        self.client: aiohttp.ClientSession = None
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        api_key = os.getenv("SGLANG_API_KEY") or os.getenv("OPENAI_API_KEY")
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.api_type: str = api_type

    async def __aenter__(self) -> Self:
        import aiohttp  # pylint: disable=import-outside-toplevel,import-error

        self.client = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_value, tb) -> None:
        await self.client.close()

    async def __call__(  # pylint: disable=too-many-branches,too-many-statements,too-many-locals
        self, request_record: RequestRecord
    ) -> RequestRecord:
        payload = request_record.chat_cmpl.model_dump(exclude_none=True)
        payload.setdefault("stream", False)
        if self.timeout is not None and "timeout" not in payload:
            payload["timeout"] = self.timeout
        if self.include_server_metrics:
            if "stream_options" not in payload or payload["stream_options"] is None:
                payload["stream_options"] = {"include_usage": True}
            else:
                payload["stream_options"]["include_usage"] = True
        debug_config = payload.pop("debug_config", None)
        if debug_config is not None and debug_config.get("ignore_eos"):
            payload["ignore_eos"] = True

        response_format = payload.get("response_format")
        if (
            response_format is not None
            and response_format["type"] == "structural_tag"
            and "tags" in response_format
        ):
            response_format["structures"] = response_format.pop("tags")
            for tag in response_format["structures"]:
                if isinstance(tag.get("schema"), str):
                    tag["schema"] = json.loads(tag["schema"])

        generated_text = ""
        first_chunk_output_str = ""
        time_to_first_token_s = None
        start_time = time.monotonic()
        server_metrics = None

        try:
            async with self.client.post(self.url, json=payload, headers=self.headers) as response:
                assert response.status == 200, await response.text()
                if payload["stream"]:
                    async for chunk in response.content:
                        chunk = chunk.strip()
                        if not chunk or chunk == b"\n":
                            continue
                        # Get rid of the prefix "data: " and suffix "\n"
                        raw_data = chunk[6:].strip()
                        if raw_data == b"[DONE]":
                            continue
                        data = json.loads(raw_data)
                        if not data["choices"]:
                            continue
                        delta = data["choices"][0]["delta"]
                        content = delta.get("content", None)
                        if content is not None and not time_to_first_token_s:
                            time_to_first_token_s = time.monotonic() - start_time
                            first_chunk_output_str = content
                        usage = data.get("usage")
                        if self.include_server_metrics and usage is not None:
                            # fmt: off
                            # pylint: disable=line-too-long
                            server_metrics = ServerMetrics(
                                input_tokens=usage["extra"]["prompt_tokens"],
                                prefill_tokens=usage["extra"]["prefill_tokens"],
                                output_tokens=usage["extra"]["completion_tokens"],
                                end_to_end_latency_s=usage["extra"]["end_to_end_latency_s"],
                                prefill_tokens_per_s=usage["extra"]["prefill_tokens_per_s"],
                                inter_token_latency_s=usage["extra"]["inter_token_latency_s"],
                                time_per_output_token_s=1 / usage["extra"]["decode_tokens_per_s"],
                                time_to_first_token_s=usage["extra"]["ttft_s"],
                            )
                            # pylint: enable=line-too-long
                            # fmt: on

                        if content is not None:
                            generated_text += content
                else:
                    data = await response.json()
                    generated_text = data["choices"][0]["message"]["content"]
                    usage = data.get("usage")
                    if self.include_server_metrics and usage is not None:
                        # fmt: off
                        # pylint: disable=line-too-long
                        server_metrics = ServerMetrics(
                            input_tokens=usage["extra"]["prompt_tokens"],
                            prefill_tokens=usage["extra"]["prefill_tokens"],
                            output_tokens=usage["extra"]["completion_tokens"],
                            end_to_end_latency_s=usage["extra"]["end_to_end_latency_s"],
                            prefill_tokens_per_s=usage["extra"]["prefill_tokens_per_s"],
                            inter_token_latency_s=usage["extra"]["inter_token_latency_s"],
                            time_per_output_token_s=1 / usage["extra"]["decode_tokens_per_s"],
                            time_to_first_token_s=usage["extra"]["ttft_s"],
                        )
                        # pylint: enable=line-too-long
                        # fmt: on
        except Exception:  # pylint: disable=broad-except
            error_msg = "API endpoint errored when sending request: " + traceback.format_exc()
            logger.info(error_msg)
            finish_time = time.monotonic()
            request_record.output_str = generated_text
            request_record.first_chunk_output_str = first_chunk_output_str
            request_record.metrics = Metrics(
                success=False,
                start_time=start_time,
                finish_time=finish_time,
                end_to_end_latency_s=finish_time - start_time,
                input_tokens=request_record.metrics.input_tokens,
                time_to_first_token_s=time_to_first_token_s,
                server_metrics=server_metrics,
                exec_feature=request_record.metrics.exec_feature,
            )
            request_record.error_msg = error_msg
            return request_record

        finish_time = time.monotonic()
        request_record.output_str = generated_text
        request_record.first_chunk_output_str = first_chunk_output_str
        success = True
        error_msg = None
        if len(generated_text) == 0:
            success = False
            error_msg = "Empty generated text."
        request_record.metrics = Metrics(
            success=success,
            start_time=start_time,
            finish_time=finish_time,
            end_to_end_latency_s=finish_time - start_time,
            input_tokens=request_record.metrics.input_tokens,
            time_to_first_token_s=time_to_first_token_s,
            server_metrics=server_metrics,
            exec_feature=request_record.metrics.exec_feature,
        )
        request_record.error_msg = error_msg
        return request_record


SUPPORTED_BACKENDS = ["sglang"]


def create_api_endpoint(args: argparse.Namespace) -> APIEndPoint:
    """Create an API endpoint instance with regard to the specified endpoint kind."""
    if args.api_endpoint == "sglang":
        return OpenAIChatEndPoint(
            args.host, args.port, args.api_endpoint, args.timeout, args.include_server_metrics
        )
    raise ValueError(f'Unrecognized endpoint "{args.api_endpoint}"')
