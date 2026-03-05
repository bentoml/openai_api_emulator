import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
import random

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tiktoken


# Pydantic models for request/response
class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: Optional[str | List[ContentItem]] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 1.0


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class DeltaChoice(BaseModel):
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[DeltaChoice]


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


AVAILABLE_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-0301",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-32k",
    "text-davinci-003",
    "text-davinci-002",
]

SAMPLE_RESPONSES = [
    "Hello! How can I assist you today?",
    "I'm here to help you with any questions you might have.",
    "That's an interesting question. Let me think about it.",
    "I understand what you're asking. Here's my response:",
    "Thank you for your question. I'd be happy to help.",
    "Based on the information provided, I can offer the following insights:",
    "Let me provide you with a comprehensive answer to your query.",
    "I appreciate you reaching out. Here's what I can tell you:",
]

FILLER_PHRASES = [
    "Additionally, I want to mention that",
    "Furthermore, it's important to note that",
    "Moreover, we should consider that",
    "In fact, this reminds me that",
    "It's worth noting that",
    "Please also consider that",
    "Also, I should add that",
    "On a related note,",
    "To elaborate further,",
    "In this context,",
]

EXTENSION_TEMPLATES = [
    "this is a very interesting topic that deserves careful consideration",
    "there are many aspects to explore in this particular area of discussion",
    "we can approach this from multiple different perspectives and viewpoints",
    "the implications of this are quite significant and far-reaching in nature",
    "this subject matter has various nuances that are worth examining closely",
    "there are several factors that contribute to the overall understanding here",
    "the complexity of this issue requires thorough analysis and careful thought",
]

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None


def _get_timing_params(request: Request) -> tuple[float, float, int, int]:
    ttft_ms = float(request.headers.get("X-TTFT-MS", 100))
    itl_ms = float(request.headers.get("X-ITL-MS", 50))
    output_length = int(request.headers.get("X-OUTPUT-LENGTH", 20))
    sse_batch_size = int(request.headers.get("X-SSE-BATCH-SIZE", 4))
    return ttft_ms / 1000.0, itl_ms / 1000.0, output_length, sse_batch_size


def _count_tokens(text: str) -> int:
    if encoding is None:
        return len(text) // 4
    try:
        return len(encoding.encode(text))
    except Exception:
        return len(text) // 4


def _generate_response_content(target_tokens: int) -> str:
    base_response = random.choice(SAMPLE_RESPONSES)
    current_content = base_response

    current_tokens = _count_tokens(current_content)
    if current_tokens >= target_tokens:
        if encoding is not None:
            try:
                encoded = encoding.encode(current_content)
                return encoding.decode(encoded[:target_tokens])
            except Exception:
                pass
        words = current_content.split()
        return " ".join(words[: max(1, int(target_tokens // 1.3))])

    while current_tokens < target_tokens:
        filler = random.choice(FILLER_PHRASES)
        extension = random.choice(EXTENSION_TEMPLATES)
        addition = f" {filler} {extension}."
        addition_tokens = _count_tokens(addition)

        if current_tokens + addition_tokens <= target_tokens:
            current_content += addition
            current_tokens += addition_tokens
        else:
            remaining = target_tokens - current_tokens
            if remaining > 0 and encoding is not None:
                try:
                    encoded_addition = encoding.encode(addition)
                    partial = encoding.decode(encoded_addition[:remaining])
                    current_content += partial
                except Exception:
                    current_content += " more"
            else:
                current_content += " more"
            break

    return current_content.strip()


async def _stream_response(
    request_data: ChatCompletionRequest,
    ttft: float,
    itl: float,
    output_length: int,
    sse_batch_size: int,
) -> AsyncGenerator[str, None]:
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    await asyncio.sleep(ttft)

    content = _generate_response_content(output_length)

    first_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=request_data.model,
        choices=[DeltaChoice(index=0, delta={"role": "assistant", "content": ""}, finish_reason=None)],
    )
    yield f"data: {first_chunk.model_dump_json()}\n\n"

    if encoding is not None:
        try:
            tokens = encoding.encode(content)
            for i in range(0, len(tokens), sse_batch_size):
                if i > 0:
                    await asyncio.sleep(itl)
                batch_text = encoding.decode(tokens[i : i + sse_batch_size])
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=request_data.model,
                    choices=[DeltaChoice(index=0, delta={"content": batch_text}, finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        except Exception:
            words = content.split()
            for i in range(0, len(words), sse_batch_size):
                if i > 0:
                    await asyncio.sleep(itl)
                batch_text = " ".join(words[i : i + sse_batch_size]) + " "
                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=request_data.model,
                    choices=[DeltaChoice(index=0, delta={"content": batch_text}, finish_reason=None)],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
    else:
        words = content.split()
        for i in range(0, len(words), sse_batch_size):
            if i > 0:
                await asyncio.sleep(itl)
            batch_text = " ".join(words[i : i + sse_batch_size]) + " "
            chunk = ChatCompletionStreamResponse(
                id=request_id,
                created=created,
                model=request_data.model,
                choices=[DeltaChoice(index=0, delta={"content": batch_text}, finish_reason=None)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    final_chunk = ChatCompletionStreamResponse(
        id=request_id,
        created=created,
        model=request_data.model,
        choices=[DeltaChoice(index=0, delta={}, finish_reason="stop")],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


app = FastAPI(title="OpenAI Emulator")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        body = await request.json()
        request_data = ChatCompletionRequest(**body)
        ttft, itl, output_length, sse_batch_size = _get_timing_params(request)

        if request_data.stream:
            return StreamingResponse(
                _stream_response(request_data, ttft, itl, output_length, sse_batch_size),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        await asyncio.sleep(ttft)
        content = _generate_response_content(output_length)
        await asyncio.sleep(itl * output_length)

        prompt_tokens = len(str(request_data.messages)) // 4
        completion_tokens = _count_tokens(content)

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request_data.model,
            choices=[Choice(index=0, message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )
        return response.model_dump()

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/models")
async def models():
    data = [ModelInfo(id=m, created=int(time.time()), owned_by="openai") for m in AVAILABLE_MODELS]
    return ModelsResponse(data=data).model_dump()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": int(time.time())}
