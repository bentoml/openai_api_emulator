import asyncio
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
import random

import bentoml
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import tiktoken


# Pydantic models for request/response
class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"  # auto, low, high


class ContentItem(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class Message(BaseModel):
    role: str
    content: Optional[str | List[ContentItem]] = (
        None  # Support both string and array formats
    )


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


app = FastAPI()

my_image = bentoml.images.Image(python_version="3.11").requirements_file(
    "requirements.txt"
)


@bentoml.asgi_app(app)
@bentoml.service(
    image=my_image,
    name="openai_emulator",
    workers="cpu_count",
)
class OpenAIEmulator:
    def __init__(self):
        self.available_models = [
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

        # Sample response templates
        self.sample_responses = [
            "Hello! How can I assist you today?",
            "I'm here to help you with any questions you might have.",
            "That's an interesting question. Let me think about it.",
            "I understand what you're asking. Here's my response:",
            "Thank you for your question. I'd be happy to help.",
            "Based on the information provided, I can offer the following insights:",
            "Let me provide you with a comprehensive answer to your query.",
            "I appreciate you reaching out. Here's what I can tell you:",
        ]

        # Initialize tiktoken encoder for GPT models
        try:
            self.encoding = tiktoken.get_encoding(
                "cl100k_base"
            )  # Used by GPT-3.5 and GPT-4
        except Exception:
            # Fallback if tiktoken fails to initialize
            self.encoding = None

    def _get_timing_params(self, request: Request) -> tuple[float, float, int]:
        """Extract timing parameters from headers"""
        ttft_ms = float(request.headers.get("X-TTFT-MS", 100))  # Default 100ms
        itl_ms = float(request.headers.get("X-ITL-MS", 50))  # Default 50ms
        output_length = int(
            request.headers.get("X-OUTPUT-LENGTH", 20)
        )  # Default 20 tokens

        return ttft_ms / 1000.0, itl_ms / 1000.0, output_length

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        if self.encoding is None:
            # Fallback to character-based estimation
            return len(text) // 4

        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback if tiktoken encoding fails
            return len(text) // 4

    def _generate_response_content(self, target_tokens: int) -> str:
        """Generate response content with exact token count using tiktoken"""

        # Start with a base response
        base_response = random.choice(self.sample_responses)
        current_content = base_response

        # If we already have enough tokens, truncate
        current_tokens = self._count_tokens(current_content)
        if current_tokens >= target_tokens:
            # Truncate by encoding and decoding exact number of tokens
            if self.encoding is not None:
                try:
                    encoded = self.encoding.encode(current_content)
                    truncated = encoded[:target_tokens]
                    return self.encoding.decode(truncated)
                except Exception:
                    pass

            # Fallback: truncate by words
            words = current_content.split()
            estimated_words = max(1, target_tokens // 1.3)
            return " ".join(words[: int(estimated_words)])

        # Extend content to reach target tokens
        filler_phrases = [
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

        extension_templates = [
            "this is a very interesting topic that deserves careful consideration",
            "there are many aspects to explore in this particular area of discussion",
            "we can approach this from multiple different perspectives and viewpoints",
            "the implications of this are quite significant and far-reaching in nature",
            "this subject matter has various nuances that are worth examining closely",
            "there are several factors that contribute to the overall understanding here",
            "the complexity of this issue requires thorough analysis and careful thought",
        ]

        while current_tokens < target_tokens:
            # Add a filler phrase
            filler = random.choice(filler_phrases)
            extension = random.choice(extension_templates)
            addition = f" {filler} {extension}."

            # Check if adding this would exceed target
            addition_tokens = self._count_tokens(addition)
            if current_tokens + addition_tokens <= target_tokens:
                current_content += addition
                current_tokens += addition_tokens
            else:
                # Add partial content to reach exact target
                remaining_tokens = target_tokens - current_tokens
                if remaining_tokens > 0:
                    if self.encoding is not None:
                        try:
                            # Encode the addition and take only what we need
                            encoded_addition = self.encoding.encode(addition)
                            truncated_addition = encoded_addition[:remaining_tokens]
                            partial_addition = self.encoding.decode(truncated_addition)
                            current_content += partial_addition
                        except Exception:
                            # Fallback: add a simple word
                            current_content += " more"
                    else:
                        # Fallback: add a simple word
                        current_content += " more"
                break

        return current_content.strip()

    async def _stream_response(
        self,
        request_data: ChatCompletionRequest,
        ttft: float,
        itl: float,
        output_length: int,
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response chunks"""
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        # Wait for TTFT before first token
        await asyncio.sleep(ttft)

        # Generate content
        content = self._generate_response_content(output_length)
        words = content.split()

        # First chunk with role
        first_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=request_data.model,
            choices=[
                DeltaChoice(
                    index=0,
                    delta={"role": "assistant", "content": ""},
                    finish_reason=None,
                )
            ],
        )
        yield f"data: {first_chunk.model_dump_json()}\n\n"

        # Stream tokens (using tiktoken for more accurate tokenization)
        if self.encoding is not None:
            try:
                # Encode content to get actual tokens
                tokens = self.encoding.encode(content)

                # Stream each token
                for i, token in enumerate(tokens):
                    if i > 0:  # Wait ITL between tokens (except first)
                        await asyncio.sleep(itl)

                    # Decode single token to text
                    token_text = self.encoding.decode([token])

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request_data.model,
                        choices=[
                            DeltaChoice(
                                index=0,
                                delta={"content": token_text},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
            except Exception:
                # Fallback to word-based streaming
                words = content.split()
                for i, word in enumerate(words):
                    if i > 0:
                        await asyncio.sleep(itl)

                    chunk = ChatCompletionStreamResponse(
                        id=request_id,
                        created=created,
                        model=request_data.model,
                        choices=[
                            DeltaChoice(
                                index=0,
                                delta={"content": word + " "},
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {chunk.model_dump_json()}\n\n"
        else:
            # Fallback to word-based streaming
            words = content.split()
            for i, word in enumerate(words):
                if i > 0:
                    await asyncio.sleep(itl)

                chunk = ChatCompletionStreamResponse(
                    id=request_id,
                    created=created,
                    model=request_data.model,
                    choices=[
                        DeltaChoice(
                            index=0, delta={"content": word + " "}, finish_reason=None
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Final chunk
        final_chunk = ChatCompletionStreamResponse(
            id=request_id,
            created=created,
            model=request_data.model,
            choices=[DeltaChoice(index=0, delta={}, finish_reason="stop")],
        )
        yield f"data: {final_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    @app.post("/v1/chat/completions")
    async def chat_completions(self, request: Request):
        try:
            # Parse request
            body = await request.json()
            request_data = ChatCompletionRequest(**body)

            # Get timing parameters from headers
            ttft, itl, output_length = self._get_timing_params(request)

            if request_data.stream:
                # Return streaming response
                return StreamingResponse(
                    self._stream_response(request_data, ttft, itl, output_length),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Wait for TTFT before responding
                await asyncio.sleep(ttft)

                # Generate non-streaming response
                content = self._generate_response_content(output_length)

                # Simulate additional processing time based on ITL and output length
                # For non-streaming, ITL represents the per-token processing delay
                additional_delay = itl * output_length
                await asyncio.sleep(additional_delay)

                # Calculate token counts (only output needs to be accurate)
                prompt_tokens = (
                    len(str(request_data.messages)) // 4
                )  # Simple estimate for input
                completion_tokens = self._count_tokens(content)  # Accurate for output
                total_tokens = prompt_tokens + completion_tokens

                response = ChatCompletionResponse(
                    id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    created=int(time.time()),
                    model=request_data.model,
                    choices=[
                        Choice(
                            index=0,
                            message=Message(role="assistant", content=content),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    ),
                )

                return response.model_dump()

        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.get("/v1/models")
    async def models(self):
        """Return available models"""
        model_list = []
        for model_id in self.available_models:
            model_list.append(
                ModelInfo(id=model_id, created=int(time.time()), owned_by="openai")
            )

        response = ModelsResponse(data=model_list)
        return response.model_dump()

    @app.get("/health")
    async def health_check(self):
        """Health check endpoint"""
        return {"status": "healthy", "timestamp": int(time.time())}
