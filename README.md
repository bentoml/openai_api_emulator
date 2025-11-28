# OpenAI API Emulator

A BentoML-based service that emulates OpenAI's Chat Completion and Models APIs with customizable timing parameters.

## Features

- **Chat Completions API** (`/v1/chat/completions`)
  - Non-streaming responses (`stream: false`)
  - Streaming responses (`stream: true`) with Server-Sent Events
  - Compatible with OpenAI API format

- **Models API** (`/v1/models`)
  - Returns list of available model names
  - Compatible with OpenAI API format

- **Multimodal Support (GPT-4 Vision)**
  - Support for text + image inputs (base64 and URL formats)
  - Compatible with OpenAI GPT-4 Vision API format
  - Images are accepted but not actually processed (mock responses)

- **Precise Token Counting with TikToken**
  - Uses OpenAI's official `tiktoken` library for accurate output token counting
  - Exact token-level control for response lengths
  - True token-by-token streaming (not word-based)
  - Input tokens use simple estimation, output tokens are precise

- **Customizable Timing Parameters**
  - `X-TTFT-MS`: Time To First Token in milliseconds
  - `X-ITL-MS`: Inter-Token Latency in milliseconds
    - **Streaming**: Delay between each actual token
    - **Non-streaming**: Per-token processing delay (total = ITL × output_length)
  - `X-OUTPUT-LENGTH`: Output length in exact tokens (not approximate)

- **Health Check** (`/health`)
  - Service health monitoring endpoint

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Start the Server

```bash
bentoml serve service.py:OpenAIEmulator
```

The server will start on `http://localhost:3000` by default.

### API Endpoints

#### 1. Chat Completions (Non-streaming)

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-TTFT-MS: 200" \
  -H "X-ITL-MS: 50" \
  -H "X-OUTPUT-LENGTH: 25" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": false
  }'
```

#### 2. Chat Completions (Streaming)

```bash
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-TTFT-MS: 150" \
  -H "X-ITL-MS: 75" \
  -H "X-OUTPUT-LENGTH: 30" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

#### 3. Models API

```bash
curl http://localhost:3000/v1/models
```

#### 4. Multimodal Requests (Images)

```bash
# Base64 image
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-OUTPUT-LENGTH: 30" \
  -d '{
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请描述这张图片"},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
            }
          }
        ]
      }
    ],
    "stream": false
  }'

# URL image
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-OUTPUT-LENGTH: 25" \
  -d '{
    "model": "gpt-4-vision-preview",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请描述这张图片"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://example.com/cat.png",
              "detail": "high"
            }
          }
        ]
      }
    ],
    "stream": false
  }'
```

#### 5. Health Check

```bash
curl http://localhost:3000/health
```

### Timing Parameters

| Header | Description | Default | Example |
|--------|-------------|---------|---------|
| `X-TTFT-MS` | Time to first token (ms) | 100 | 200 |
| `X-ITL-MS` | **Streaming**: delay between tokens<br>**Non-streaming**: per-token processing delay | 50 | 75 |
| `X-OUTPUT-LENGTH` | Output length in **exact tokens** using tiktoken | 20 | 30 |

**Timing Calculation Examples:**
- **Streaming** (30 tokens, TTFT=200ms, ITL=75ms): 200ms + (29 × 75ms) = ~2.4s total
- **Non-streaming** (30 tokens, TTFT=200ms, ITL=75ms): 200ms + (30 × 75ms) = ~2.5s total

### Available Models

Any model name works.

## Testing

### Manual Testing

Run the test scripts to verify endpoints:

```bash
# Test multimodal (image) requests
python test_multimodal.py

# Test all endpoints (if available)
python test_api.py
```

### Load Testing with Locust

Start the load test:

```bash
locust -f locustfile.py --host=http://localhost:3000
```

Then open http://localhost:8089 to configure and run load tests.

#### Available User Classes

1. **OpenAIEmulatorUser**: Comprehensive testing with various scenarios
2. **HighThroughputUser**: High-frequency requests for performance testing

### Example Usage with OpenAI Python Client

```python
import openai

# Configure client to use the emulator
client = openai.OpenAI(
    api_key="dummy-key",  # Any key works
    base_url="http://localhost:3000/v1"
)

# Non-streaming chat
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_headers={
        "X-TTFT-MS": "200",
        "X-OUTPUT-LENGTH": "25"
    }
)
print(response.choices[0].message.content)

# Streaming chat
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Tell me a joke"}],
    stream=True,
    extra_headers={
        "X-TTFT-MS": "150",
        "X-ITL-MS": "75",
        "X-OUTPUT-LENGTH": "30"
    }
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")

# Multimodal (Vision) example
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    ],
    extra_headers={
        "X-OUTPUT-LENGTH": "30"
    }
)
print(response.choices[0].message.content)
```

## Response Format

### Non-streaming Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1699999999,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 25,
    "total_tokens": 35
  }
}
```

### Streaming Response

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699999999,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699999999,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"Hello "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699999999,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{"content":"there! "},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1699999999,"model":"gpt-3.5-turbo","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Development

### Project Structure

```
openai_api_emulator/
├── service.py          # Main BentoML service
├── locustfile.py      # Load testing scenarios
├── test_api.py        # Manual test script
├── requirements.txt   # Dependencies
├── bentofile.yaml     # BentoML configuration
└── README.md          # This file
```

### Adding New Features

1. Modify `service.py` to add new endpoints or functionality
2. Update `locustfile.py` to test new features
3. Update `test_api.py` for manual verification
4. Update this README with usage examples

## Troubleshooting

### Common Issues

1. **Port already in use**: Change the port with `bentoml serve service.py:OpenAIEmulator --port 3001`
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Streaming not working**: Check that your client properly handles Server-Sent Events

### Logs

BentoML logs are available in the console when running the service. For debugging timing issues, look for the actual sleep durations in the logs.
