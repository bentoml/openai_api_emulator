from locust import task, between, HttpUser
import json
import base64
import random


class OpenAIEmulatorUser(HttpUser):
    """
    Simulates a user that sends requests to the OpenAI API emulator.
    Tests both streaming and non-streaming chat completions with various timing parameters.
    """

    wait_time = between(1, 3)

    # Static variables for image generation
    IMAGE_BASE64_SIZE = 1024  # Size in bytes for generated base64 images
    MULTIMODAL_TEST_RATIO = 0.3  # 30% of requests will include images

    def on_start(self):
        """Initialize test data"""
        self.models = [
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-3.5-turbo-16k"
        ]

        self.sample_messages = [
            [{"role": "user", "content": "Hello, how are you?"}],
            [{"role": "user", "content": "Can you explain quantum computing?"}],
            [{"role": "user", "content": "Write a short story about a robot."}],
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": "What is the meaning of life?"}]
        ]

        # Vision prompts for multimodal testing
        self.vision_prompts = [
            "请描述这张图片",
            "这张图片里有什么？",
            "分析这张图片的内容",
            "What do you see in this image?",
            "Describe the details in this picture",
            "What objects are visible in this image?"
        ]

    def _generate_fake_image_base64(self) -> str:
        """Generate a fake base64 image of specified size"""
        # Create random bytes of the desired size
        fake_image_data = bytes([random.randint(0, 255) for _ in range(self.IMAGE_BASE64_SIZE)])
        # Encode to base64
        b64_data = base64.b64encode(fake_image_data).decode('utf-8')
        return b64_data

    def _create_multimodal_message(self, text_prompt: str) -> dict:
        """Create a multimodal message with text + image"""
        image_b64 = self._generate_fake_image_base64()

        # Random choice between base64 and URL format
        if random.choice([True, False]):
            # Base64 format
            image_url = f"data:image/png;base64,{image_b64}"
        else:
            # URL format (fake URL)
            image_url = f"https://example.com/test_image_{random.randint(1000, 9999)}.jpg"

        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": random.choice(["low", "high", "auto"])
                    }
                }
            ]
        }

    @task(3)
    def test_chat_completion_non_stream(self):
        """Test non-streaming chat completion (text + occasional images)"""

        # Decide whether to include an image
        include_image = random.random() < self.MULTIMODAL_TEST_RATIO

        if include_image:
            # Use vision model and create multimodal message
            model = random.choice(["gpt-4-vision-preview", "gpt-4-turbo-2024-04-09"])
            vision_prompt = random.choice(self.vision_prompts)
            messages = [self._create_multimodal_message(vision_prompt)]
        else:
            # Regular text-only message
            model = random.choice(self.models)
            messages = random.choice(self.sample_messages)

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": random.randint(10, 100)
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": str(random.randint(50, 200)),    # TTFT: 50-200ms
            "X-ITL-MS": str(random.randint(20, 100)),     # ITL: 20-100ms
            "X-OUTPUT-LENGTH": str(random.randint(10, 50)) # Output: 10-50 tokens
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="chat_completion_non_stream",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def test_chat_completion_stream(self):
        """Test streaming chat completion (text + occasional images)"""

        # Decide whether to include an image
        include_image = random.random() < self.MULTIMODAL_TEST_RATIO

        if include_image:
            # Use vision model and create multimodal message
            model = random.choice(["gpt-4-vision-preview", "gpt-4-turbo-2024-04-09"])
            vision_prompt = random.choice(self.vision_prompts)
            messages = [self._create_multimodal_message(vision_prompt)]
        else:
            # Regular text-only message
            model = random.choice(self.models)
            messages = random.choice(self.sample_messages)

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "max_tokens": random.randint(10, 100)
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": str(random.randint(100, 300)),    # TTFT: 100-300ms
            "X-ITL-MS": str(random.randint(30, 80)),       # ITL: 30-80ms
            "X-OUTPUT-LENGTH": str(random.randint(15, 40)) # Output: 15-40 tokens
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="chat_completion_stream",
            stream=True,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    # Read streaming response
                    chunk_count = 0
                    for line in response.iter_lines():
                        if line:
                            line_text = line.decode('utf-8')
                            if line_text.startswith('data: '):
                                chunk_count += 1
                                if line_text == 'data: [DONE]':
                                    break

                    if chunk_count > 0:
                        response.success()
                    else:
                        response.failure("No streaming chunks received")
                except Exception as e:
                    response.failure(f"Streaming error: {str(e)}")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_models_endpoint(self):
        """Test the models endpoint"""
        with self.client.get(
            "/v1/models",
            name="models_endpoint",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "data" in data and isinstance(data["data"], list):
                        response.success()
                    else:
                        response.failure("Invalid models response format")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_health_check(self):
        """Test health check endpoint"""
        with self.client.get(
            "/health",
            name="health_check",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "status" in data and data["status"] == "healthy":
                        response.success()
                    else:
                        response.failure("Unhealthy status")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_timing_parameters(self):
        """Test specific timing parameter scenarios"""
        import random

        # Test high TTFT scenario
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Test message"}],
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": "500",  # High TTFT
            "X-ITL-MS": "25",    # Low ITL
            "X-OUTPUT-LENGTH": "15"
        }

        start_time = self.environment.runner.start_time

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="timing_test_high_ttft",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(1)
    def test_multimodal_only(self):
        """Test pure multimodal requests with various image sizes"""

        # Randomly adjust image size for this test
        original_size = self.IMAGE_BASE64_SIZE
        test_sizes = [512, 1024, 2048, 4096]  # Different image sizes to test
        self.IMAGE_BASE64_SIZE = random.choice(test_sizes)

        try:
            model = random.choice(["gpt-4-vision-preview", "gpt-4-turbo-2024-04-09"])
            vision_prompt = random.choice(self.vision_prompts)

            # Create message with multiple images sometimes
            if random.random() < 0.3:  # 30% chance of multiple images
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "比较这些图片"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self._generate_fake_image_base64()}",
                                "detail": "high"
                            }
                        },
                        {"type": "text", "text": "和"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"https://example.com/image_{random.randint(1000,9999)}.jpg",
                                "detail": "low"
                            }
                        }
                    ]
                }]
            else:
                messages = [self._create_multimodal_message(vision_prompt)]

            payload = {
                "model": model,
                "messages": messages,
                "stream": random.choice([True, False])
            }

            headers = {
                "Content-Type": "application/json",
                "X-TTFT-MS": str(random.randint(100, 400)),
                "X-ITL-MS": str(random.randint(30, 100)),
                "X-OUTPUT-LENGTH": str(random.randint(20, 60))
            }

            with self.client.post(
                "/v1/chat/completions",
                json=payload,
                headers=headers,
                name="multimodal_only",
                stream=payload["stream"],
                catch_response=True
            ) as response:
                if response.status_code == 200:
                    if payload["stream"]:
                        # Handle streaming response
                        chunk_count = 0
                        for line in response.iter_lines():
                            if line and line.decode('utf-8').startswith('data: '):
                                chunk_count += 1
                                if line.decode('utf-8') == 'data: [DONE]':
                                    break
                        if chunk_count > 0:
                            response.success()
                        else:
                            response.failure("No streaming chunks")
                    else:
                        # Handle non-streaming response
                        try:
                            data = response.json()
                            if "choices" in data:
                                response.success()
                            else:
                                response.failure("Invalid response")
                        except:
                            response.failure("Invalid JSON")
                else:
                    response.failure(f"HTTP {response.status_code}")

        finally:
            # Restore original image size
            self.IMAGE_BASE64_SIZE = original_size


class HighThroughputUser(HttpUser):
    """
    User class for testing high throughput scenarios
    """
    wait_time = between(0.1, 0.5)

    @task
    def rapid_requests(self):
        """Send rapid requests to test concurrency"""
        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": "Quick test"}],
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": "50",
            "X-ITL-MS": "10",
            "X-OUTPUT-LENGTH": "5"
        }

        self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name="rapid_request"
        )


class LargeImageUser(HttpUser):
    """
    User class specifically for testing large image payloads
    """
    wait_time = between(2, 5)

    # Large image configurations
    LARGE_IMAGE_SIZES = [8192, 16384, 32768]  # 8KB, 16KB, 32KB base64 images

    def on_start(self):
        """Initialize large image test data"""
        self.vision_models = ["gpt-4-vision-preview", "gpt-4-turbo-2024-04-09"]
        self.large_image_prompts = [
            "详细分析这张高分辨率图片",
            "Please provide a comprehensive analysis of this high-quality image",
            "描述图片中的所有细节",
            "What are all the objects and details you can identify in this image?"
        ]

    def _generate_large_image_base64(self, size_bytes: int) -> str:
        """Generate a large fake base64 image"""
        fake_data = bytes([random.randint(0, 255) for _ in range(size_bytes)])
        return base64.b64encode(fake_data).decode('utf-8')

    @task
    def test_large_image_request(self):
        """Test with large image payloads"""
        image_size = random.choice(self.LARGE_IMAGE_SIZES)
        model = random.choice(self.vision_models)
        prompt = random.choice(self.large_image_prompts)

        large_b64 = self._generate_large_image_base64(image_size)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{large_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "X-TTFT-MS": str(random.randint(200, 600)),  # Higher TTFT for large images
            "X-OUTPUT-LENGTH": str(random.randint(40, 80))  # More detailed responses
        }

        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            headers=headers,
            name=f"large_image_{image_size}b",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                try:
                    data = response.json()
                    if "choices" in data and len(data["choices"]) > 0:
                        response.success()
                    else:
                        response.failure("Invalid response format")
                except:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"HTTP {response.status_code}")