import argparse
import base64
import json
from pathlib import Path

import requests


DEFAULT_PROMPT = (
    "Summarize this image clearly. Mention visible objects, text, workflow steps, "
    "systems, labels, and any important relationships. Do not invent details."
)


def summarize_image(
    image_path: Path,
    *,
    model: str,
    ollama_url: str,
    prompt: str,
    timeout: int,
) -> str:
    image_data = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    response = requests.post(
        f"{ollama_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_data],
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 700,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()

    payload = response.json()
    return str(payload.get("message", {}).get("content", "")).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test one local image with Ollama vision model."
    )
    parser.add_argument("image", help="Path to the image file to summarize.")
    parser.add_argument("--model", default="qwen2.5vl", help="Ollama model name.")
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Ollama base URL.",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send.")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw structured result with model and image metadata.",
    )
    args = parser.parse_args()

    image_path = Path(args.image).expanduser().resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    summary = summarize_image(
        image_path,
        model=args.model,
        ollama_url=args.ollama_url,
        prompt=args.prompt,
        timeout=args.timeout,
    )

    if args.json:
        print(
            json.dumps(
                {
                    "image": str(image_path),
                    "model": args.model,
                    "ollama_url": args.ollama_url,
                    "summary": summary,
                },
                indent=2,
            )
        )
        return

    print(summary)


if __name__ == "__main__":
    main()
