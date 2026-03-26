"""
Gemini Vision service for furniture analysis and report generation.
Uses the new google-genai SDK with Gemini 2.0 Flash.
"""

import os
import json
import base64
import time

from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

_client = None

MAX_RETRIES = 5
RETRY_BASE_DELAY = 5  # seconds

MODEL = "gemini-2.5-flash"


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get one free at https://aistudio.google.com"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def _call_with_retry(fn, *args, **kwargs):
    """Call a Gemini API function with exponential backoff on 429 errors."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "Too Many Requests" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    time.sleep(delay)
                    continue
            raise
    raise RuntimeError("Max retries exceeded for Gemini API call")


def analyze_furniture(crop_base64: str) -> dict:
    """
    Send a cropped furniture image to Gemini Vision for detailed analysis.

    Args:
        crop_base64: Base64-encoded JPEG image (without the data:image/jpeg;base64, prefix)

    Returns:
        Structured dict with furniture attributes.
    """
    client = _get_client()
    image_bytes = base64.b64decode(crop_base64)

    prompt = """Analyze this furniture image and return a JSON object with these fields:
{
  "type": "the furniture type, e.g. sofa, chair, table, bed, cabinet",
  "material": "primary material, e.g. wood, metal, fabric, leather, plastic",
  "color": "dominant color(s)",
  "style": "design style, e.g. modern, traditional, mid-century, industrial",
  "approximate_dimensions": "rough size description, e.g. 3-seater, large, compact",
  "condition_assessment": "good, fair, worn, damaged — with brief explanation",
  "brand_guess": "brand name if identifiable, otherwise null",
  "search_keywords": "comma-separated keywords ideal for searching this exact item online",
  "description": "one-sentence natural language description of the item"
}

Return ONLY valid JSON, no markdown fences or extra text."""

    response = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            prompt,
        ],
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "type": "unknown",
            "material": "unknown",
            "color": "unknown",
            "style": "unknown",
            "approximate_dimensions": "unknown",
            "condition_assessment": "unknown",
            "brand_guess": None,
            "search_keywords": "",
            "description": text[:200],
            "raw_response": text,
        }


def generate_report(items: list[dict]) -> str:
    """
    Generate a comprehensive furniture report from analyzed items and search results.

    Args:
        items: List of dicts, each containing:
            - detection: original YOLO detection data
            - analysis: Gemini furniture analysis
            - search_results: dict with 'exact' and/or 'alternative' listings
            - city: target city for search

    Returns:
        Markdown-formatted report string.
    """
    client = _get_client()
    items_summary = json.dumps(items, indent=2, default=str)

    prompt = f"""You are a professional interior designer writing a furniture assessment report.

Based on the following detected furniture items, their AI analysis, and marketplace search results,
generate a comprehensive, well-structured report in Markdown format.

The report should include:
1. **Executive Summary** — total items found, overall condition assessment
2. **Item-by-Item Analysis** — for each furniture piece:
   - Description and condition
   - Exact match results (if available) with prices and stores
   - Alternative suggestions (if available) with prices and stores
3. **Budget Summary** — estimated total cost for replacements/purchases
4. **Recommendations** — which items to keep, repair, or replace

Data:
{items_summary}

Write a clear, professional report. Use tables where appropriate for pricing comparisons."""

    response = _call_with_retry(
        client.models.generate_content,
        model=MODEL,
        contents=prompt,
    )
    return response.text
