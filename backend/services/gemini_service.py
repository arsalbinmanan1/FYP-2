"""
Gemini Vision service for furniture analysis and report generation.
Uses the new google-genai SDK with Gemini 2.0 Flash.
"""

import os
import json
import base64
import time
from textwrap import dedent

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

    prompt = dedent(
        """
        You are an expert furniture authenticator, interior designer, materials analyst, and second-hand marketplace evaluator.

        Your task is to analyze the uploaded furniture image with extreme precision and extract every visually inferable detail possible. Focus on structure, craftsmanship, materials, wear, proportions, aesthetics, functionality, and marketplace relevance.

        Return a JSON object with the following schema:

        {
          "type": {
            "main_category": "primary furniture category",
            "specific_type": "more precise type classification",
            "functional_use": "intended practical use",
            "room_suitability": ["living room", "office", "bedroom"]
          },

          "materials": {
            "primary_material": "main visible material",
            "secondary_materials": ["list", "of", "additional", "materials"],
            "upholstery_material": "fabric/leather/velvet/etc or null",
            "frame_material": "wood/metal/plastic/etc or null",
            "surface_finish": "matte, glossy, lacquered, distressed, brushed, polished, etc",
            "material_confidence": "high/medium/low"
          },

          "colors": {
            "dominant_colors": ["main visible colors"],
            "accent_colors": ["secondary/accent colors"],
            "color_temperature": "warm/cool/neutral",
            "pattern_details": "solid, striped, textured, floral, geometric, etc"
          },

          "design_analysis": {
            "style": ["modern", "mid-century", "industrial", "scandinavian", etc],
            "design_influences": ["possible design inspirations or eras"],
            "shape_language": "boxy, curved, minimalist, angular, organic, etc",
            "visual_weight": "light, balanced, bulky, heavy",
            "luxury_level": "budget, mid-range, premium, luxury"
          },

          "construction_details": {
            "leg_style": "tapered, hidden, metal frame, caster wheels, etc",
            "armrest_style": "rolled, track, armless, curved, etc",
            "backrest_style": "tufted, straight, reclined, paneled, etc",
            "cushion_structure": "fixed, removable, segmented, plush, firm",
            "joinery_or_build_quality_notes": "visible craftsmanship observations"
          },

          "dimensions_estimate": {
            "size_category": "compact, medium, oversized",
            "estimated_seating_capacity": "1-seater, 2-seater, 3-seater, etc",
            "estimated_dimensions_cm": {
              "width": "approximate width range",
              "depth": "approximate depth range",
              "height": "approximate height range"
            },
            "space_usage": "small apartment friendly, office scale, large-room furniture, etc"
          },

          "condition_assessment": {
            "overall_condition": "excellent, good, fair, worn, damaged",
            "wear_level": "minimal, moderate, heavy",
            "visible_damage": ["scratches", "fabric pilling", "stains", "dents", etc],
            "structural_integrity_guess": "stable, possibly loose, unknown",
            "restoration_needed": true,
            "cleanliness_assessment": "clean, dusty, stained, aged, etc",
            "condition_summary": "short expert-level explanation"
          },

          "ergonomics_and_comfort": {
            "comfort_level_guess": "soft, firm, ergonomic, lounge-oriented, etc",
            "posture_support": "upright, relaxed, reclining, neutral",
            "intended_usage_duration": "short-term sitting, extended lounging, work seating, etc"
          },

          "brand_and_market_analysis": {
            "brand_guess": "possible manufacturer or null",
            "brand_confidence": "high/medium/low",
            "estimated_price_range_usd": {
              "new": "estimated retail range",
              "used": "estimated resale range"
            },
            "marketplace_category": "ikea-style, designer furniture, vintage collectible, mass-market, handcrafted, etc"
          },

          "photo_analysis": {
            "image_quality": "high/medium/low",
            "visible_angle": "front, side, angled, top-down, etc",
            "occlusion_notes": "parts hidden or unclear",
            "background_context": "environment clues from surroundings"
          },

          "search_keywords": [
            "highly specific search phrases optimized for marketplace and reverse image search"
          ],

          "tags": [
            "short marketplace tags"
          ],

          "description": "A detailed one-to-two sentence natural language description written like a premium marketplace listing.",

          "confidence_score": {
            "overall_analysis_confidence": 0-100
          }
        }

        Important instructions:
        - Be extremely observant and infer as many realistic details as possible from visual evidence.
        - Do NOT leave fields empty unless absolutely impossible to infer.
        - If uncertain, provide best-effort educated guesses with lower confidence wording.
        - Use marketplace-friendly terminology suitable for eBay, Facebook Marketplace, Chairish, IKEA search, Wayfair search, and Google Lens optimization.
        - Capture tiny details like stitching, tufting, grain texture, edge profile, leg geometry, hardware style, and visible wear patterns.
        - Infer probable manufacturing quality and price tier from appearance.
        - Search keywords should be highly descriptive and long-tail optimized.
        - Return ONLY valid JSON.
        - Do not use markdown fences.
        - Do not include explanations outside the JSON.
        """
    ).strip()

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
