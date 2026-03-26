"""
Product search service for furniture matching.
Primary: SerpAPI Google Shopping. Fallback: Gemini with strict location-based search.
"""

import os
import json
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

COUNTRY = "Pakistan"
CURRENCY = "PKR"

MAX_RETRIES = 3
RETRY_BASE_DELAY = 5


def _get_serpapi_key() -> Optional[str]:
    return os.getenv("SERPAPI_KEY")


def _build_exact_query(analysis: dict, city: str) -> str:
    parts = []
    if analysis.get("brand_guess"):
        parts.append(analysis["brand_guess"])
    if analysis.get("type"):
        parts.append(analysis["type"])
    if analysis.get("material"):
        parts.append(analysis["material"])
    if analysis.get("color"):
        parts.append(analysis["color"])
    if analysis.get("style"):
        parts.append(analysis["style"])
    parts.append(f"buy in {city} {COUNTRY}")
    return " ".join(parts)


def _build_alternative_query(analysis: dict, city: str) -> str:
    parts = []
    if analysis.get("approximate_dimensions"):
        parts.append(analysis["approximate_dimensions"])
    if analysis.get("type"):
        parts.append(analysis["type"])
    if analysis.get("style"):
        parts.append(analysis["style"])
    parts.append(f"furniture in {city} {COUNTRY}")
    return " ".join(parts)


def search_with_serpapi(query: str, city: str) -> list[dict]:
    try:
        from serpapi import GoogleSearch
    except ImportError:
        return []

    api_key = _get_serpapi_key()
    if not api_key or api_key == "your_serpapi_key_here":
        return []

    params = {
        "engine": "google_shopping",
        "q": query,
        "location": f"{city}, {COUNTRY}",
        "gl": "pk",
        "hl": "en",
        "api_key": api_key,
        "num": 8,
    }

    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        shopping_results = results.get("shopping_results", [])

        listings = []
        for item in shopping_results[:8]:
            listings.append({
                "title": item.get("title", ""),
                "price": item.get("extracted_price") or item.get("price", "N/A"),
                "store": item.get("source", ""),
                "link": item.get("link", ""),
                "thumbnail": item.get("thumbnail", ""),
                "rating": item.get("rating"),
                "reviews": item.get("reviews"),
                "source": "serpapi",
            })
        return listings
    except Exception:
        return []


def _call_gemini_with_retry(client, model: str, contents: str):
    for attempt in range(MAX_RETRIES):
        try:
            return client.models.generate_content(model=model, contents=contents)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            raise


def search_with_gemini(query: str, city: str, mode: str = "exact") -> list[dict]:
    """Use Gemini to find furniture listings strictly within the specified city."""
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return []

        client = genai.Client(api_key=api_key)

        if mode == "exact":
            prompt = f"""You are a local furniture shopping assistant for {city}, {COUNTRY}.

A user wants to find this EXACT furniture item: "{query}"

STRICT RULES:
- ONLY return results from stores, shops, and marketplaces located IN or delivering TO {city}, {COUNTRY}
- Prices MUST be in {CURRENCY} (Pakistani Rupees)
- Focus on these local sources: OLX {city}, Daraz.pk, Facebook Marketplace {city}, local furniture markets in {city} (e.g. Furniture Market, Timber Market), PakWheels (for home items)
- If the item is from a known brand, include the brand's local dealer in {city} if one exists
- DO NOT include international stores that don't ship to {COUNTRY}
- DO NOT include results from other countries

Return a JSON array of 5-8 listings:
[{{
  "title": "exact product name with details",
  "price": "price in PKR (e.g. Rs. 45,000)",
  "store": "store name and location in {city}",
  "link": "URL if known, otherwise empty string",
  "thumbnail": "",
  "source": "gemini"
}}]

Return ONLY the JSON array, no markdown fences or extra text."""
        else:
            prompt = f"""You are a local furniture shopping assistant for {city}, {COUNTRY}.

A user wants to find ALTERNATIVE furniture similar to: "{query}"

Find similar items with comparable features (size, type, style) but from different brands or stores.

STRICT RULES:
- ONLY return results from stores, shops, and marketplaces located IN or delivering TO {city}, {COUNTRY}
- Prices MUST be in {CURRENCY} (Pakistani Rupees)
- Focus on these local sources: OLX {city}, Daraz.pk, Facebook Marketplace {city}, local furniture markets in {city}, Interwood, Habitt, ChenOne, Packages Mall stores
- DO NOT include international stores that don't ship to {COUNTRY}
- DO NOT include results from other countries
- Suggest alternatives at various price points (budget, mid-range, premium)

Return a JSON array of 5-8 listings:
[{{
  "title": "product name with key attributes (e.g. '7 Seater L-Shape Sofa - Velvet Fabric')",
  "price": "price in PKR (e.g. Rs. 85,000)",
  "store": "store name and area in {city} (e.g. 'Habitt - Gulberg, {city}')",
  "link": "URL if known, otherwise empty string",
  "thumbnail": "",
  "source": "gemini"
}}]

Return ONLY the JSON array, no markdown fences or extra text."""

        response = _call_gemini_with_retry(client, "gemini-2.5-flash", prompt)
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        return json.loads(text)
    except Exception:
        return []


def search_exact_match(analysis: dict, city: str) -> list[dict]:
    query = _build_exact_query(analysis, city)
    results = search_with_serpapi(query, city)
    if not results:
        results = search_with_gemini(query, city, mode="exact")
    return results


def search_alternative(analysis: dict, city: str) -> list[dict]:
    query = _build_alternative_query(analysis, city)
    results = search_with_serpapi(query, city)
    if not results:
        results = search_with_gemini(query, city, mode="alternative")
    return results
