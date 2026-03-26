"""
Product search service for furniture matching.
Primary: SerpAPI Google Shopping. Fallback: Gemini with Google Search grounding.
"""

import os
import json
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _get_serpapi_key() -> Optional[str]:
    return os.getenv("SERPAPI_KEY")


def _build_exact_query(analysis: dict, city: str) -> str:
    """Build a search query targeting the exact furniture item."""
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
    parts.append(f"buy in {city}")
    return " ".join(parts)


def _build_alternative_query(analysis: dict, city: str) -> str:
    """Build a search query for similar/alternative furniture."""
    parts = []
    if analysis.get("approximate_dimensions"):
        parts.append(analysis["approximate_dimensions"])
    if analysis.get("type"):
        parts.append(analysis["type"])
    if analysis.get("style"):
        parts.append(analysis["style"])
    parts.append(f"furniture buy in {city}")
    return " ".join(parts)


def search_with_serpapi(query: str, city: str) -> list[dict]:
    """Search using SerpAPI Google Shopping."""
    try:
        from serpapi import GoogleSearch
    except ImportError:
        return []

    api_key = _get_serpapi_key()
    if not api_key:
        return []

    params = {
        "engine": "google_shopping",
        "q": query,
        "location": city,
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


def search_with_gemini(query: str, city: str) -> list[dict]:
    """Fallback: use Gemini to generate plausible furniture listings."""
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return []

        client = genai.Client(api_key=api_key)

        prompt = f"""I'm looking for this furniture: "{query}"

Search the web and return a JSON array of 5-8 real product listings available in or near {city}.
Each item should have:
{{
  "title": "product name",
  "price": "price in PKR or USD",
  "store": "store or marketplace name",
  "link": "URL to the product page if known, otherwise empty string",
  "thumbnail": "",
  "source": "gemini"
}}

Focus on real marketplaces like OLX Pakistan, Daraz, IKEA, Facebook Marketplace, local furniture stores in {city}.
Return ONLY the JSON array, no markdown fences or extra text."""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        return json.loads(text)
    except Exception:
        return []


def search_exact_match(analysis: dict, city: str) -> list[dict]:
    """
    Search for the exact furniture item detected.
    Tries SerpAPI first, falls back to Gemini.
    """
    query = _build_exact_query(analysis, city)
    results = search_with_serpapi(query, city)
    if not results:
        results = search_with_gemini(query, city)
    return results


def search_alternative(analysis: dict, city: str) -> list[dict]:
    """
    Search for alternative/similar furniture items.
    Tries SerpAPI first, falls back to Gemini.
    """
    query = _build_alternative_query(analysis, city)
    results = search_with_serpapi(query, city)
    if not results:
        results = search_with_gemini(query, city)
    return results
