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

# Domains we treat as real Pakistan-local shopping / classifieds (for grounding URL fill).
_LOCAL_SHOP_URL_MARKERS = (
    "daraz.pk",
    "olx.com.pk",
    "facebook.com/marketplace",
    "interwood.pk",
    "habitt.com",
    "chenone.com",
    "packages.com.pk",
    "metropolitan.pk",
    "shophive.com",
    "ishopping.pk",
    "goto.com.pk",
    "homeshopping.pk",
    "vmart.pk",
)


def _get_serpapi_key() -> Optional[str]:
    return os.getenv("SERPAPI_KEY")


def _as_search_token(val) -> str:
    """Turn nested analysis values into a single search token; never a dict/list repr."""
    if val is None:
        return ""
    if isinstance(val, str):
        s = val.strip()
        if s.lower() in ("unknown", "null", "none", "n/a"):
            return ""
        return s
    if isinstance(val, bool):
        return str(val).lower()
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, list):
        return " ".join(_as_search_token(x) for x in val if x is not None)
    if isinstance(val, dict):
        return ""
    return str(val).strip()


def _brand_phrase(analysis: dict) -> str:
    b = analysis.get("brand_and_market_analysis")
    if isinstance(b, dict):
        return _as_search_token(b.get("brand_guess"))
    return _as_search_token(analysis.get("brand_guess"))


def _type_phrase(analysis: dict) -> str:
    t = analysis.get("type")
    if isinstance(t, dict):
        bits = []
        for key in ("main_category", "specific_type", "functional_use"):
            bits.append(_as_search_token(t.get(key)))
        rs = t.get("room_suitability")
        if isinstance(rs, list):
            bits.append(_as_search_token(rs))
        return " ".join(b for b in bits if b)
    return _as_search_token(t)


def _material_phrase(analysis: dict) -> str:
    m = analysis.get("materials")
    if isinstance(m, dict):
        bits = [
            _as_search_token(m.get("primary_material")),
            _as_search_token(m.get("upholstery_material")),
            _as_search_token(m.get("frame_material")),
            _as_search_token(m.get("surface_finish")),
        ]
        sec = m.get("secondary_materials")
        if isinstance(sec, list):
            bits.append(_as_search_token(sec))
        return " ".join(b for b in bits if b)
    return _as_search_token(analysis.get("material"))


def _color_phrase(analysis: dict) -> str:
    c = analysis.get("colors")
    if isinstance(c, dict):
        bits = []
        for key in ("dominant_colors", "accent_colors"):
            v = c.get(key)
            if isinstance(v, list):
                bits.append(_as_search_token(v))
            else:
                bits.append(_as_search_token(v))
        bits.append(_as_search_token(c.get("pattern_details")))
        return " ".join(b for b in bits if b)
    return _as_search_token(analysis.get("color"))


def _style_phrase(analysis: dict) -> str:
    d = analysis.get("design_analysis")
    if isinstance(d, dict):
        bits = []
        st = d.get("style")
        if isinstance(st, list):
            bits.append(_as_search_token(st))
        else:
            bits.append(_as_search_token(st))
        bits.append(_as_search_token(d.get("shape_language")))
        bits.append(_as_search_token(d.get("visual_weight")))
        return " ".join(b for b in bits if b)
    return _as_search_token(analysis.get("style"))


def _dimensions_phrase(analysis: dict) -> str:
    de = analysis.get("dimensions_estimate")
    if isinstance(de, dict):
        bits = [
            _as_search_token(de.get("size_category")),
            _as_search_token(de.get("estimated_seating_capacity")),
            _as_search_token(de.get("space_usage")),
        ]
        ed = de.get("estimated_dimensions_cm")
        if isinstance(ed, dict):
            for key in ("width", "depth", "height"):
                bits.append(_as_search_token(ed.get(key)))
        return " ".join(b for b in bits if b)
    return _as_search_token(analysis.get("approximate_dimensions"))


def _search_keywords_phrase(analysis: dict) -> str:
    kw = analysis.get("search_keywords")
    if isinstance(kw, list):
        return _as_search_token(kw)
    return _as_search_token(kw)


def _build_exact_query(analysis: dict, city: str) -> str:
    parts = []
    b = _brand_phrase(analysis)
    if b:
        parts.append(b)
    t = _type_phrase(analysis)
    if t:
        parts.append(t)
    m = _material_phrase(analysis)
    if m:
        parts.append(m)
    c = _color_phrase(analysis)
    if c:
        parts.append(c)
    s = _style_phrase(analysis)
    if s:
        parts.append(s)
    kw = _search_keywords_phrase(analysis)
    if kw:
        parts.append(kw)
    parts.append(f"buy in {city} {COUNTRY}")
    return " ".join(parts)


def _build_alternative_query(analysis: dict, city: str) -> str:
    parts = []
    d = _dimensions_phrase(analysis)
    if d:
        parts.append(d)
    t = _type_phrase(analysis)
    if t:
        parts.append(t)
    s = _style_phrase(analysis)
    if s:
        parts.append(s)
    kw = _search_keywords_phrase(analysis)
    if kw:
        parts.append(kw)
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


def _call_gemini_with_retry(client, model: str, contents: str, config=None):
    for attempt in range(MAX_RETRIES):
        try:
            kwargs = {"model": model, "contents": contents}
            if config is not None:
                kwargs["config"] = config
            return client.models.generate_content(**kwargs)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
            raise


def _grounding_uris_from_response(response) -> list[str]:
    """Collect unique https URLs from Gemini Google Search grounding metadata."""
    out: list[str] = []
    try:
        cand = response.candidates[0]
        gm = getattr(cand, "grounding_metadata", None)
        if gm is None:
            return out
        chunks = getattr(gm, "grounding_chunks", None) or []
        for ch in chunks:
            web = getattr(ch, "web", None)
            if web is None:
                continue
            uri = getattr(web, "uri", None)
            if uri and str(uri).startswith("http"):
                out.append(str(uri).strip())
    except (IndexError, AttributeError, TypeError):
        pass
    # Dedupe, preserve order
    seen = set()
    unique = []
    for u in out:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _is_local_shopping_url(url: str) -> bool:
    u = url.lower()
    return any(m in u for m in _LOCAL_SHOP_URL_MARKERS)


def _needs_real_link(link: str) -> bool:
    s = (link or "").strip()
    if not s.startswith("https://"):
        return True
    low = s.lower()
    if "example.com" in low or "localhost" in low or "placeholder" in low:
        return True
    return False


def _assign_links_from_grounding(listings: list[dict], uris: list[str]) -> list[dict]:
    local_pool = [u for u in uris if _is_local_shopping_url(u)]
    any_pool = [
        u
        for u in uris
        if u.startswith("https://") and "example.com" not in u.lower()
    ]
    used: set[str] = set()

    def _next_from(pool: list[str]) -> Optional[str]:
        for u in pool:
            if u not in used:
                used.add(u)
                return u
        return None

    for item in listings:
        if not isinstance(item, dict):
            continue
        if _needs_real_link(item.get("link", "")):
            chosen = _next_from(local_pool) or _next_from(any_pool)
            if chosen:
                item["link"] = chosen
    return listings


def search_with_gemini(query: str, city: str, mode: str = "exact") -> list[dict]:
    """Use Gemini with Google Search grounding to find real PK-local listing URLs."""
    try:
        from google import genai
        from google.genai import types

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return []

        client = genai.Client(api_key=api_key)

        search_hint = (
            f'First use Google Search for queries like: buy "{query}" {city} Pakistan PKR; '
            f'Daraz.pk "{query}"; OLX {city} furniture "{query}". '
            "Only use pages that ship to or are based in Pakistan."
        )

        if mode == "exact":
            prompt = f"""You are a furniture shopping researcher for {city}, {COUNTRY}.

User request (exact match): "{query}"

{search_hint}

LINK RULES (critical):
- You MUST use Google Search and base every listing on real pages you find.
- For EACH listing, "link" MUST be a full https URL copied exactly from a search result (same page the title/price/store refer to).
- Prefer well-known Pakistan marketplaces and brands: Daraz.pk, OLX.com.pk (item or search URL for {city}), Interwood.pk, Habitt.com, ChenOne.com, Packages/Metropolitan mall store pages, Facebook Marketplace listings for {city} when the URL is real.
- NEVER invent URLs, NEVER use example.com, NEVER use shortened opaque links you cannot verify, NEVER leave "link" empty.
- If you only find category or search result pages, use those real URLs — still full https from search results.
- Do not use Amazon US/UK/EU or other sites that do not serve {COUNTRY} for this item.

CONTENT RULES:
- Only results that deliver to or are located in {city} / {COUNTRY}; prices in {CURRENCY}.
- 5–8 listings; vary sources where possible.

Return a JSON array only:
[{{
  "title": "exact product name with details",
  "price": "price in PKR as shown on the page (e.g. Rs. 45,000)",
  "store": "seller or store name as on the page",
  "link": "https://... full URL from search results",
  "thumbnail": "",
  "source": "gemini"
}}]

Return ONLY the JSON array, no markdown fences or extra text."""
        else:
            prompt = f"""You are a furniture shopping researcher for {city}, {COUNTRY}.

User request (similar / alternative items): "{query}"

{search_hint}
Also search for comparable furniture (same type/size/style) from different sellers on Daraz, OLX, and major Pakistani furniture brands.

LINK RULES (critical):
- You MUST use Google Search and base every listing on real pages you find.
- For EACH listing, "link" MUST be a full https URL copied exactly from a search result.
- Prefer Daraz.pk, OLX.com.pk ({city}), Interwood.pk, Habitt.com, ChenOne.com, Packages Mall / local showroom pages with real URLs.
- NEVER invent URLs, NEVER use example.com, NEVER leave "link" empty. Use real category/search URLs from results if needed.
- No non-Pakistan shopping sites that do not ship to {COUNTRY}.

CONTENT RULES:
- Prices in {CURRENCY}; suggest budget, mid-range, and premium options where search results allow.
- 5–8 listings.

Return a JSON array only:
[{{
  "title": "product name with key attributes",
  "price": "price in PKR from the page",
  "store": "store or seller as on the page",
  "link": "https://... full URL from search results",
  "thumbnail": "",
  "source": "gemini"
}}]

Return ONLY the JSON array, no markdown fences or extra text."""

        grounding_tool = types.Tool(google_search=types.GoogleSearch())
        config = types.GenerateContentConfig(tools=[grounding_tool])

        response = _call_gemini_with_retry(
            client, "gemini-2.5-flash", prompt, config=config
        )
        text = response.text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
            text = text.strip()

        listings = json.loads(text)
        if not isinstance(listings, list):
            return []

        grounding_uris = _grounding_uris_from_response(response)
        listings = _assign_links_from_grounding(listings, grounding_uris)
        return listings
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
