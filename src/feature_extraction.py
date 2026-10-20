"""
AI-Based Phishing Website Detection
Phase 6B: Live URL Feature Extraction Module

Extracts the exact 54 numerical features expected by the trained Random Forest model
(models/random_forest_tuned.joblib) in the exact order specified in models/feature_names.json.

Key Design Principles:
- 100% Non-crashing: safe fallback for network timeouts or parsing anomalies.
- Deterministic feature ordering enforced via models/feature_names.json.
- Clear, well-documented constants for heuristics (phish_hints, brands, shorteners, TLDs).
- Validated numerical output with NaN/Inf prevention.
"""

import ipaddress
import json
import math
import re
from itertools import groupby
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import tldextract


# ==============================================================================
# CONFIGURATION & CONSTANTS FOR HEURISTIC FEATURES
# (Documented and easily inspectable for viva examination)
# ==============================================================================

# 1. Phishing Hint Keywords (Common credential-harvesting tokens in URLs)
PHISH_HINTS_KEYWORDS: List[str] = [
    "login",
    "signin",
    "sign-in",
    "account",
    "banking",
    "secure",
    "webscr",
    "update",
    "verify",
    "verification",
    "confirm",
    "password",
    "credential",
    "admin",
    "auth",
    "ebayisapi",
    "wallet",
    "security",
    "authenticate",
    "billing",
    "support",
    "recovery",
]

# 2. Recognized Brand Names (Institutions frequently targeted for impersonation)
RECOGNIZED_BRANDS: List[str] = [
    "google",
    "apple",
    "microsoft",
    "amazon",
    "paypal",
    "facebook",
    "netflix",
    "dropbox",
    "yahoo",
    "alibaba",
    "ebay",
    "chase",
    "wellsfargo",
    "citi",
    "bankofamerica",
    "barclays",
    "hsbc",
    "instagram",
    "twitter",
    "linkedin",
    "whatsapp",
    "telegram",
    "adobe",
    "steam",
    "visa",
    "mastercard",
]

# 3. Known URL Shortening and Free Hosting Redirect Services
SHORTENING_SERVICES: set = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "mcaf.ee",
    "rebrand.ly",
    "cutt.ly",
    "appspot.com",
    "blogspot.com",
    "hubspot.com",
    "sharepoint.com",
    "wordpress.com",
}

# 4. Suspicious / High-Abuse Top-Level Domains (Identified in benchmark & Spamhaus abuse feeds)
SUSPICIOUS_TLDS: set = {
    "ae",
    "asia",
    "au",
    "audio",
    "bj",
    "club",
    "exposed",
    "ga",
    "icu",
    "id",
    "la",
    "link",
    "media",
    "ml",
    "mx",
    "online",
    "party",
    "pe",
    "review",
    "rs",
    "tk",
    "top",
    "website",
    "work",
    "xyz",
    "gq",
    "cf",
    "fit",
    "buzz",
}

# Project root anchor for reliable relative path resolution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cache for models/feature_names.json
_FEATURE_NAMES_CACHE: Optional[List[str]] = None


def get_expected_feature_names(
    feature_names_path: str = "models/feature_names.json",
) -> List[str]:
    """Load and cache the canonical 54 feature names in their required order."""
    global _FEATURE_NAMES_CACHE
    if _FEATURE_NAMES_CACHE is not None:
        return _FEATURE_NAMES_CACHE

    p = Path(feature_names_path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p

    if not p.exists():
        raise FileNotFoundError(f"Feature names file missing at: {p}")

    with open(p, "r", encoding="utf-8") as f:
        _FEATURE_NAMES_CACHE = json.load(f)

    if len(_FEATURE_NAMES_CACHE) != 54:
        raise ValueError(f"Expected 54 feature names, but found {len(_FEATURE_NAMES_CACHE)}")

    return _FEATURE_NAMES_CACHE


# ==============================================================================
# URL NORMALIZATION & HELPER FUNCTIONS
# ==============================================================================

def normalize_url(url: str) -> str:
    """Safely sanitize, trim, and ensure URL has a valid scheme."""
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")

    cleaned = url.strip()
    if not cleaned:
        raise ValueError("URL cannot be empty.")

    # Prepend http:// if scheme is missing so urllib and tldextract parse correctly
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", cleaned):
        cleaned = "http://" + cleaned

    return cleaned


def is_ip_address(host: str) -> bool:
    """Check if the hostname is a valid IPv4 or IPv6 address."""
    clean_host = host.split(":")[0]  # strip port if present
    try:
        ipaddress.ip_address(clean_host)
        return True
    except ValueError:
        return False


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not text:
        return 0.0
    prob = [text.count(c) / len(text) for c in set(text)]
    return -sum(p * math.log2(p) for p in prob)


def get_network_redirections(url: str, timeout: float = 1.5) -> Tuple[int, int]:
    """
    Measure live HTTP redirects safely:
    - nb_redirection: total redirect hops
    - nb_external_redirection: redirect hops that cross to a different domain

    Safe Fallback:
    If network is down, site is offline, or request times out after 1.5s,
    safely returns (0, 0) to ensure zero crashes and zero UI stalls.
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/118.0.0.0 Safari/537.36"
            )
        }
        resp = requests.head(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
        history = resp.history
        nb_redirection = len(history)

        if nb_redirection == 0:
            return 0, 0

        initial_host = urlparse(url).netloc.lower()
        nb_ext_redirection = sum(
            1 for hop in history if urlparse(hop.url).netloc.lower() != initial_host
        )
        return nb_redirection, nb_ext_redirection
    except Exception:
        # Graceful fallback: Network request failure must never break the pipeline
        return 0, 0


# ==============================================================================
# CORE FEATURE EXTRACTION ENGINE
# ==============================================================================

def extract_features_dict(
    url: str,
    check_network: bool = True,
    network_timeout: float = 1.5,
) -> Dict[str, Union[int, float]]:
    """
    Extract all 54 numerical features for a single URL string.
    Returns a dictionary of feature_name -> numerical_value.
    """
    clean_url = normalize_url(url)
    parsed = urlparse(clean_url)
    ext = tldextract.extract(clean_url)

    host = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query
    tld = ext.suffix.lower()
    subdomain = ext.subdomain.lower()
    domain = ext.domain.lower()

    # 1-3. Lengths & IP
    length_url = len(clean_url)
    length_hostname = len(host)
    ip_flag = 1 if is_ip_address(host) else 0

    # 4-20. Special character counts
    nb_dots = clean_url.count(".")
    nb_hyphens = clean_url.count("-")
    nb_at = clean_url.count("@")
    nb_qm = clean_url.count("?")
    nb_and = clean_url.count("&")
    nb_eq = clean_url.count("=")
    nb_underscore = clean_url.count("_")
    nb_tilde = clean_url.count("~")
    nb_percent = clean_url.count("%")
    nb_slash = clean_url.count("/")
    nb_star = clean_url.count("*")
    nb_colon = clean_url.count(":")
    nb_comma = clean_url.count(",")
    nb_semicolumn = clean_url.count(";")
    nb_dollar = clean_url.count("$")
    nb_space = clean_url.count(" ") + clean_url.count("%20")
    nb_www = clean_url.lower().count("www")

    # 21. nb_com: 'com' substrings outside primary TLD
    total_com = clean_url.lower().count("com")
    tld_is_com = 1 if (host.endswith(".com") or host.endswith(".com/")) else 0
    nb_com = max(0, total_com - tld_is_com)

    # 22. nb_dslash: double slashes after scheme
    nb_dslash = clean_url.count("//") - (1 if "://" in clean_url else 0)
    nb_dslash = max(0, nb_dslash)

    # 23-24. Protocol tokens in path / host
    http_in_path = 1 if "http" in path.lower() else 0
    https_token = 1 if ("https" in host or "https" in path.lower()) else 0

    # 25-26. Digit ratios
    digits_url = sum(c.isdigit() for c in clean_url)
    ratio_digits_url = round(digits_url / max(1, length_url), 6)
    digits_host = sum(c.isdigit() for c in host)
    ratio_digits_host = round(digits_host / max(1, length_hostname), 6)

    # 27-28. Punycode & Port
    punycode = 1 if "xn--" in host else 0
    port = 1 if (parsed.port and parsed.port not in (80, 443)) else 0

    # 29-30. TLD positions
    tld_in_path = 1 if (tld and tld in path.lower()) else 0
    tld_in_subdomain = 1 if (tld and tld in subdomain) else 0

    # 31. abnormal_subdomain: IP host, hex hashes, or UUIDs in subdomain
    has_hex_hash = bool(re.search(r"[a-f0-9]{20,}", subdomain))
    has_uuid = bool(
        re.search(
            r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}",
            subdomain,
        )
    )
    abnormal_subdomain = 1 if (ip_flag or has_hex_hash or has_uuid) else 0

    # 32. nb_subdomains
    nb_subdomains = len(subdomain.split(".")) if subdomain else 0

    # 33. prefix_suffix: hyphen in registered domain
    prefix_suffix = 1 if "-" in domain else 0

    # 34. random_domain: High character entropy or abnormal consonant clusters
    domain_entropy = calculate_entropy(domain)
    consonant_cluster = bool(re.search(r"[bcdfghjklmnpqrstvwxyz]{5,}", domain))
    random_domain = 1 if (domain_entropy > 3.8 or consonant_cluster) else 0

    # 35. shortening_service
    registered_domain = f"{domain}.{tld}" if tld else domain
    shortening_service = (
        1 if (registered_domain in SHORTENING_SERVICES or host in SHORTENING_SERVICES) else 0
    )

    # 36. path_extension: dynamic scripts or executable extensions in path
    path_extension = (
        1
        if any(
            path.lower().endswith(ext_str)
            for ext_str in [".php", ".exe", ".cgi", ".bin", ".sh", ".asp", ".aspx", ".jsp"]
        )
        else 0
    )

    # 37-38. Network Redirections (with graceful fallback to 0)
    if check_network:
        nb_redirection, nb_external_redirection = get_network_redirections(
            clean_url, timeout=network_timeout
        )
    else:
        nb_redirection, nb_external_redirection = 0, 0

    # 39-49. Token / Word Statistics (Stripping scheme and 'www' prefix)
    clean_host = host[4:] if host.startswith("www.") else host
    host_tokens = [w for w in re.split(r"[^a-zA-Z0-9]+", clean_host) if w]
    path_tokens = [w for w in re.split(r"[^a-zA-Z0-9]+", f"{path} {query}") if w]
    all_tokens = host_tokens + path_tokens

    length_words_raw = len(all_tokens)

    # char_repeat: maximum run length of any consecutive repeated character in URL
    char_repeat = max((len(list(g)) for _, g in groupby(clean_url)), default=0)

    # Shortest words
    shortest_words_raw = min((len(t) for t in all_tokens), default=0)
    shortest_word_host = min((len(t) for t in host_tokens), default=0)
    shortest_word_path = min((len(t) for t in path_tokens), default=0)

    # Longest words
    longest_words_raw = max((len(t) for t in all_tokens), default=0)
    longest_word_host = max((len(t) for t in host_tokens), default=0)
    longest_word_path = max((len(t) for t in path_tokens), default=0)

    # Average word lengths
    avg_words_raw = (
        round(sum(len(t) for t in all_tokens) / len(all_tokens), 6) if all_tokens else 0.0
    )
    avg_word_host = (
        round(sum(len(t) for t in host_tokens) / len(host_tokens), 6) if host_tokens else 0.0
    )
    avg_word_path = (
        round(sum(len(t) for t in path_tokens) / len(path_tokens), 6) if path_tokens else 0.0
    )

    # 50. phish_hints: count of hint keywords in path and query
    url_lower = clean_url.lower()
    path_query_lower = f"{path} {query}".lower()
    phish_hints = sum(1 for kw in PHISH_HINTS_KEYWORDS if kw in path_query_lower)

    # 51-53. Brand names in domain, subdomain, path
    domain_in_brand = 1 if any(b in domain for b in RECOGNIZED_BRANDS) else 0
    brand_in_subdomain = 1 if any(b in subdomain for b in RECOGNIZED_BRANDS) else 0
    brand_in_path = 1 if any(b in path_query_lower for b in RECOGNIZED_BRANDS) else 0

    # 54. suspecious_tld
    suspecious_tld = 1 if tld in SUSPICIOUS_TLDS else 0

    # Assemble raw feature dictionary
    raw_dict = {
        "length_url": length_url,
        "length_hostname": length_hostname,
        "ip": ip_flag,
        "nb_dots": nb_dots,
        "nb_hyphens": nb_hyphens,
        "nb_at": nb_at,
        "nb_qm": nb_qm,
        "nb_and": nb_and,
        "nb_eq": nb_eq,
        "nb_underscore": nb_underscore,
        "nb_tilde": nb_tilde,
        "nb_percent": nb_percent,
        "nb_slash": nb_slash,
        "nb_star": nb_star,
        "nb_colon": nb_colon,
        "nb_comma": nb_comma,
        "nb_semicolumn": nb_semicolumn,
        "nb_dollar": nb_dollar,
        "nb_space": nb_space,
        "nb_www": nb_www,
        "nb_com": nb_com,
        "nb_dslash": nb_dslash,
        "http_in_path": http_in_path,
        "https_token": https_token,
        "ratio_digits_url": ratio_digits_url,
        "ratio_digits_host": ratio_digits_host,
        "punycode": punycode,
        "port": port,
        "tld_in_path": tld_in_path,
        "tld_in_subdomain": tld_in_subdomain,
        "abnormal_subdomain": abnormal_subdomain,
        "nb_subdomains": nb_subdomains,
        "prefix_suffix": prefix_suffix,
        "random_domain": random_domain,
        "shortening_service": shortening_service,
        "path_extension": path_extension,
        "nb_redirection": nb_redirection,
        "nb_external_redirection": nb_external_redirection,
        "length_words_raw": length_words_raw,
        "char_repeat": char_repeat,
        "shortest_words_raw": shortest_words_raw,
        "shortest_word_host": shortest_word_host,
        "shortest_word_path": shortest_word_path,
        "longest_words_raw": longest_words_raw,
        "longest_word_host": longest_word_host,
        "longest_word_path": longest_word_path,
        "avg_words_raw": avg_words_raw,
        "avg_word_host": avg_word_host,
        "avg_word_path": avg_word_path,
        "phish_hints": phish_hints,
        "domain_in_brand": domain_in_brand,
        "brand_in_subdomain": brand_in_subdomain,
        "brand_in_path": brand_in_path,
        "suspecious_tld": suspecious_tld,
    }

    return raw_dict


def extract_features(
    url: str,
    check_network: bool = True,
    network_timeout: float = 1.5,
    return_type: str = "dataframe",
) -> Union[pd.DataFrame, pd.Series, np.ndarray, Dict[str, Any]]:
    """
    Primary interface for URL feature extraction.
    
    Guarantees:
    - Exact 54 feature names in the exact order of models/feature_names.json
    - Zero NaNs, zero Infs, all float/int numbers
    - Compatible directly with models/random_forest_tuned.joblib (unscaled)
    
    Parameters:
        url: Raw input URL string
        check_network: Whether to test live HTTP redirections (default: True)
        network_timeout: Timeout in seconds for HTTP requests (default: 1.5s)
        return_type: 'dataframe' (default), 'series', 'array', or 'dict'
    """
    raw_dict = extract_features_dict(
        url,
        check_network=check_network,
        network_timeout=network_timeout,
    )

    expected_names = get_expected_feature_names("models/feature_names.json")

    # Strictly enforce exact feature order and completeness
    ordered_dict = {}
    for name in expected_names:
        if name not in raw_dict:
            raise KeyError(f"Feature '{name}' missing from extracted dictionary.")
        val = raw_dict[name]
        # Validate numerical integrity
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise TypeError(f"Feature '{name}' has non-numerical value: {val}")
        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"Feature '{name}' contains NaN or Infinite value.")
        ordered_dict[name] = val

    if return_type == "dataframe":
        return pd.DataFrame([ordered_dict], columns=expected_names)
    elif return_type == "series":
        return pd.Series(ordered_dict, index=expected_names)
    elif return_type == "array":
        return np.array([ordered_dict[name] for name in expected_names], dtype=np.float64)
    elif return_type == "dict":
        return ordered_dict
    else:
        raise ValueError(f"Unknown return_type: {return_type}")
