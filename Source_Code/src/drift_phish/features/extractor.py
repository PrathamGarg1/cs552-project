"""Lexical URL feature extraction aligned with Krishnan & Naik (2025) Table 3.

URL-only features — no HTML / third-party APIs.

``extract_paper_features`` is the full set used by the experiments;
``extract_url_features`` (33 features) is the subset used by the optional web demo.
"""

from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np

# Suspicious terms from paper Section 3.3
_SUSPICIOUS_TERMS = [
    "login",
    "signin",
    "verify",
    "account",
    "secure",
    "payment",
    "billing",
    "free",
    "prize",
    "password",
    "support",
    "authenticate",
    "service",
    "reset",
    "unlock",
    "transaction",
    "claim",
    "winning",
    "alert",
    "suspend",
    "reactivate",
    "update",
    "confirm",
    "wallet",
    "credential",
]

_SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "bit.do",
    "cutt.ly",
}

_SUSPICIOUS_TLDS = {
    "tk",
    "ml",
    "ga",
    "cf",
    "gq",
    "xyz",
    "top",
    "club",
    "work",
    "click",
    "link",
    "info",
    "zip",
    "mov",
}

_SUSPICIOUS_EXTS = {
    ".exe",
    ".zip",
    ".rar",
    ".scr",
    ".js",
    ".jar",
    ".apk",
    ".bat",
    ".cmd",
    ".vbs",
}

# Brands commonly spoofed in phishing URLs (paper: brand-name proximity).
_BRANDS = [
    "paypal",
    "apple",
    "microsoft",
    "google",
    "amazon",
    "facebook",
    "instagram",
    "netflix",
    "whatsapp",
    "bankofamerica",
    "wellsfargo",
    "chase",
    "citibank",
    "icloud",
    "outlook",
    "office365",
    "linkedin",
    "dropbox",
    "adobe",
]

URL_FEATURE_NAMES = [
    "url_length",
    "domain_length",
    "path_length",
    "token_count",
    "num_subdirectories",
    "url_shortening",
    "special_char_count",
    "number_count",
    "suspicious_tld",
    "has_port",
    "entropy",
    "is_ip",
    "num_dots",
    "num_slashes",
    "num_hyphens",
    "num_at",
    "num_equals",
    "num_qmark",
    "num_ampersand",
    "num_percent",
    "num_underscore",
    "non_alphanumeric_chars",
    "homoglyph_detection",
    "url_encoded",
    "javascript_data_uri",
    "suspicious_file_extension",
    "suspicious_term_frequency",
    "brand_name_proximity",
    "multiple_redirects",
    "redirect_chain_complexity",
    "has_https",
    "query_length",
    "num_params",
]

_IP = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
# Common Latin/Cyrillic lookalikes + mixed scripts
_HOMOGLYPH = re.compile(
    r"[\u0400-\u04FF\u0430\u0435\u043E\u0440\u0441\u0443\u0445\u0456]"
)


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts: dict[str, int] = {}
    for ch in text:
        counts[ch] = counts.get(ch, 0) + 1
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_url_features(url: str) -> dict[str, float]:
    """Extract paper-style lexical features from a raw URL string."""
    raw = (url or "").strip()
    if not raw:
        return {name: 0.0 for name in URL_FEATURE_NAMES}

    lowered = raw.lower()
    try:
        parsed = urlparse(raw if "://" in raw else f"http://{raw}")
    except Exception:
        return {name: 0.0 for name in URL_FEATURE_NAMES}

    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    labels = [p for p in host.split(".") if p]
    tld = labels[-1] if labels else ""
    tokens = re.split(r"[/\-_.=?&]+", lowered)
    tokens = [t for t in tokens if t]

    specials = sum(ch in "@-_=%&?./+#" for ch in raw)
    digits = sum(ch.isdigit() for ch in raw)
    non_alnum = sum(not ch.isalnum() for ch in raw)
    term_hits = sum(1 for term in _SUSPICIOUS_TERMS if term in lowered)
    encoded = lowered.count("%")
    js_data = 1.0 if ("javascript:" in lowered or "data:" in lowered) else 0.0
    has_short = 1.0 if any(s in host for s in _SHORTENERS) else 0.0
    sus_tld = 1.0 if tld in _SUSPICIOUS_TLDS else 0.0
    sus_ext = 1.0 if any(path.lower().endswith(ext) for ext in _SUSPICIOUS_EXTS) else 0.0
    homo = 1.0 if _HOMOGLYPH.search(raw) else 0.0
    params = len([p for p in query.split("&") if p]) if query else 0
    brand_hits = sum(1 for b in _BRANDS if b in lowered)
    # Lexical proxies for redirect complexity (no live HTTP fetch).
    redirect_markers = sum(
        lowered.count(tok)
        for tok in ("redirect", "redir", "url=", "returnurl", "continue=", "next=")
    )
    multi_redir = 1.0 if redirect_markers >= 2 or lowered.count("http") >= 2 else 0.0
    redir_score = float(redirect_markers + (2 if multi_redir else 0) + (1 if "@" in raw else 0))

    return {
        "url_length": float(len(raw)),
        "domain_length": float(len(host)),
        "path_length": float(len(path)),
        "token_count": float(len(tokens)),
        "num_subdirectories": float(path.count("/")),
        "url_shortening": has_short,
        "special_char_count": float(specials),
        "number_count": float(digits),
        "suspicious_tld": sus_tld,
        "has_port": 1.0 if parsed.port else 0.0,
        "entropy": _entropy(lowered),
        "is_ip": 1.0 if _IP.match(host) else 0.0,
        "num_dots": float(raw.count(".")),
        "num_slashes": float(raw.count("/")),
        "num_hyphens": float(raw.count("-")),
        "num_at": float(raw.count("@")),
        "num_equals": float(raw.count("=")),
        "num_qmark": float(raw.count("?")),
        "num_ampersand": float(raw.count("&")),
        "num_percent": float(encoded),
        "num_underscore": float(raw.count("_")),
        "non_alphanumeric_chars": float(non_alnum),
        "homoglyph_detection": homo,
        "url_encoded": 1.0 if encoded >= 2 else 0.0,
        "javascript_data_uri": js_data,
        "suspicious_file_extension": sus_ext,
        "suspicious_term_frequency": float(term_hits),
        "brand_name_proximity": float(brand_hits),
        "multiple_redirects": multi_redir,
        "redirect_chain_complexity": redir_score,
        "has_https": 1.0 if parsed.scheme == "https" else 0.0,
        "query_length": float(len(query)),
        "num_params": float(params),
    }


# ---------------------------------------------------------------------------
# Extended paper feature set (Table 3 + the feature names that appear in the
# paper's Table 8 / Figs. 4-7). Superset of ``URL_FEATURE_NAMES``.
# ---------------------------------------------------------------------------
_EXTRA_FEATURE_NAMES = [
    "scheme_http",
    "num_digits_path",
    "num_digits_query",
    "num_hyphens_host",
    "num_delimiters_path",
    "longest_path_token_length",
    "num_tlds",
    "num_subdomains",
    "tld_length",
    "longest_host_label_length",
    "digit_ratio",
    "host_entropy",
    "has_login",
    "has_verify",
    "has_billing",
    "has_free",
    "has_secure",
    "has_admin",
    "has_account",
    "has_update",
    "has_password",
    "javascript_obfuscation",
    "iframe_usage",
    "domain_mismatch",
    "redirects_chain_length",
]
PAPER_FEATURE_NAMES = list(URL_FEATURE_NAMES) + _EXTRA_FEATURE_NAMES

_TLD_TOKEN = re.compile(
    r"\.(com|net|org|edu|gov|info|biz|io|co|us|uk|de|ru|cn|tk|ml|ga|cf|gq|xyz|top|club|online|site)(?=[/.:?#&=\-_]|$)"
)
_DOMAIN_LIKE = re.compile(r"(?:[a-z0-9\-]+\.)+[a-z]{2,6}")
_JS_OBFUSCATION = ("eval(", "unescape(", "fromcharcode", "atob(", "document.write", "%3cscript")
_SLD_PARTS = {"co", "com", "org", "net", "gov", "ac", "edu"}


def _registered_domain(host: str) -> str:
    labels = [p for p in host.split(".") if p]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _SLD_PARTS and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def extract_paper_features(url: str) -> dict[str, float]:
    """Full lexical feature dict (``PAPER_FEATURE_NAMES``). No network access."""
    base = extract_url_features(url)
    raw = (url or "").strip()
    out = dict(base)
    if not raw:
        out.update({name: 0.0 for name in _EXTRA_FEATURE_NAMES})
        return out
    lowered = raw.lower()
    has_scheme = "://" in raw
    try:
        parsed = urlparse(raw if has_scheme else f"http://{raw}")
        host = (parsed.hostname or "").lower()
        path = parsed.path or ""
        query = parsed.query or ""
        userinfo = "@" in (parsed.netloc or "")
    except Exception:
        out.update({name: 0.0 for name in _EXTRA_FEATURE_NAMES})
        return out
    labels = [p for p in host.split(".") if p]
    path_tokens = [t for t in re.split(r"[/\-_.=?&]+", path.lower()) if t]
    n = max(len(raw), 1)
    n_digits = sum(ch.isdigit() for ch in raw)
    n_alpha = sum(ch.isalpha() for ch in raw)
    reg = _registered_domain(host)
    embedded = [d for d in _DOMAIN_LIKE.findall(path.lower() + " " + query.lower())]
    mismatch = 1.0 if userinfo else 0.0
    if embedded and any(_registered_domain(d) != reg for d in embedded):
        mismatch = 1.0

    def has(term: str) -> float:
        return 1.0 if term in lowered else 0.0

    out.update(
        {
            "scheme_http": 1.0 if parsed.scheme == "http" and has_scheme else 0.0,
            "num_digits_path": float(sum(ch.isdigit() for ch in path)),
            "num_digits_query": float(sum(ch.isdigit() for ch in query)),
            "num_hyphens_host": float(host.count("-")),
            "num_delimiters_path": float(sum(path.count(c) for c in "/-_.")),
            "longest_path_token_length": float(max((len(t) for t in path_tokens), default=0)),
            "num_tlds": float(len(_TLD_TOKEN.findall(lowered))),
            "num_subdomains": float(max(len(labels) - 2, 0)),
            "tld_length": float(len(labels[-1]) if labels else 0),
            "longest_host_label_length": float(max((len(p) for p in labels), default=0)),
            "digit_ratio": n_digits / n,
            "host_entropy": _entropy(host),
            "has_login": has("login"),
            "has_verify": has("verify"),
            "has_billing": has("billing"),
            "has_free": has("free"),
            "has_secure": has("secure"),
            "has_admin": has("admin"),
            "has_account": has("account"),
            "has_update": has("update"),
            "has_password": has("password"),
            "javascript_obfuscation": 1.0 if any(t in lowered for t in _JS_OBFUSCATION) else 0.0,
            "iframe_usage": 1.0 if ("iframe" in lowered or "%3ciframe" in lowered) else 0.0,
            "domain_mismatch": mismatch,
            "redirects_chain_length": float(base["redirect_chain_complexity"]),
        }
    )
    _ = n_alpha
    return out


def paper_feature_vector(url: str) -> np.ndarray:
    feats = extract_paper_features(url)
    return np.fromiter((feats[n] for n in PAPER_FEATURE_NAMES), dtype=np.float64,
                       count=len(PAPER_FEATURE_NAMES))


def extract_matrix(urls, feature_names: list[str] | None = None) -> np.ndarray:
    """Vectorise an iterable of URLs into an (n, d) float32 matrix."""
    names = feature_names or PAPER_FEATURE_NAMES
    rows = []
    for u in urls:
        f = extract_paper_features(u)
        rows.append([f[n] for n in names])
    return np.asarray(rows, dtype=np.float32)


def url_feature_vector(url: str) -> np.ndarray:
    feats = extract_url_features(url)
    return np.asarray([feats[n] for n in URL_FEATURE_NAMES], dtype=np.float64)
