/** Lexical URL features — mirrors Python extract_url_features (paper Table 3). */

export const FEATURE_NAMES = [
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
  "num_params"
] as const;

const SUSPICIOUS_TERMS = [
  "login","signin","verify","account","secure","payment","billing","free","prize",
  "password","support","authenticate","service","reset","unlock","transaction",
  "claim","winning","alert","suspend","reactivate","update","confirm","wallet","credential",
];

const BRANDS = [
  "paypal","apple","microsoft","google","amazon","facebook","instagram","netflix",
  "whatsapp","bankofamerica","wellsfargo","chase","citibank","icloud","outlook",
  "office365","linkedin","dropbox","adobe",
];

const SHORTENERS = [
  "bit.ly","tinyurl.com","t.co","goo.gl","ow.ly","is.gd","buff.ly","adf.ly","bit.do","cutt.ly",
];

const SUSPICIOUS_TLDS = new Set([
  "tk","ml","ga","cf","gq","xyz","top","club","work","click","link","info","zip","mov",
]);

const SUSPICIOUS_EXTS = [
  ".exe",".zip",".rar",".scr",".js",".jar",".apk",".bat",".cmd",".vbs",
];

const IP = /^(\d{1,3}\.){3}\d{1,3}$/;
const HOMOGLYPH = /[\u0400-\u04FF]/;

function entropy(text: string): number {
  if (!text) return 0;
  const counts = new Map<string, number>();
  for (const ch of text) counts.set(ch, (counts.get(ch) || 0) + 1);
  const n = text.length;
  let e = 0;
  for (const c of counts.values()) {
    const p = c / n;
    e -= p * Math.log2(p);
  }
  return e;
}

export function extractUrlFeatures(url: string): Record<string, number> {
  const raw = (url || "").trim();
  if (!raw) return Object.fromEntries(FEATURE_NAMES.map((n) => [n, 0]));

  let parsed: URL;
  try {
    parsed = new URL(raw.includes("://") ? raw : `http://${raw}`);
  } catch {
    return Object.fromEntries(FEATURE_NAMES.map((n) => [n, 0]));
  }

  const host = parsed.hostname || "";
  const path = parsed.pathname || "";
  const query = parsed.search ? parsed.search.slice(1) : "";
  const lowered = raw.toLowerCase();
  const labels = host.split(".").filter(Boolean);
  const tld = labels.length ? labels[labels.length - 1] : "";
  const tokens = lowered.split(/[/\-_.=?&]+/).filter(Boolean);
  const specials = [...raw].filter((ch) => "@-_=%&?./+#".includes(ch)).length;
  const digits = [...raw].filter((ch) => /\d/.test(ch)).length;
  const nonAlnum = [...raw].filter((ch) => !/[a-zA-Z0-9]/.test(ch)).length;
  const termHits = SUSPICIOUS_TERMS.reduce((n, t) => n + (lowered.includes(t) ? 1 : 0), 0);
  const brandHits = BRANDS.reduce((n, b) => n + (lowered.includes(b) ? 1 : 0), 0);
  const encoded = (lowered.match(/%/g) || []).length;
  const jsData = lowered.includes("javascript:") || lowered.includes("data:") ? 1 : 0;
  const hasShort = SHORTENERS.some((s) => host.includes(s)) ? 1 : 0;
  const susTld = SUSPICIOUS_TLDS.has(tld) ? 1 : 0;
  const pathLower = path.toLowerCase();
  const susExt = SUSPICIOUS_EXTS.some((ext) => pathLower.endsWith(ext)) ? 1 : 0;
  const homo = HOMOGLYPH.test(raw) ? 1 : 0;
  const params = query ? query.split("&").filter(Boolean).length : 0;
  const redirectMarkers = ["redirect", "redir", "url=", "returnurl", "continue=", "next="].reduce(
    (n, tok) => n + (lowered.split(tok).length - 1),
    0,
  );
  const multiRedir = redirectMarkers >= 2 || (lowered.split("http").length - 1) >= 2 ? 1 : 0;
  const redirScore = redirectMarkers + (multiRedir ? 2 : 0) + (raw.includes("@") ? 1 : 0);

  return {
    url_length: raw.length,
    domain_length: host.length,
    path_length: path.length,
    token_count: tokens.length,
    num_subdirectories: (path.match(/\//g) || []).length,
    url_shortening: hasShort,
    special_char_count: specials,
    number_count: digits,
    suspicious_tld: susTld,
    has_port: parsed.port ? 1 : 0,
    entropy: entropy(lowered),
    is_ip: IP.test(host) ? 1 : 0,
    num_dots: (raw.match(/\./g) || []).length,
    num_slashes: (raw.match(/\//g) || []).length,
    num_hyphens: (raw.match(/-/g) || []).length,
    num_at: (raw.match(/@/g) || []).length,
    num_equals: (raw.match(/=/g) || []).length,
    num_qmark: (raw.match(/\?/g) || []).length,
    num_ampersand: (raw.match(/&/g) || []).length,
    num_percent: encoded,
    num_underscore: (raw.match(/_/g) || []).length,
    non_alphanumeric_chars: nonAlnum,
    homoglyph_detection: homo,
    url_encoded: encoded >= 2 ? 1 : 0,
    javascript_data_uri: jsData,
    suspicious_file_extension: susExt,
    suspicious_term_frequency: termHits,
    brand_name_proximity: brandHits,
    multiple_redirects: multiRedir,
    redirect_chain_complexity: redirScore,
    has_https: parsed.protocol === "https:" ? 1 : 0,
    query_length: query.length,
    num_params: params,
  };
}

export function featureVector(url: string): number[] {
  const feats = extractUrlFeatures(url);
  return FEATURE_NAMES.map((n) => feats[n] ?? 0);
}
