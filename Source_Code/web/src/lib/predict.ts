import { featureVector } from "./features";

export type ModelWeights = {
  feature_names: string[];
  classes: number[];
  coef: number[];
  intercept: number;
  scaler_mean: number[];
  scaler_scale: number[];
  metrics?: Record<string, number>;
};

export type Prediction = {
  label: "Phishing" | "Legitimate";
  phishingProbability: number;
  features: { name: string; value: number }[];
};

/** Frontend-only allowlist: well-known brands always score as legitimate. */
const LEGIT_ALLOWLIST = [
  "wikipedia.org",
  "google.com",
  "youtube.com",
  "youtu.be",
  "facebook.com",
  "fb.com",
  "instagram.com",
  "twitter.com",
  "x.com",
  "linkedin.com",
  "github.com",
  "microsoft.com",
  "apple.com",
  "amazon.com",
  "netflix.com",
  "reddit.com",
  "paypal.com",
  "whatsapp.com",
  "tiktok.com",
  "openai.com",
  "chatgpt.com",
  "cursor.com",
  "uci.edu",
  "stackexchange.com",
  "stackoverflow.com",
] as const;

function extractHostname(rawUrl: string): string | null {
  try {
    const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(rawUrl)
      ? rawUrl
      : `https://${rawUrl}`;
    return new URL(withScheme).hostname.toLowerCase().replace(/\.$/, "");
  } catch {
    return null;
  }
}

/** True if host is exactly a listed domain or a subdomain of one. */
export function isAllowlistedLegitHost(rawUrl: string): boolean {
  const host = extractHostname(rawUrl);
  if (!host) return false;
  return LEGIT_ALLOWLIST.some(
    (domain) => host === domain || host.endsWith(`.${domain}`),
  );
}

function sigmoid(z: number): number {
  if (z >= 0) {
    const ez = Math.exp(-z);
    return 1 / (1 + ez);
  }
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

/** Score URL with exported online-SGD lexical model. class 0 = phishing. */
export function predictUrl(url: string, model: ModelWeights): Prediction {
  const raw = featureVector(url);
  const x = raw.map((v, i) => {
    const scale = model.scaler_scale[i] || 1;
    return (v - model.scaler_mean[i]) / scale;
  });

  let z = model.intercept;
  for (let i = 0; i < model.coef.length; i++) z += model.coef[i] * x[i];

  // Pipeline predicts P(class=1) via logistic; classes [0,1] => phishing is 0
  const pLegit = sigmoid(z);
  let pPhish = 1 - pLegit;
  let predClass = pLegit >= 0.5 ? 1 : 0;

  // Frontend override: popular brands never show as phishing in the demo UI.
  if (isAllowlistedLegitHost(url)) {
    predClass = 1;
    pPhish = Math.min(pPhish, 0.05);
  }

  const features = model.feature_names
    .map((name, i) => ({ name, value: Number(raw[i].toFixed(4)) }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return {
    label: predClass === 0 ? "Phishing" : "Legitimate",
    phishingProbability: pPhish,
    features,
  };
}
