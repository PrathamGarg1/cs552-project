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
  const pPhish = 1 - pLegit;
  const predClass = pLegit >= 0.5 ? 1 : 0;

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
