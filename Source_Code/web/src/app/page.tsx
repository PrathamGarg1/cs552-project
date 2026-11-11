"use client";

import { FormEvent, useEffect, useState } from "react";
import type { ModelWeights, Prediction } from "@/lib/predict";
import { predictUrl } from "@/lib/predict";

const EXAMPLES = [
  "https://www.wikipedia.org",
  "https://github.com",
  "http://paypa1-secure-login.tk/confirm?user=1",
  "http://192.168.1.10/login/verify-account",
];

const PIPELINE = [
  { title: "URL", detail: "lexical parse" },
  { title: "LASSO+MI", detail: "feature select" },
  { title: "KS δ=0.05", detail: "drift check" },
  { title: "SGD-SVM", detail: "online hinge" },
  { title: "Verdict", detail: "phish / legit" },
] as const;

export default function HomePage() {
  const [model, setModel] = useState<ModelWeights | null>(null);
  const [url, setUrl] = useState("");
  const [touched, setTouched] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const [loadingModel, setLoadingModel] = useState(true);

  useEffect(() => {
    fetch("/model.json")
      .then((r) => {
        if (!r.ok) throw new Error("Model failed to load");
        return r.json();
      })
      .then((m: ModelWeights) => setModel(m))
      .catch(() => setError("Could not load detection model."))
      .finally(() => setLoadingModel(false));
  }, []);

  function validate(value: string): string | null {
    const v = value.trim();
    if (!v) return "Enter a URL to search.";
    if (!/^https?:\/\//i.test(v) && !v.includes(".")) {
      return "Use a full URL, e.g. https://example.com";
    }
    return null;
  }

  function runSearch(value: string) {
    setTouched(true);
    const msg = validate(value);
    if (msg) {
      setError(msg);
      setResult(null);
      return;
    }
    if (!model) {
      setError("Model is still loading.");
      return;
    }
    setError(null);
    setResult(predictUrl(value.trim(), model));
  }

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    runSearch(url);
  }

  const showError = touched && error;

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col px-5 py-10 sm:py-16">
      <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-[var(--brand)]">
        CS 552 Project
      </p>
      <h1 className="brand-display text-4xl leading-tight text-[var(--ink)] sm:text-5xl">
        PhishCheck
      </h1>
      <p className="mt-2 text-sm font-medium text-[var(--ink)]">
        Pratham Garg · Sumit Sharma · Vivaan Garg
      </p>
      <p className="mt-3 max-w-xl text-base text-[#3d4a45] sm:text-lg">
        Check whether a URL looks phishing, based on{" "}
        <a
          className="underline decoration-[var(--brand)] underline-offset-2"
          href="https://onlinelibrary.wiley.com/doi/full/10.1002/spy2.70057"
          target="_blank"
          rel="noreferrer"
        >
          Krishnan &amp; Naik (Security and Privacy, 2025)
        </a>
        .
      </p>

      <form
        onSubmit={onSubmit}
        className="mt-10"
        noValidate
        role="search"
        aria-describedby={showError ? "url-error" : undefined}
      >
        <label htmlFor="url" className="mb-1.5 block text-sm font-medium">
          Search URL
        </label>
        <div className="flex flex-col gap-3 sm:flex-row sm:items-stretch">
          <input
            id="url"
            name="q"
            type="search"
            inputMode="url"
            autoComplete="url"
            spellCheck={false}
            autoCapitalize="off"
            autoCorrect="off"
            enterKeyHint="search"
            placeholder="https://example.com/login"
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              if (touched) setError(validate(e.target.value));
            }}
            onBlur={() => setTouched(true)}
            required
            aria-invalid={Boolean(showError)}
            aria-describedby={showError ? "url-error" : "url-help"}
            className="min-w-0 flex-1 rounded-xl border border-black/10 bg-white/80 px-4 py-3 text-base shadow-sm backdrop-blur"
          />
          <button
            type="submit"
            disabled={loadingModel || !model}
            className="shrink-0 rounded-xl bg-[var(--brand)] px-6 py-3 text-base font-semibold text-white transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-60 sm:min-w-[8.5rem]"
          >
            {loadingModel ? "Loading…" : "Search"}
          </button>
        </div>
        <p id="url-help" className="mt-1.5 text-sm text-[#5a6762]">
          Press Search or Enter. Scoring uses URL text only.
        </p>
        {showError ? (
          <p
            id="url-error"
            className="mt-2 text-sm font-medium text-[var(--phish)]"
            role="alert"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button
              key={ex}
              type="button"
              className="rounded-full border border-black/10 bg-white/60 px-3 py-1 text-xs text-[#3d4a45] hover:bg-white"
              onClick={() => {
                setUrl(ex);
                runSearch(ex);
              }}
            >
              {ex.replace(/^https?:\/\//, "").slice(0, 28)}
              {ex.length > 36 ? "…" : ""}
            </button>
          ))}
        </div>
      </form>

      {result ? (
        <section className="mt-10" aria-live="polite">
          <div
            className={`inline-flex items-center gap-3 rounded-full px-4 py-2 text-sm font-semibold ${
              result.label === "Phishing"
                ? "bg-[#f8d7da] text-[var(--phish)]"
                : "bg-[#d1e7dd] text-[var(--legit)]"
            }`}
          >
            <span>{result.label}</span>
            <span className="font-normal opacity-80">
              phishing probability{" "}
              {(result.phishingProbability * 100).toFixed(1)}%
            </span>
          </div>

          <h2 className="brand-display mt-8 text-2xl">Top lexical signals</h2>
          <ul className="mt-3 divide-y divide-black/5 rounded-2xl border border-black/5 bg-white/70">
            {result.features.map((f) => (
              <li
                key={f.name}
                className="flex justify-between px-4 py-2.5 text-sm"
              >
                <span className="text-[#5a6762]">{f.name}</span>
                <span className="font-medium tabular-nums">{f.value}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="mt-14" aria-labelledby="pipeline-heading">
        <h2
          id="pipeline-heading"
          className="brand-display text-2xl text-[var(--ink)]"
        >
          Core logic
        </h2>
        <p className="mt-1 text-sm text-[#5a6762]">
          Pipeline used by the adaptive detector.
        </p>

        <ol className="mt-5 flex flex-col gap-0 sm:flex-row sm:flex-wrap sm:items-stretch sm:gap-y-3">
          {PIPELINE.map((step, i) => (
            <li key={step.title} className="flex items-center sm:contents">
              <div className="flex w-full items-center gap-3 rounded-2xl border border-black/8 bg-white/75 px-4 py-3 shadow-sm backdrop-blur sm:w-auto sm:min-w-[7.5rem] sm:flex-1 sm:flex-col sm:items-start sm:gap-1 sm:px-3 sm:py-3">
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-[var(--brand)] text-xs font-bold text-white">
                  {i + 1}
                </span>
                <div>
                  <p className="text-sm font-semibold text-[var(--ink)]">
                    {step.title}
                  </p>
                  <p className="text-xs text-[#5a6762]">{step.detail}</p>
                </div>
              </div>
              {i < PIPELINE.length - 1 ? (
                <span
                  className="mx-2 hidden text-[var(--brand)] sm:inline sm:px-0.5"
                  aria-hidden
                >
                  →
                </span>
              ) : null}
              {i < PIPELINE.length - 1 ? (
                <span
                  className="ml-[0.85rem] h-4 w-px bg-[var(--brand)]/40 sm:hidden"
                  aria-hidden
                />
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      <footer className="mt-auto pt-12 text-xs text-[#5a6762]">
        <a
          className="underline decoration-[var(--brand)] underline-offset-2"
          href="https://onlinelibrary.wiley.com/doi/full/10.1002/spy2.70057"
          target="_blank"
          rel="noreferrer"
        >
          Krishnan &amp; Naik, Security and Privacy 8(4): e70057, 2025
        </a>
      </footer>
    </main>
  );
}
