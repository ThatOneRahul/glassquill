/** Typed client for server.py. All numbers shown on the site come from here. */

export type Verdict = "ai" | "human" | "unsure";

export interface Contribution {
  feature: string;
  name: string;
  value: number;
  shap: number;
  typical: { human: number; ai: number } | null;
}

export interface Analysis {
  verdict: Verdict;
  p_final: number;
  p_style: number;
  p_words: number;
  models_agree: boolean;
  shap_base: number;
  contributions: Contribution[];
  highlights: { token: string; score: number | null }[];
  words: number;
  ms: number;
}

export type Row = Record<string, string | number>;

export interface Meta {
  examples: { label: string; text: string }[];
  typical: Record<string, { human: number; ai: number }>;
  limits: { min_words: number; max_chars: number; unsure: [number, number] };
  results: {
    in_domain: Row[];
    by_corpus: Row[];
    out_of_domain: Row[];
    paraphrase: Row[];
    ood_by_generator: Row[];
    shap_global: Row[];
    sources: Row[];
  };
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { "Content-Type": "application/json" } });
  } catch {
    throw new ApiError("Can't reach the GlassQuill server. Is `python server.py` running?");
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : "Something went wrong. Please try again.";
    throw new ApiError(detail);
  }
  return body as T;
}

export const analyze = (text: string, signal?: AbortSignal) =>
  request<Analysis>("/api/analyze", { method: "POST", body: JSON.stringify({ text }), signal });

let metaPromise: Promise<Meta> | null = null;
/** Fetched once per page load and shared by every page. */
export const getMeta = () => (metaPromise ??= request<Meta>("/api/meta").catch((e) => {
  metaPromise = null;
  throw e;
}));

/** The app's verdict rule (mean of XGBoost and TF-IDF + LR), as named in the reports. */
export const APP_MODEL = "XGBoost + LR (app)";
