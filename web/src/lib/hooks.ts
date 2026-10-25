import { useEffect, useState } from "react";
import { getMeta, type Meta } from "./api";

/** Shared /api/meta data with loading and error state. */
export function useMeta() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let live = true;
    getMeta()
      .then((m) => live && setMeta(m))
      .catch((e: Error) => live && setError(e.message));
    return () => {
      live = false;
    };
  }, []);
  return { meta, error };
}

export type Theme = "light" | "dark";

/** Dashboard theme, remembered per browser. The landing page is always dark. */
export function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem("gq-theme");
      if (saved === "light" || saved === "dark") return saved;
    } catch {
      /* storage blocked: fall back to the system setting */
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  });
  useEffect(() => {
    try {
      localStorage.setItem("gq-theme", theme);
    } catch {
      /* ignore */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** Rates and fractions read better as percentages. */
export const fmtFeature = (feature: string, v: number) =>
  /_(frac|rate|ratio)$/.test(feature) ? pct(v) : v.toFixed(2);
