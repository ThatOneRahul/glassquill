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

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** Rates and fractions read better as percentages. */
export const fmtFeature = (feature: string, v: number) =>
  /_(frac|rate|ratio)$/.test(feature) ? pct(v) : v.toFixed(2);
