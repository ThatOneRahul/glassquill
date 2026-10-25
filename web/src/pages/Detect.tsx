import { AnimatePresence, motion, useAnimate } from "motion/react";
import {
  ArrowDownRight,
  ArrowUpRight,
  ClipboardPaste,
  Copy,
  Eraser,
  Info,
  ScanText,
  Search,
  Sparkles,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Bar, Button, CountUp, PageHeader, Segmented, Skeleton, Toast } from "../components/ui";
import { analyze, ApiError, type Analysis, type Contribution, type Verdict } from "../lib/api";
import { fmtFeature, useMeta } from "../lib/hooks";
import { ease, group, lift, pop, spring } from "../lib/motion";
import { Seo } from "../lib/seo";

const VERDICT: Record<Verdict, { title: string; text: string; bg: string; ring: string }> = {
  ai: { title: "Likely AI-generated", text: "text-ai", bg: "bg-ai", ring: "ring-ai/30" },
  human: { title: "Likely human-written", text: "text-human", bg: "bg-human", ring: "ring-human/30" },
  unsure: { title: "Unclear: could be either", text: "text-unsure", bg: "bg-unsure", ring: "ring-unsure/30" },
};
const asWhole = (v: number) => Math.round(v).toString();

export default function Detect() {
  const { meta, error: metaError } = useMeta();
  const minWords = meta?.limits.min_words ?? 50;
  const maxChars = meta?.limits.max_chars ?? 12000;

  const [text, setText] = useState("");
  const [result, setResult] = useState<Analysis | null>(null);
  const [analysed, setAnalysed] = useState("");   // the text the current result belongs to
  const [runId, setRunId] = useState(0);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; tone: "error" | "ok" } | null>(null);
  const [hint, setHint] = useState<string | null>(null);
  const [scope, shake] = useAnimate();
  const abort = useRef<AbortController | null>(null);
  const resultRef = useRef<HTMLDivElement>(null);

  const words = useMemo(() => (text.trim() ? text.trim().split(/\s+/).length : 0), [text]);
  const closeToast = useCallback(() => setToast(null), []);

  async function run() {
    if (words < minWords) {
      setHint(`Add ${minWords - words} more word${minWords - words === 1 ? "" : "s"}: short texts do not give the models enough to go on.`);
      shake(scope.current, { x: [0, -8, 8, -5, 5, 0] }, { duration: 0.4 });
      return;
    }
    setHint(null);
    abort.current?.abort();
    abort.current = new AbortController();
    setLoading(true);
    try {
      const r = await analyze(text, abort.current.signal);
      setResult(r);
      setAnalysed(text);
      setRunId((n) => n + 1);
      if (window.matchMedia("(max-width: 1279px)").matches) {
        requestAnimationFrame(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }));
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") setToast({ msg: e instanceof ApiError ? e.message : "Analysis failed. Please try again.", tone: "error" });
    } finally {
      setLoading(false);
    }
  }

  async function paste() {
    try {
      const clip = await navigator.clipboard.readText();
      if (clip) setText(clip.slice(0, maxChars));
    } catch {
      setToast({ msg: "Your browser blocked clipboard access. Paste with Ctrl+V instead.", tone: "error" });
    }
  }

  function clear() {
    abort.current?.abort();
    setText("");
    setResult(null);
    setHint(null);
  }

  async function copyResult() {
    if (!result) return;
    const top = result.contributions.slice(0, 3).map((c) => `- ${c.name}: ${fmtFeature(c.feature, c.value)} (pushes towards ${c.shap > 0 ? "AI" : "human"})`);
    const summary = [
      `GlassQuill: ${VERDICT[result.verdict].title} (${Math.round(result.p_final * 100)}% AI)`,
      `Style model ${Math.round(result.p_style * 100)}% · Word model ${Math.round(result.p_words * 100)}%`,
      "Top reasons:", ...top,
      "A detector score is a second opinion, never proof on its own.",
    ].join("\n");
    try {
      await navigator.clipboard.writeText(summary);
      setToast({ msg: "Result copied to clipboard.", tone: "ok" });
    } catch {
      setToast({ msg: "Copy failed: clipboard access is blocked.", tone: "error" });
    }
  }

  useEffect(() => () => abort.current?.abort(), []);

  return (
    <>
      <Seo title="Detector" description="Paste 50 to 400 words of English and get a verdict on whether it reads as human-written or AI-generated, with the evidence behind it." />
      <div className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
      <PageHeader eyebrow="Detector" title="Paste a text." accent="Read the evidence.">
        <p>50 to 400 words of English works best. Or try a real text the model never saw while training:</p>
        <div className="mt-5 flex flex-wrap gap-2">
          {meta
            ? meta.examples.map((e) => (
                <motion.button
                  key={e.label}
                  whileHover={{ y: -2 }}
                  whileTap={{ scale: 0.96 }}
                  transition={spring.press}
                  onClick={() => {
                    setText(e.text);
                    setHint(null);
                  }}
                  className="inline-flex h-10 items-center gap-2 rounded-full border border-white/15 px-4 text-sm font-semibold text-white transition-colors hover:border-white/35 hover:bg-white/5"
                >
                  <Sparkles className="size-3.5 text-flame" aria-hidden="true" /> {e.label}
                </motion.button>
              ))
            : metaError
              ? <p className="text-sm text-white/55">Examples are unavailable right now; you can still paste your own text.</p>
              : Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-10 w-36 !rounded-full" />)}
        </div>
      </PageHeader>

      <motion.div initial="hidden" animate="show" variants={group(0.07, 0.25)} className="grid grid-cols-1 gap-4 xl:grid-cols-12">

        {/* ---- input ---- */}
        <motion.section variants={pop} className="card flex flex-col p-5 sm:p-6 xl:col-span-7" aria-labelledby="input-h">
          <div className="flex items-center justify-between gap-3">
            <h2 id="input-h" className="font-semibold">Your text</h2>
            <div className="flex gap-1">
              <IconButton label="Paste from clipboard" onClick={paste}><ClipboardPaste className="size-4" /></IconButton>
              <IconButton label="Clear" onClick={clear} disabled={!text && !result}><Eraser className="size-4" /></IconButton>
            </div>
          </div>
          <div ref={scope} className="mt-3 flex-1">
            <label htmlFor="text" className="sr-only">Text to analyse</label>
            <textarea
              id="text"
              value={text}
              maxLength={maxChars}
              onChange={(e) => {
                setText(e.target.value);
                if (hint) setHint(null);
              }}
              onKeyDown={(e) => {
                if ((e.ctrlKey || e.metaKey) && e.key === "Enter") run();
              }}
              aria-invalid={!!hint}
              aria-describedby="text-help"
              placeholder="Paste an English text of 50 to 400 words…"
              className={`block h-64 w-full resize-y rounded-2xl border bg-canvas/60 p-4 text-[15px] leading-relaxed text-ink outline-none transition-colors placeholder:text-ink/35 focus:border-violet focus:bg-white sm:h-72 dark:bg-white/[0.04] dark:text-white dark:placeholder:text-white/50 dark:focus:bg-white/[0.06] ${hint ? "border-ai" : "border-line dark:border-white/10"}`}
            />
          </div>
          <div id="text-help" className="mt-3" aria-live="polite">
            <AnimatePresence mode="wait" initial={false}>
              {hint ? (
                <motion.p key="hint" initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} className="text-sm text-ai">{hint}</motion.p>
              ) : (
                <motion.div key="count" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex items-center gap-3">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-black/5 dark:bg-white/10">
                    <motion.div className={`h-full origin-left rounded-full ${words >= minWords ? "bg-human" : "bg-violet"}`} animate={{ scaleX: Math.min(1, words / minWords) }} transition={spring.layout} />
                  </div>
                  <span className="num shrink-0 text-xs text-ink/60 dark:text-white/55">
                    {words} words{words > 400 ? " · only the first ~256 tokens are scored" : words >= minWords ? " · ready" : ` · ${minWords - words} to go`}
                  </span>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="hidden text-xs text-ink/45 sm:block dark:text-white/55">
              <kbd className="rounded border border-line px-1.5 py-0.5 font-sans dark:border-white/15">Ctrl</kbd> +{" "}
              <kbd className="rounded border border-line px-1.5 py-0.5 font-sans dark:border-white/15">Enter</kbd> to analyse
            </p>
            <Button onClick={run} loading={loading} icon={<ScanText className="size-4" aria-hidden="true" />} className="w-full sm:w-auto sm:min-w-40">
              {loading ? "Analysing…" : "Analyse"}
            </Button>
          </div>
        </motion.section>

        {/* ---- verdict ---- */}
        <motion.section ref={resultRef} variants={pop} className="card relative overflow-hidden p-5 sm:p-6 xl:col-span-5" aria-labelledby="verdict-h" aria-live="polite" aria-busy={loading}>
          <h2 id="verdict-h" className="font-semibold">Verdict</h2>
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.div key="load" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="mt-6 space-y-4">
                <Skeleton className="h-6 w-40" />
                <Skeleton className="h-20 w-48" />
                <Skeleton className="h-3 w-full" />
                <div className="grid grid-cols-2 gap-3"><Skeleton className="h-20" /><Skeleton className="h-20" /></div>
              </motion.div>
            ) : result ? (
              <VerdictView key={`r${runId}`} r={result} onCopy={copyResult} stale={text.trim() !== analysed.trim()} />
            ) : (
              <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="grid min-h-72 place-items-center text-center">
                <div>
                  <div className="mx-auto grid size-14 place-items-center rounded-2xl bg-lavender-soft text-violet dark:bg-violet/15 dark:text-lavender">
                    <ScanText className="size-6" aria-hidden="true" />
                  </div>
                  <p className="mt-4 font-semibold">No text analysed yet</p>
                  <p className="mx-auto mt-1 max-w-60 text-sm text-ink/55 dark:text-white/50">Paste a text or pick an example, then press Analyse.</p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.section>

        {/* ---- how to read the score ---- */}
        <motion.section variants={pop} className="card flex flex-wrap items-center gap-x-8 gap-y-3 px-6 py-4 xl:col-span-12" aria-labelledby="bands">
          <h2 id="bands" className="flex items-center gap-2 text-sm font-semibold">
            <Info className="size-4 text-white/55" aria-hidden="true" /> Reading the verdict
          </h2>
          <ul className="flex flex-wrap gap-x-6 gap-y-2 text-sm">
            <Band color="bg-ai" label="Likely AI" range="65 – 100%" />
            <Band color="bg-unsure" label="Unclear" range="35 – 65%" />
            <Band color="bg-human" label="Likely human" range="0 – 35%" />
          </ul>
          <p className="text-xs text-white/55 lg:ml-auto">Average of two models. A second opinion with reasons, never proof.</p>
        </motion.section>

        {/* ---- explanations (appear after a result) ---- */}
        <AnimatePresence>
          {result && !loading && (
            <motion.div key={`x${runId}`} className="grid grid-cols-1 gap-4 xl:col-span-12 xl:grid-cols-12" initial="hidden" animate="show" exit={{ opacity: 0 }} variants={group(0.08, 0.15)}>
              <Reasons items={result.contributions.slice(0, 5)} />
              <ShapChart items={result.contributions} />
              <WordEvidence highlights={result.highlights} />
              <FeatureTable items={result.contributions} />
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
      </div>

      <Toast message={toast?.msg ?? null} tone={toast?.tone} onClose={closeToast} />
    </>
  );
}

/* ------------------------------------------------------------------ */

function Band({ color, label, range }: { color: string; label: string; range: string }) {
  return (
    <li className="flex items-center gap-2">
      <span className={`size-2.5 rounded-full ${color}`} aria-hidden="true" />
      <span className="font-medium">{label}</span>
      <span className="num text-ink/55 dark:text-white/50">{range}</span>
    </li>
  );
}

function IconButton({ label, onClick, disabled, children }: { label: string; onClick: () => void; disabled?: boolean; children: React.ReactNode }) {
  return (
    <motion.button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
      whileHover={disabled ? undefined : { scale: 1.06 }}
      whileTap={disabled ? undefined : { scale: 0.92 }}
      transition={spring.press}
      className="grid size-9 place-items-center rounded-full text-ink/60 hover:bg-black/5 hover:text-ink disabled:opacity-35 dark:text-white/60 dark:hover:bg-white/10 dark:hover:text-white"
    >
      {children}
    </motion.button>
  );
}

function VerdictView({ r, onCopy, stale }: { r: Analysis; onCopy: () => void; stale: boolean }) {
  const v = VERDICT[r.verdict];
  const [copied, setCopied] = useState(false);
  return (
    <motion.div initial="hidden" animate="show" exit={{ opacity: 0 }} variants={group(0.08)} className="mt-4">
      <AnimatePresence initial={false}>
        {stale && (
          <motion.p
            role="status"
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            className="mb-3 rounded-xl bg-unsure/10 px-3 py-2 text-xs font-medium text-unsure"
          >
            You've edited the text since this result. Press Analyse to update it.
          </motion.p>
        )}
      </AnimatePresence>
      <motion.span variants={pop} className={`inline-flex items-center gap-2 rounded-full bg-current/10 px-3 py-1 text-sm font-semibold ${v.text}`}>
        <span className={`size-2 rounded-full ${v.bg}`} aria-hidden="true" /> {v.title}
      </motion.span>
      <motion.p variants={pop} className={`mt-3 flex items-baseline ${v.text}`}>
        <CountUp value={r.p_final * 100} format={asWhole} className="text-7xl font-extrabold" />
        <span className="ml-1 text-3xl font-bold">%</span>
        <span className="ml-3 text-sm font-medium text-ink/55 dark:text-white/60">chance it is AI-generated</span>
      </motion.p>
      <motion.div variants={pop} className="relative mt-4 h-2.5 rounded-full bg-gradient-to-r from-human/25 via-unsure/25 to-ai/25">
        {/* full-width wrapper slid by x%: a transform, so no layout work per frame */}
        <motion.div
          className="pointer-events-none absolute inset-0"
          initial={{ x: "0%" }}
          animate={{ x: `${r.p_final * 100}%` }}
          transition={{ duration: 1.1, ease: ease.out, delay: 0.2 }}
        >
          <span className={`absolute top-1/2 left-0 size-4 -translate-x-1/2 -translate-y-1/2 rounded-full ring-4 ring-white dark:ring-ink-2 ${v.bg}`} />
        </motion.div>
      </motion.div>
      {/* reference: big split numerals "46 | 74" */}
      <motion.div variants={pop} className="mt-6 grid grid-cols-2 gap-3">
        <ModelScore label="Style model" sub="XGBoost · 18 signals" value={r.p_style} />
        <ModelScore label="Word model" sub="TF-IDF + LR" value={r.p_words} />
      </motion.div>
      <motion.div variants={pop} className="mt-5 flex flex-wrap items-center justify-between gap-3">
        <p className={`text-sm ${r.models_agree ? "text-ink/60 dark:text-white/55" : "font-medium text-unsure"}`}>
          {r.models_agree ? "Both models agree." : "The models disagree. Treat this with extra caution."}
          <span className="block text-xs text-ink/40 dark:text-white/55">{r.words} words · analysed in {r.ms} ms</span>
        </p>
        <Button
          tone="outline"
          className="h-9 px-4 text-xs"
          done={copied}
          icon={<Copy className="size-3.5" aria-hidden="true" />}
          onClick={() => {
            onCopy();
            setCopied(true);
            setTimeout(() => setCopied(false), 1800);
          }}
        >
          {copied ? "Copied" : "Copy result"}
        </Button>
      </motion.div>
    </motion.div>
  );
}

function ModelScore({ label, sub, value }: { label: string; sub: string; value: number }) {
  return (
    <motion.div whileHover={lift} className="rounded-2xl bg-canvas/70 p-4 dark:bg-white/[0.04]">
      <p className="text-xs font-medium text-ink/55 dark:text-white/50">{label}</p>
      <p className="num mt-1 text-3xl font-extrabold">
        <CountUp value={value * 100} format={asWhole} />
        <span className="text-lg">%</span>
      </p>
      <p className="mb-3 text-[11px] text-ink/40 dark:text-white/55">{sub}</p>
      <Bar value={value} className="bg-violet" delay={0.3} />
    </motion.div>
  );
}

function Direction({ shap }: { shap: number }) {
  const ai = shap > 0;
  return (
    <span className={`inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 text-xs font-bold ${ai ? "bg-ai/10 text-ai" : "bg-human/10 text-human"}`}>
      {ai ? <ArrowUpRight className="size-3.5" aria-hidden="true" /> : <ArrowDownRight className="size-3.5" aria-hidden="true" />}
      {ai ? "AI" : "human"}
    </span>
  );
}

function Reasons({ items }: { items: Contribution[] }) {
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-5" aria-labelledby="reasons-h">
      <h2 id="reasons-h" className="font-semibold">Top reasons</h2>
      <p className="text-xs text-ink/50 dark:text-white/55">From the style model, compared with typical writing</p>
      <motion.ul className="mt-4 divide-y divide-line dark:divide-white/10" variants={group(0.06)}>
        {items.map((c) => (
          <motion.li key={c.feature} variants={{ hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0 } }} className="flex items-center justify-between gap-3 py-3">
            <div className="min-w-0">
              <p className="text-sm">
                <span className="font-semibold">{c.name}</span> is <span className="num font-semibold">{fmtFeature(c.feature, c.value)}</span>
              </p>
              {c.typical && (
                <p className="text-xs text-ink/50 dark:text-white/55">
                  humans ~{fmtFeature(c.feature, c.typical.human)} · AI ~{fmtFeature(c.feature, c.typical.ai)}
                </p>
              )}
            </div>
            <Direction shap={c.shap} />
          </motion.li>
        ))}
      </motion.ul>
    </motion.section>
  );
}

/** Diverging bars: how far each feature pushed the style model, left = human, right = AI. */
function ShapChart({ items }: { items: Contribution[] }) {
  const top = items.slice(0, 10);
  const max = Math.max(...top.map((c) => Math.abs(c.shap)), 1e-6);
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-7" aria-labelledby="shap-h">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div>
          <h2 id="shap-h" className="font-semibold">Why: feature contributions</h2>
          <p className="text-xs text-ink/50 dark:text-white/55">SHAP values of the style model, 10 strongest of 18</p>
        </div>
        <div className="flex gap-3 text-xs">
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-human" />towards human</span>
          <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-ai" />towards AI</span>
        </div>
      </div>
      <ul className="mt-5 space-y-2.5" aria-label="Feature contributions">
        {top.map((c, i) => {
          const w = Math.abs(c.shap) / max;
          const ai = c.shap > 0;
          return (
            <li key={c.feature} className="grid grid-cols-[minmax(0,9rem)_1fr] items-center gap-3 sm:grid-cols-[12rem_1fr]">
              <span className="truncate text-xs font-medium text-ink/70 dark:text-white/65" title={c.name}>{c.name}</span>
              <div className="relative grid h-6 grid-cols-2">
                <span className="absolute inset-y-0 left-1/2 w-px bg-line dark:bg-white/15" aria-hidden="true" />
                <div className="flex justify-end">
                  {!ai && (
                    <motion.span className="h-full origin-right rounded-l-md bg-human" style={{ width: `${w * 100}%` }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 0.7, ease: ease.out, delay: 0.2 + i * 0.04 }} />
                  )}
                </div>
                <div className="flex">
                  {ai && (
                    <motion.span className="h-full origin-left rounded-r-md bg-ai" style={{ width: `${w * 100}%` }} initial={{ scaleX: 0 }} animate={{ scaleX: 1 }} transition={{ duration: 0.7, ease: ease.out, delay: 0.2 + i * 0.04 }} />
                  )}
                </div>
                <span className={`num pointer-events-none absolute top-1/2 -translate-y-1/2 text-[11px] font-semibold ${ai ? "right-1 text-ai" : "left-1 text-human"}`}>
                  {c.shap > 0 ? "+" : ""}{c.shap.toFixed(2)}
                </span>
              </div>
            </li>
          );
        })}
      </ul>
    </motion.section>
  );
}

function WordEvidence({ highlights }: { highlights: Analysis["highlights"] }) {
  const [mode, setMode] = useState<"all" | "strong">("all");
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-12" aria-labelledby="words-h">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 id="words-h" className="font-semibold">Word evidence</h2>
          <p className="text-xs text-ink/50 dark:text-white/55">How the word model weighs each word: rose pushes towards AI, green towards human</p>
        </div>
        <Segmented label="Highlight" value={mode} onChange={setMode} options={[{ value: "all", label: "All words" }, { value: "strong", label: "Strongest only" }]} />
      </div>
      <p className="mt-4 text-[15px] leading-8">
        {highlights.map((h, i) => {
          const s = h.score ?? 0;
          const show = h.score !== null && (mode === "all" || Math.abs(s) >= 0.25);
          const alpha = show ? 0.12 + Math.min(1, Math.abs(s)) * 0.5 : 0;
          return (
            <span
              key={i}
              title={show ? `${s > 0 ? "towards AI" : "towards human"} (${s.toFixed(2)})` : undefined}
              className="rounded px-0.5 transition-[background-color] duration-300"
              style={{ backgroundColor: show ? (s > 0 ? `rgb(244 63 94 / ${alpha})` : `rgb(5 150 105 / ${alpha})`) : "transparent" }}
            >
              {h.token}
            </span>
          );
        })}
      </p>
    </motion.section>
  );
}

function FeatureTable({ items }: { items: Contribution[] }) {
  const [q, setQ] = useState("");
  const [dir, setDir] = useState<"all" | "ai" | "human">("all");
  const rows = items.filter(
    (c) =>
      (dir === "all" || (dir === "ai") === c.shap > 0) &&
      (c.name.toLowerCase().includes(q.toLowerCase()) || c.feature.includes(q.toLowerCase())),
  );
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-12" aria-labelledby="table-h">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="table-h" className="font-semibold">All 18 features</h2>
        <div className="flex flex-wrap items-center gap-2">
          <label className="relative">
            <span className="sr-only">Search features</span>
            <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-ink/40 dark:text-white/55" aria-hidden="true" />
            <input
              type="search"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="Search features"
              className="h-9 w-48 rounded-full border border-line bg-transparent pr-3 pl-9 text-sm outline-none focus:border-violet dark:border-white/10"
            />
          </label>
          <Segmented label="Filter by direction" value={dir} onChange={setDir} options={[{ value: "all", label: "All" }, { value: "ai", label: "→ AI" }, { value: "human", label: "→ human" }]} />
        </div>
      </div>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="text-left text-xs text-ink/50 dark:text-white/55">
              <th scope="col" className="py-2 pr-3 font-medium">Feature</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">This text</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Typical human</th>
              <th scope="col" className="py-2 pr-3 text-right font-medium">Typical AI</th>
              <th scope="col" className="py-2 text-right font-medium">Pushes towards</th>
            </tr>
          </thead>
          <tbody>
            <AnimatePresence initial={false}>
              {rows.map((c) => (
                <motion.tr key={c.feature} layout="position" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="border-t border-line dark:border-white/10">
                  <td className="py-2.5 pr-3">
                    <span className="font-medium">{c.name}</span>
                    <span className="block font-mono text-[11px] text-ink/40 dark:text-white/55">{c.feature}</span>
                  </td>
                  <td className="num py-2.5 pr-3 text-right font-semibold">{fmtFeature(c.feature, c.value)}</td>
                  <td className="num py-2.5 pr-3 text-right text-ink/60 dark:text-white/55">{c.typical ? fmtFeature(c.feature, c.typical.human) : "–"}</td>
                  <td className="num py-2.5 pr-3 text-right text-ink/60 dark:text-white/55">{c.typical ? fmtFeature(c.feature, c.typical.ai) : "–"}</td>
                  <td className="py-2.5 text-right"><Direction shap={c.shap} /> <span className="num ml-1 text-xs text-ink/45 dark:text-white/55">{c.shap > 0 ? "+" : ""}{c.shap.toFixed(2)}</span></td>
                </motion.tr>
              ))}
            </AnimatePresence>
          </tbody>
        </table>
        {rows.length === 0 && <p className="py-8 text-center text-sm text-ink/50 dark:text-white/55">No features match “{q}”.</p>}
      </div>
    </motion.section>
  );
}
