import { motion, useReducedMotion, useScroll, useSpring, useTransform } from "motion/react";
import { ArrowRight, Fingerprint, Gauge, ScanText, ShieldCheck, Sparkles, Type } from "lucide-react";
import { useRef } from "react";
import { Link } from "react-router-dom";
import { SiteLayout } from "../components/SiteLayout";
import { Bar, CountUp, Petals, Reveal, RevealGroup, Skeleton } from "../components/ui";
import { APP_MODEL, type Row } from "../lib/api";
import { useMeta } from "../lib/hooks";
import { ease, lift, pop, rise, spring } from "../lib/motion";
import { Seo } from "../lib/seo";

const asPct = (v: number) => `${v.toFixed(1)}%`;
const asInt = (v: number) => Math.round(v).toLocaleString("en-IN");
const find = (rows: Row[] | undefined, model: string) => rows?.find((r) => r.model === model);

export default function Home() {
  const { meta } = useMeta();
  const app = find(meta?.results.in_domain, APP_MODEL);
  const appOod = find(meta?.results.out_of_domain, APP_MODEL);
  const bert = find(meta?.results.in_domain, "DistilBERT");
  const texts = meta?.results.sources.reduce((s, r) => s + Number(r.human) + Number(r.ai), 0);

  return (
    <SiteLayout>
      <Seo
        title="GlassQuill"
        description="Is this text written by a person or an AI? GlassQuill gives a verdict and shows the evidence behind it: perplexity, style signals and word-level weights."
      />
      <ScrollProgress />
      <Hero />

      {/* ---- stats strip ---- */}
      <section aria-label="Key results" className="border-y border-white/10 bg-white/[0.02]">
        <RevealGroup className="mx-auto grid max-w-6xl grid-cols-2 gap-px px-4 sm:px-6 md:grid-cols-4">
          <Stat label="Test accuracy" hint="4,829 unseen texts" value={app ? Number(app.accuracy) * 100 : undefined} format={asPct} />
          <Stat label="Humans wrongly flagged" hint="false positive rate" value={app ? Number(app.fpr) * 100 : undefined} format={asPct} />
          <Stat label="Accuracy on 27 unseen generators" hint="MAGE benchmark" value={appOod ? Number(appOod.accuracy) * 100 : undefined} format={asPct} />
          <Stat label="Training texts" hint="17 domains, 50/50 human/AI" value={texts} format={asInt} />
        </RevealGroup>
      </section>

      <HowItWorks />

      {/* ---- two models ---- */}
      <section className="mx-auto max-w-6xl px-4 py-24 sm:px-6 md:py-32">
        <Reveal>
          <p className="eyebrow">Two explainable models</p>
          <h2 className="mt-3 max-w-2xl text-4xl font-bold tracking-tight text-balance md:text-5xl">
            One reads the rhythm. One reads the words.
          </h2>
          <p className="mt-4 max-w-2xl text-white/60">
            The verdict is the average of two models that make different mistakes. Both can show
            exactly why they decided, so a teacher can check the reasoning instead of trusting a number.
          </p>
        </Reveal>
        <RevealGroup className="mt-12 grid gap-4 md:grid-cols-2" gap={0.12}>
          <ModelCard
            icon={Gauge}
            title="Style model"
            sub="XGBoost on 18 signals, explained with SHAP"
            points={[
              "How predictable the text is to GPT-2 (perplexity)",
              "How even the sentence rhythm is (burstiness)",
              "Vocabulary richness, word length, punctuation, readability",
            ]}
          />
          <ModelCard
            icon={Type}
            title="Word model"
            sub="TF-IDF + logistic regression, exact word weights"
            points={[
              "Which words and word pairs appear",
              "Every word gets a weight towards human or AI",
              "Shown as coloured text, so you can read the evidence",
            ]}
          />
        </RevealGroup>
      </section>

      {/* ---- false positives ---- */}
      <section className="relative overflow-hidden border-y border-white/10 bg-ink-2/60">
        <div className="mx-auto grid max-w-6xl items-center gap-12 px-4 py-24 sm:px-6 md:grid-cols-2 md:py-32">
          <Reveal>
            <p className="eyebrow">Built around the costly mistake</p>
            <h2 className="mt-3 text-4xl font-bold tracking-tight text-balance md:text-5xl">
              Wrongly accusing a person is the error that matters.
            </h2>
            <p className="mt-4 text-white/60">
              A false positive flags a human as AI. GlassQuill reports that rate for every model and
              picks its verdict rule to keep it low, while showing an <em>unclear</em> band instead of guessing.
            </p>
          </Reveal>
          <Reveal className="card !bg-white/[0.04] p-6 !shadow-none ring-1 ring-white/10" delay={0.1}>
            <h3 className="text-sm font-semibold text-white/80">Humans wrongly flagged as AI (test set)</h3>
            {app && bert ? (
              <div className="mt-6 space-y-5">
                <FprRow label="GlassQuill verdict" value={Number(app.fpr)} strong />
                <FprRow label="DistilBERT (black box)" value={Number(bert.fpr)} />
              </div>
            ) : (
              <div className="mt-6 space-y-5"><Skeleton className="h-10" /><Skeleton className="h-10" /></div>
            )}
            <p className="mt-6 text-xs text-white/45">Lower is better. Bars run from 0 to 10%. Same 4,829 test texts for both.</p>
          </Reveal>
        </div>
      </section>

      {/* ---- results teaser ---- */}
      <section className="mx-auto max-w-6xl px-4 py-24 sm:px-6 md:py-32">
        <div className="flex flex-wrap items-end justify-between gap-6">
          <Reveal>
            <p className="eyebrow">Measured, not claimed</p>
            <h2 className="mt-3 text-4xl font-bold tracking-tight md:text-5xl">Five models, one test set.</h2>
          </Reveal>
          <Reveal delay={0.1}>
            <Link to="/results" className="group inline-flex items-center gap-2 text-sm font-semibold text-lavender">
              All results <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </Reveal>
        </div>
        <RevealGroup className="mt-10 space-y-4">
          {meta
            ? [...meta.results.in_domain]
                .sort((a, b) => Number(b.accuracy) - Number(a.accuracy))
                .map((r, i) => (
                  <motion.div key={String(r.model)} variants={pop} className="grid grid-cols-[minmax(0,10rem)_1fr_3.5rem] items-center gap-4 sm:grid-cols-[14rem_1fr_4rem]">
                    <span className={`truncate text-sm ${r.model === APP_MODEL ? "font-semibold text-white" : "text-white/60"}`}>
                      {r.model === APP_MODEL ? "GlassQuill verdict" : String(r.model)}
                    </span>
                    <Bar value={Number(r.accuracy)} delay={i * 0.06} track="bg-white/10" className={r.model === APP_MODEL ? "bg-gradient-to-r from-lavender to-violet" : "bg-white/35"} />
                    <span className="num text-right text-sm font-semibold">{(Number(r.accuracy) * 100).toFixed(1)}%</span>
                  </motion.div>
                ))
            : Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-6" />)}
        </RevealGroup>
      </section>

      {/* ---- CTA ---- */}
      <section className="px-4 pb-24 sm:px-6">
        <Reveal className="relative mx-auto max-w-6xl overflow-hidden rounded-[2rem] bg-gradient-to-br from-lilac via-lavender to-violet px-6 py-16 text-ink sm:px-12">
          <Petals className="pointer-events-none absolute -top-16 -right-16 size-72 opacity-60 md:size-96" />
          <h2 className="relative max-w-xl text-4xl font-bold tracking-tight text-balance md:text-5xl">Paste a text. Read the evidence.</h2>
          <p className="relative mt-3 max-w-md text-ink/70">50 to 400 words of English works best. Nothing you paste is stored.</p>
          <motion.div className="relative mt-8 inline-block" whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} transition={spring.press}>
            <Link to="/detect" className="inline-flex h-12 items-center gap-2 rounded-full bg-ink px-6 text-sm font-semibold text-white">
              <ScanText className="size-4" aria-hidden="true" /> Open the detector
            </Link>
          </motion.div>
        </Reveal>
      </section>
    </SiteLayout>
  );
}

/* ------------------------------------------------------------------ */

function ScrollProgress() {
  const { scrollYProgress } = useScroll();
  const scaleX = useSpring(scrollYProgress, { stiffness: 200, damping: 40, restDelta: 0.001 });
  return <motion.div aria-hidden="true" style={{ scaleX }} className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left bg-flame" />;
}

const HEADLINE = ["Human", "or", "AI?"];

function Hero() {
  const ref = useRef<HTMLElement>(null);
  const reduce = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const cardY = useTransform(scrollYProgress, [0, 1], reduce ? [0, 0] : [0, 90]);
  const glowOpacity = useTransform(scrollYProgress, [0, 0.8], [1, 0]);

  return (
    <section ref={ref} className="relative overflow-hidden pt-32 pb-20 md:pt-40 md:pb-28">
      <motion.div
        aria-hidden="true"
        style={{ opacity: glowOpacity }}
        className="pointer-events-none absolute top-0 left-1/2 h-[520px] w-[min(900px,100%)] -translate-x-1/2 rounded-full bg-violet/25 blur-[120px]"
      />
      <div className="relative mx-auto grid max-w-6xl items-center gap-14 px-4 sm:px-6 lg:grid-cols-[1.05fr_1fr]">
        <motion.div initial="hidden" animate="show" variants={{ hidden: {}, show: { transition: { staggerChildren: 0.09, delayChildren: 0.1 } } }}>
          <motion.p variants={rise} className="eyebrow inline-flex items-center gap-2">
            <Sparkles className="size-3.5" aria-hidden="true" /> Explainable AI-text detection
          </motion.p>
          <h1 className="mt-5 text-[clamp(3rem,9vw,6.5rem)] leading-[0.92] font-extrabold tracking-[-0.045em]">
            <span className="sr-only">Human or AI? See the evidence.</span>
            <span aria-hidden="true" className="block">
              {HEADLINE.map((w) => (
                <motion.span key={w} variants={rise} className="mr-[0.22em] inline-block">
                  {w}
                </motion.span>
              ))}
            </span>
            <motion.span aria-hidden="true" variants={rise} className="block text-flame">
              See the evidence.
            </motion.span>
          </h1>
          <motion.p variants={rise} className="mt-6 max-w-lg text-lg leading-relaxed text-white/65">
            GlassQuill tells you whether a text reads as human-written or AI-generated, and shows every
            signal behind the verdict. A second opinion with reasons, never a bare score.
          </motion.p>
          <motion.div variants={rise} className="mt-9 flex flex-wrap gap-3">
            <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} transition={spring.press}>
              <Link to="/detect" className="inline-flex h-12 items-center gap-2 rounded-full bg-white px-6 text-sm font-semibold text-ink">
                Analyse a text <ArrowRight className="size-4" aria-hidden="true" />
              </Link>
            </motion.div>
            <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} transition={spring.press}>
              <a href="#how" className="inline-flex h-12 items-center rounded-full border border-white/15 px-6 text-sm font-semibold text-white hover:border-white/30">
                How it works
              </a>
            </motion.div>
          </motion.div>
        </motion.div>

        <motion.div style={{ y: cardY }} initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.9, ease: ease.out, delay: 0.35 }}>
          <HeroCard />
        </motion.div>
      </div>
    </section>
  );
}

/** A real result: the ChatGPT example from the detector (style 98%, words 100%, verdict 99%). */
function HeroCard() {
  const reasons = [
    ["Average word length", "6.98", "humans ~4.57, AI ~4.78"],
    ["Unevenness of surprise", "2.31", "humans ~2.90, AI ~2.32"],
    ["Sentence length spread", "2.42", "humans ~9.59, AI ~6.91"],
  ];
  return (
    <div className="relative rounded-[1.75rem] bg-canvas p-3 text-ink shadow-[0_40px_120px_-30px_rgb(124_58_237/0.55)]">
      <div className="relative overflow-hidden rounded-[1.375rem] bg-gradient-to-br from-lilac to-lavender p-5">
        <Petals className="pointer-events-none absolute -top-10 -right-10 size-44 opacity-80" />
        <p className="relative text-xs font-semibold text-ink/60">Example · ChatGPT text</p>
        <div className="relative mt-1 flex items-end gap-3">
          <span className="num text-6xl font-extrabold text-ai">99%</span>
          <span className="mb-2 rounded-full bg-ink px-3 py-1 text-xs font-semibold text-white">Likely AI-generated</span>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3">
        {[["Style model", 0.98], ["Word model", 1]].map(([l, v]) => (
          <div key={String(l)} className="card p-4">
            <p className="text-xs text-ink/55">{l}</p>
            <p className="num mt-1 text-2xl font-bold">{Math.round(Number(v) * 100)}%</p>
            <div className="mt-3"><Bar value={Number(v)} delay={0.6} className="bg-violet" /></div>
          </div>
        ))}
      </div>
      <ul className="card mt-3 divide-y divide-line p-2">
        {reasons.map(([n, v, t], i) => (
          <motion.li key={n} initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.8 + i * 0.1, duration: 0.5, ease: ease.out }} className="flex items-center justify-between gap-3 px-3 py-2.5">
            <span className="min-w-0">
              <span className="block truncate text-sm font-semibold">{n} is {v}</span>
              <span className="block truncate text-xs text-ink/50">{t}</span>
            </span>
            <span className="shrink-0 rounded-full bg-ai/10 px-2 py-0.5 text-xs font-bold text-ai">→ AI</span>
          </motion.li>
        ))}
      </ul>
    </div>
  );
}

function Stat({ label, hint, value, format }: { label: string; hint: string; value?: number; format: (v: number) => string }) {
  return (
    <motion.div variants={pop} className="bg-ink px-2 py-8 sm:px-6">
      {value === undefined ? <Skeleton className="h-10 w-24" /> : <CountUp value={value} format={format} className="text-4xl font-extrabold md:text-5xl" />}
      <p className="mt-2 text-sm font-medium text-white/80">{label}</p>
      <p className="text-xs text-white/40">{hint}</p>
    </motion.div>
  );
}

const STEPS = [
  { icon: ScanText, title: "Paste a text", body: "50 to 400 words of English. It is analysed in memory and never stored." },
  { icon: Fingerprint, title: "Two models read it", body: "GPT-2 scores how predictable every word is. The style model turns that and 12 writing-style measures into a score; the word model weighs every word." },
  { icon: ShieldCheck, title: "You get the evidence", body: "A verdict with an unclear band, the top reasons compared with typical human and AI writing, a SHAP chart and coloured words." },
];

function HowItWorks() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 70%", "end 60%"] });
  const fill = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });
  return (
    <section id="how" className="mx-auto max-w-6xl px-4 py-24 sm:px-6 md:py-32">
      <div className="grid gap-12 md:grid-cols-[0.9fr_1.1fr]">
        <Reveal className="md:sticky md:top-28 md:self-start">
          <p className="eyebrow">How it works</p>
          <h2 className="mt-3 text-4xl font-bold tracking-tight text-balance md:text-5xl">Three steps, nothing hidden.</h2>
          <p className="mt-4 text-white/60">Every number on the result page can be traced back to the text you pasted.</p>
        </Reveal>
        <div ref={ref} className="relative pl-10">
          <div className="absolute top-2 bottom-2 left-[15px] w-px bg-white/10" aria-hidden="true" />
          <motion.div aria-hidden="true" style={{ scaleY: fill }} className="absolute top-2 bottom-2 left-[15px] w-px origin-top bg-flame" />
          <ol className="space-y-14">
            {STEPS.map((s, i) => (
              <motion.li key={s.title} initial={{ opacity: 0, y: 30 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, amount: 0.6 }} transition={{ duration: 0.6, ease: ease.out }} className="relative">
                <span className="absolute top-0 -left-10 grid size-8 place-items-center rounded-full bg-ink text-xs font-bold text-flame ring-1 ring-flame/50">{i + 1}</span>
                <s.icon className="size-6 text-lavender" aria-hidden="true" />
                <h3 className="mt-3 text-2xl font-bold tracking-tight">{s.title}</h3>
                <p className="mt-2 max-w-md leading-relaxed text-white/60">{s.body}</p>
              </motion.li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
}

function ModelCard({ icon: Icon, title, sub, points }: { icon: typeof Gauge; title: string; sub: string; points: string[] }) {
  return (
    <motion.article variants={pop} whileHover={lift} className="group rounded-[var(--radius-card)] border border-white/10 bg-white/[0.03] p-7 transition-colors hover:border-lavender/40">
      <div className="flex items-center justify-between">
        <span className="grid size-11 place-items-center rounded-2xl bg-lavender text-ink">
          <Icon className="size-5" aria-hidden="true" />
        </span>
        <ArrowRight className="size-5 -translate-x-1 text-white/30 opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100" aria-hidden="true" />
      </div>
      <h3 className="mt-6 text-2xl font-bold tracking-tight">{title}</h3>
      <p className="mt-1 text-sm text-white/50">{sub}</p>
      <ul className="mt-6 space-y-2.5">
        {points.map((p) => (
          <li key={p} className="flex gap-3 text-sm text-white/75">
            <span className="mt-2 size-1.5 shrink-0 rounded-full bg-flame" aria-hidden="true" />
            {p}
          </li>
        ))}
      </ul>
    </motion.article>
  );
}

function FprRow({ label, value, strong }: { label: string; value: number; strong?: boolean }) {
  return (
    <div>
      <div className="flex justify-between text-sm">
        <span className={strong ? "font-semibold text-white" : "text-white/60"}>{label}</span>
        <span className="num font-bold">{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="mt-2">
        <Bar value={value / 0.1} track="bg-white/10" className={strong ? "bg-human" : "bg-ai/80"} />
      </div>
    </div>
  );
}
