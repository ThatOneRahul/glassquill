import { motion } from "motion/react";
import { Brain, Database, Eraser, Scale, SplitSquareHorizontal, Sparkles } from "lucide-react";
import { Skeleton } from "../components/ui";
import { fmtFeature, useMeta } from "../lib/hooks";
import { ease, group, lift, pop } from "../lib/motion";
import { Seo } from "../lib/seo";

const PIPELINE = [
  { icon: Database, title: "Collect", body: "HC3, HAP-E and M4GT: human writing next to text from ChatGPT, GPT-4o, GPT-4, Llama-3, Mixtral, Gemma and Cohere." },
  { icon: Eraser, title: "Clean", body: "Remove artifacts that give the answer away: tokenizer leftovers in human text, markdown in AI text, link placeholders, curly quotes." },
  { icon: Scale, title: "Balance", body: "Equal human and AI counts in each of 17 domains, so topic never predicts the label. 50 to 400 words per text." },
  { icon: SplitSquareHorizontal, title: "Split by group", body: "70 / 15 / 15 by question or document, so the same topic is never in train and test. 5-fold GroupKFold for tuning." },
  { icon: Brain, title: "Train four models", body: "Perplexity threshold, TF-IDF + logistic regression, XGBoost with monotone constraints, and DistilBERT for comparison." },
  { icon: Sparkles, title: "Explain", body: "SHAP for XGBoost, exact word weights for the word model, LIME and SHAP agreement checks for DistilBERT." },
];

const DATA = [
  { name: "HC3", who: "ChatGPT (2022) vs human answers", where: "Reddit ELI5, finance, medicine, open QA, CS Wikipedia", licence: "CC BY-SA 4.0" },
  { name: "HAP-E", who: "GPT-4o, GPT-4o-mini, Llama-3 Instruct", where: "academic, news, fiction, spoken, blogs, scripts", licence: "MIT" },
  { name: "M4GT", who: "GPT-4o, GPT-4, GPT-3.5, Mixtral, Llama-3, Gemma, Cohere", where: "arXiv, essays, peer reviews, Reddit, WikiHow, Wikipedia", licence: "per source" },
  { name: "MAGE", who: "27 generators, test only", where: "14 domains, never used for training", licence: "Apache 2.0" },
];

const FEATURE_GROUPS: [string, string[]][] = [
  ["GPT-2 surprise", ["log_ppl", "nll_std", "nll_max", "top1_frac", "top10_frac", "mean_entropy"]],
  ["Rhythm", ["avg_sent_len", "std_sent_len", "burstiness"]],
  ["Vocabulary", ["type_token_ratio", "avg_word_len", "function_word_rate"]],
  ["Punctuation and voice", ["comma_rate", "punct_rate", "first_person_rate", "contraction_rate"]],
  ["Readability", ["flesch_reading_ease", "fk_grade"]],
];

const LIMITS = [
  "A score is never proof. Use it as a second opinion and read the reasons.",
  "Plain, predictable writing, common for non-native writers, can look AI-like.",
  "Paraphrasing tools and heavy editing lower detection, and small open models often go unnoticed.",
  "English only, 50 to 400 words. Texts that mix human and AI writing are not handled.",
  "New language models appear quickly; results can go stale.",
];

export default function Method() {
  const { meta } = useMeta();
  return (
    <>
      <Seo title="Method" description="How GlassQuill was built: datasets, cleaning, leak-free splits, the 18 features, four models, explanations, and the limits of AI-text detection." />
      <motion.div initial="hidden" animate="show" variants={group(0.07)} className="space-y-4">
        <motion.header variants={pop} className="pt-2">
          <p className="eyebrow">Methodology</p>
          <h1 className="mt-1 text-3xl font-bold tracking-tight sm:text-4xl">How GlassQuill works</h1>
          <p className="mt-1 max-w-2xl text-sm text-ink/60 dark:text-white/55">
            From raw datasets to a verdict with reasons. Every step is reproducible from the project's pipeline script (<code className="font-mono text-xs">run_pipeline.py</code>).
          </p>
        </motion.header>

        {/* pipeline */}
        <motion.section variants={pop} className="card p-5 sm:p-8" aria-labelledby="pipe-h">
          <h2 id="pipe-h" className="font-semibold">The pipeline</h2>
          <ol className="relative mt-6 grid gap-6 md:grid-cols-3">
            {PIPELINE.map((s, i) => (
              <motion.li
                key={s.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.4 }}
                transition={{ duration: 0.5, ease: ease.out, delay: (i % 3) * 0.08 }}
                className="relative rounded-2xl bg-canvas/70 p-5 dark:bg-white/[0.04]"
              >
                <div className="flex items-center gap-3">
                  <span className="num text-xs font-bold text-flame">0{i + 1}</span>
                  <s.icon className="size-5 text-violet dark:text-lavender" aria-hidden="true" />
                </div>
                <h3 className="mt-3 font-semibold">{s.title}</h3>
                <p className="mt-1 text-sm leading-relaxed text-ink/60 dark:text-white/55">{s.body}</p>
              </motion.li>
            ))}
          </ol>
        </motion.section>

        {/* data */}
        <section id="data" aria-labelledby="data-h" className="scroll-mt-24">
          <h2 id="data-h" className="sr-only">Datasets</h2>
          <motion.div initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }} variants={group(0.07)} className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {DATA.map((d) => (
              <motion.article key={d.name} variants={pop} whileHover={lift} className="card p-5">
                <p className="text-xs font-semibold text-ink/45 dark:text-white/40">Dataset</p>
                <h3 className="text-xl font-bold">{d.name}</h3>
                <p className="mt-3 text-sm font-medium">{d.who}</p>
                <p className="mt-1 text-sm text-ink/55 dark:text-white/50">{d.where}</p>
                <p className="mt-4 inline-block rounded-full bg-lavender-soft px-2.5 py-0.5 text-[11px] font-semibold text-violet dark:bg-violet/15 dark:text-lavender">{d.licence}</p>
              </motion.article>
            ))}
          </motion.div>
        </section>

        {/* features */}
        <motion.section id="features" variants={pop} className="card scroll-mt-24 p-5 sm:p-8" aria-labelledby="feat-h">
          <h2 id="feat-h" className="font-semibold">The 18 features the style model reads</h2>
          <p className="text-xs text-ink/50 dark:text-white/45">Median value in the training set for human and AI texts</p>
          <div className="mt-6 grid gap-6 md:grid-cols-2 xl:grid-cols-3">
            {FEATURE_GROUPS.map(([g, feats]) => (
              <div key={g}>
                <h3 className="text-xs font-bold tracking-wider text-flame uppercase">{g}</h3>
                <ul className="mt-2 divide-y divide-line dark:divide-white/10">
                  {feats.map((f) => {
                    const t = meta?.typical[f];
                    return (
                      <li key={f} className="flex items-center justify-between gap-3 py-2 text-sm">
                        <code className="font-mono text-xs">{f}</code>
                        {t ? (
                          <span className="num text-xs text-ink/60 dark:text-white/55">
                            <span className="text-human">{fmtFeature(f, t.human)}</span> · <span className="text-ai">{fmtFeature(f, t.ai)}</span>
                          </span>
                        ) : (
                          <Skeleton className="h-4 w-20" />
                        )}
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </div>
          <p className="mt-4 text-xs text-ink/50 dark:text-white/45"><span className="text-human">green</span> = typical human · <span className="text-ai">rose</span> = typical AI</p>
        </motion.section>

        {/* limits */}
        <motion.section id="limits" variants={pop} className="scroll-mt-24 overflow-hidden rounded-[var(--radius-card)] bg-ink p-6 text-white sm:p-8 dark:ring-1 dark:ring-white/10" aria-labelledby="lim-h">
          <p className="eyebrow">Limits and ethics</p>
          <h2 id="lim-h" className="mt-2 text-2xl font-bold tracking-tight">What GlassQuill cannot do</h2>
          <motion.ul initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.3 }} variants={group(0.06)} className="mt-5 grid gap-3 md:grid-cols-2">
            {LIMITS.map((l) => (
              <motion.li key={l} variants={pop} className="flex gap-3 rounded-2xl bg-white/[0.04] p-4 text-sm leading-relaxed text-white/75">
                <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-flame" aria-hidden="true" />
                {l}
              </motion.li>
            ))}
          </motion.ul>
        </motion.section>
      </motion.div>
    </>
  );
}
