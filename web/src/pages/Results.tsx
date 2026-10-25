import { AnimatePresence, motion, useInView } from "motion/react";
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Crosshair, Gauge, ShieldCheck, Target } from "lucide-react";
import { useRef, useState } from "react";
import { Bar, CountUp, PageHeader, Segmented, Skeleton } from "../components/ui";
import { APP_MODEL, type Row } from "../lib/api";
import { useMeta } from "../lib/hooks";
import { ease, group, lift, pop } from "../lib/motion";
import { Seo } from "../lib/seo";

type SetKey = "in_domain" | "out_of_domain" | "paraphrase";
const SETS: { value: SetKey; label: string; about: string }[] = [
  { value: "in_domain", label: "Test set", about: "4,829 unseen texts from HC3, HAP-E and M4GT, the same 17 domains as training." },
  { value: "out_of_domain", label: "MAGE", about: "2,000 texts from 27 generators and 14 domains the models never trained on." },
  { value: "paraphrase", label: "Paraphrased", about: "60 AI texts from the test set, every sentence rewritten by a T5 paraphraser." },
];
const pct1 = (v: number) => `${v.toFixed(1)}%`;
const fixed3 = (v: number) => v.toFixed(3);
const whole = (v: number) => v.toFixed(0);
const label = (m: string | number) => (m === APP_MODEL ? "GlassQuill verdict" : String(m));

export default function Results() {
  const { meta, error } = useMeta();
  const [set, setSet] = useState<SetKey>("in_domain");
  const rows = meta?.results[set] ?? [];
  const app = rows.find((r) => r.model === APP_MODEL);

  return (
    <>
      <Seo title="Results" description="Accuracy, F1, ROC-AUC and false positive rates of GlassQuill's five models on the test set, the MAGE benchmark and paraphrased AI text." />
      <div className="mx-auto max-w-6xl px-4 pb-24 sm:px-6">
      <PageHeader
        eyebrow="Evaluation"
        title="Measured,"
        accent="not claimed."
        aside={<Segmented label="Test condition" value={set} onChange={setSet} options={SETS} />}
      >
        <AnimatePresence mode="wait" initial={false}>
          <motion.p key={set} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: 0.2 }}>
            {SETS.find((s) => s.value === set)?.about}
          </motion.p>
        </AnimatePresence>
      </PageHeader>
      <motion.div initial="hidden" animate="show" variants={group(0.07, 0.25)} className="space-y-4">

        {error && (
          <motion.p variants={pop} role="alert" className="card flex items-center gap-2 p-4 text-sm text-ai">
            <AlertTriangle className="size-4" /> {error}
          </motion.p>
        )}

        {/* ---- summary cards ---- */}
        <AnimatePresence mode="wait">
          <motion.div key={set} initial="hidden" animate="show" exit={{ opacity: 0 }} variants={group(0.06)} className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            {!meta ? (
              Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-32 !rounded-[var(--radius-card)]" />)
            ) : set === "paraphrase" && app ? (
              <>
                <StatCard icon={Target} title="Caught before" value={Number(app.recall_before) * 100} format={pct1} />
                <StatCard icon={Crosshair} title="Caught after paraphrasing" value={Number(app.recall_after) * 100} format={pct1} />
                <StatCard icon={ArrowDownRight} title="Drop" value={Number(app.drop) * 100} format={pct1} down />
                <StatCard icon={ShieldCheck} title="Texts tested" value={60} format={whole} />
              </>
            ) : app ? (
              <>
                <StatCard icon={Target} title="Accuracy" value={Number(app.accuracy) * 100} format={pct1} />
                <StatCard icon={Gauge} title="F1 score" value={Number(app.f1)} format={fixed3} />
                <StatCard icon={Crosshair} title="ROC-AUC" value={Number(app.roc_auc)} format={fixed3} />
                <StatCard icon={ShieldCheck} title="Humans wrongly flagged" value={Number(app.fpr) * 100} format={pct1} down />
              </>
            ) : null}
          </motion.div>
        </AnimatePresence>

        {/* ---- model comparison ---- */}
        <motion.section variants={pop} className="card p-5 sm:p-6" aria-labelledby="cmp-h">
          <h2 id="cmp-h" className="font-semibold">Model comparison</h2>
          <p className="text-xs text-white/55">GlassQuill's verdict is the average of XGBoost and TF-IDF + LR. Rule chosen on the validation set.</p>
          <div className="mt-4 overflow-x-auto">
            {meta ? (
              <AnimatePresence mode="wait">
                <motion.table key={set} initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} transition={{ duration: 0.2 }} className="w-full min-w-[640px] text-sm">
                  <thead>
                    <tr className="text-left text-xs text-white/55">
                      <th scope="col" className="py-2 pr-4 font-medium">Model</th>
                      {set === "paraphrase" ? (
                        <>
                          <th scope="col" className="w-2/5 py-2 pr-4 font-medium">Recall before → after</th>
                          <th scope="col" className="py-2 text-right font-medium">Drop</th>
                        </>
                      ) : (
                        <>
                          <th scope="col" className="w-1/3 py-2 pr-4 font-medium">Accuracy</th>
                          <th scope="col" className="py-2 pr-4 text-right font-medium">Precision</th>
                          <th scope="col" className="py-2 pr-4 text-right font-medium">Recall</th>
                          <th scope="col" className="py-2 pr-4 text-right font-medium">F1</th>
                          <th scope="col" className="py-2 pr-4 text-right font-medium">ROC-AUC</th>
                          <th scope="col" className="py-2 text-right font-medium">FPR</th>
                        </>
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <ModelRow key={String(r.model)} r={r} i={i} paraphrase={set === "paraphrase"} best={best(rows)} />
                    ))}
                  </tbody>
                </motion.table>
              </AnimatePresence>
            ) : (
              <div className="space-y-3">{Array.from({ length: 5 }, (_, i) => <Skeleton key={i} className="h-9" />)}</div>
            )}
          </div>
        </motion.section>

        <div className="grid grid-cols-1 gap-4 xl:grid-cols-12">
          <CorpusBars rows={meta?.results.by_corpus} />
          <GeneratorBars rows={meta?.results.ood_by_generator} />
        </div>
        <ShapGlobal rows={meta?.results.shap_global} />
      </motion.div>
      </div>
    </>
  );
}

const best = (rows: Row[]) => ({
  accuracy: Math.max(...rows.map((r) => Number(r.accuracy ?? 0))),
  fpr: Math.min(...rows.map((r) => Number(r.fpr ?? 1))),
});

function StatCard({ icon: Icon, title, value, format, down }: { icon: typeof Target; title: string; value: number; format: (v: number) => string; down?: boolean }) {
  return (
    <motion.div variants={pop} whileHover={lift} className="card p-5">
      <div className="flex items-start justify-between">
        <p className="text-sm text-white/55">{title}</p>
        <span className="grid size-9 place-items-center rounded-xl bg-violet/15 text-lavender">
          <Icon className="size-4" aria-hidden="true" />
        </span>
      </div>
      <CountUp value={value} format={format} className="mt-2 block text-3xl font-extrabold sm:text-4xl" />
      <p className={`mt-1 flex items-center gap-1 text-xs font-medium ${down ? "text-human" : "text-lavender"}`}>
        {down ? <ArrowDownRight className="size-3.5" /> : <ArrowUpRight className="size-3.5" />} GlassQuill verdict
      </p>
    </motion.div>
  );
}

function ModelRow({ r, i, paraphrase, best: b }: { r: Row; i: number; paraphrase: boolean; best: { accuracy: number; fpr: number } }) {
  const app = r.model === APP_MODEL;
  const n = (k: string) => Number(r[k]);
  const cell = "num py-3 pr-4 text-right";
  return (
    <motion.tr initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.05, duration: 0.35, ease: ease.out }} className={`border-t border-white/10 ${app ? "bg-violet/10" : ""}`}>
      <th scope="row" className={`py-3 pr-4 text-left ${app ? "font-semibold" : "font-medium"}`}>
        {label(r.model)}
        {app && <span className="ml-2 rounded-full bg-violet px-2 py-0.5 text-[10px] font-bold text-white">APP</span>}
      </th>
      {paraphrase ? (
        <>
          <td className="py-3 pr-4">
            <div className="flex items-center gap-3">
              <div className="flex-1"><Bar value={n("recall_after")} className={app ? "bg-violet" : "bg-white/35"} delay={i * 0.05} /></div>
              <span className="num w-28 text-right text-xs">{(n("recall_before") * 100).toFixed(0)}% → <b>{(n("recall_after") * 100).toFixed(0)}%</b></span>
            </div>
          </td>
          <td className={`${cell} !pr-0 ${n("drop") > 0.1 ? "text-ai" : ""}`}>−{(n("drop") * 100).toFixed(1)} pts</td>
        </>
      ) : (
        <>
          <td className="py-3 pr-4">
            <div className="flex items-center gap-3">
              <div className="flex-1"><Bar value={n("accuracy")} className={app ? "bg-violet" : "bg-white/35"} delay={i * 0.05} /></div>
              <span className={`num w-14 text-right ${n("accuracy") === b.accuracy ? "font-bold" : ""}`}>{(n("accuracy") * 100).toFixed(1)}%</span>
            </div>
          </td>
          <td className={cell}>{n("precision").toFixed(3)}</td>
          <td className={cell}>{n("recall").toFixed(3)}</td>
          <td className={cell}>{n("f1").toFixed(3)}</td>
          <td className={cell}>{n("roc_auc").toFixed(3)}</td>
          <td className={`${cell} !pr-0 ${n("fpr") === b.fpr ? "font-bold text-human" : ""}`}>{(n("fpr") * 100).toFixed(1)}%</td>
        </>
      )}
    </motion.tr>
  );
}

/** Vertical lavender bars, like the reference's "Impact" card. */
function CorpusBars({ rows }: { rows?: Row[] }) {
  const corpora = [
    ["hc3", "HC3", "ChatGPT 2022"],
    ["hape", "HAP-E", "GPT-4o, Llama-3"],
    ["m4gt", "M4GT", "GPT-4o, Mixtral…"],
  ];
  const f1 = (c: string, m: string) => Number(rows?.find((r) => r.corpus === c && r.model === m)?.f1 ?? 0);
  const chart = useRef<HTMLDivElement>(null);
  const seen = useInView(chart, { once: true, amount: 0.4 });   // watch the chart, not the 0-height bars
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-5" aria-labelledby="corpus-h">
      <h2 id="corpus-h" className="font-semibold">F1 by corpus</h2>
      <p className="text-xs text-white/55">GlassQuill verdict vs XGBoost alone, test set</p>
      {rows ? (
        <div ref={chart} className="mt-6 grid h-56 grid-cols-3 items-end gap-4" role="img" aria-label={corpora.map(([c, n]) => `${n}: verdict F1 ${f1(c, APP_MODEL).toFixed(3)}, XGBoost ${f1(c, "XGBoost").toFixed(3)}`).join("; ")}>
          {corpora.map(([c, name, sub], i) => (
            <div key={c} className="flex h-full flex-col justify-end">
              <div className="flex h-full items-end justify-center gap-1.5">
                {[["XGBoost", "bg-lavender/30"], [APP_MODEL, "bg-gradient-to-t from-violet to-lavender"]].map(([m, cls], j) => (
                  <motion.div
                    key={m}
                    className={`w-7 origin-bottom rounded-full sm:w-9 ${cls}`}
                    style={{ height: `${Math.max(0, (f1(c, m) - 0.5) / 0.5) * 100}%` }}
                    initial={{ scaleY: 0 }}
                    animate={{ scaleY: seen ? 1 : 0 }}
                    transition={{ duration: 0.8, ease: ease.out, delay: 0.1 + i * 0.1 + j * 0.05 }}
                  />
                ))}
              </div>
              <p className="mt-3 text-center text-sm font-semibold">{name}</p>
              <p className="text-center text-[11px] text-white/55">{sub}</p>
              <p className="num text-center text-xs font-bold text-lavender">{f1(c, APP_MODEL).toFixed(3)}</p>
            </div>
          ))}
        </div>
      ) : (
        <Skeleton className="mt-6 h-56" />
      )}
      <div className="mt-4 flex gap-4 text-xs text-white/50">
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-lavender/30" /> XGBoost</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-violet" /> GlassQuill verdict</span>
        <span className="ml-auto">axis 0.5 – 1.0</span>
      </div>
    </motion.section>
  );
}

function GeneratorBars({ rows }: { rows?: Row[] }) {
  const sorted = rows ? [...rows].sort((a, b) => Number(b[APP_MODEL]) - Number(a[APP_MODEL])) : [];
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6 xl:col-span-7" aria-labelledby="gen-h">
      <h2 id="gen-h" className="font-semibold">Who fools the detector? (MAGE)</h2>
      <p className="text-xs text-white/55">Share of each generator's texts the GlassQuill verdict labels correctly</p>
      <ul className="mt-5 space-y-2.5">
        {rows
          ? sorted.map((r, i) => {
              const v = Number(r[APP_MODEL]);
              const human = r.generator === "human";
              return (
                <li key={String(r.generator)} className="grid grid-cols-[8.5rem_1fr_3rem] items-center gap-3">
                  <span className="truncate text-xs font-medium">{human ? "Human texts" : String(r.generator)} <span className="text-white/55">· {Number(r.n)}</span></span>
                  <Bar value={v} delay={i * 0.04} className={human ? "bg-human" : v < 0.3 ? "bg-ai/70" : "bg-violet"} />
                  <span className="num text-right text-xs font-semibold">{(v * 100).toFixed(0)}%</span>
                </li>
              );
            })
          : Array.from({ length: 8 }, (_, i) => <Skeleton key={i} className="h-5" />)}
      </ul>
      <p className="mt-4 text-xs leading-relaxed text-white/50">
        Chat models (GPT-4, GPT-3.5) are caught most of the time. Small base models such as OPT, BLOOM and GPT-J
        sample unusual words, so their text looks human to every model we tested.
      </p>
    </motion.section>
  );
}

function ShapGlobal({ rows }: { rows?: Row[] }) {
  const top = rows?.slice(0, 10) ?? [];
  const max = Math.max(...top.map((r) => Number(r.mean_abs_shap)), 1e-6);
  return (
    <motion.section variants={pop} className="card p-5 sm:p-6" aria-labelledby="shapg-h">
      <h2 id="shapg-h" className="font-semibold">What the style model relies on</h2>
      <p className="text-xs text-white/55">Mean |SHAP| over the test set, and which way a higher value pushes</p>
      <motion.ul initial="hidden" whileInView="show" viewport={{ once: true, amount: 0.2 }} variants={group(0.04)} className="mt-5 grid gap-x-10 gap-y-3 md:grid-cols-2">
        {rows
          ? top.map((r, i) => (
              <motion.li key={String(r.feature)} variants={pop} className="grid grid-cols-[1fr_auto] items-center gap-x-3 gap-y-1.5">
                <span className="truncate text-sm font-medium">{String(r.name)}</span>
                <span className={`text-xs font-semibold ${r.higher_value_pushes_towards === "AI" ? "text-ai" : "text-human"}`}>
                  higher → {String(r.higher_value_pushes_towards)}
                </span>
                <div className="col-span-2"><Bar value={Number(r.mean_abs_shap) / max} delay={i * 0.03} className="bg-gradient-to-r from-lavender to-violet" /></div>
              </motion.li>
            ))
          : Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-8" />)}
      </motion.ul>
    </motion.section>
  );
}
