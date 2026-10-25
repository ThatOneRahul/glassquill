import {
  animate,
  AnimatePresence,
  motion,
  useInView,
  useReducedMotion,
  type HTMLMotionProps,
} from "motion/react";
import { Check, LoaderCircle } from "lucide-react";
import { useEffect, useId, useRef, useState, type ReactNode } from "react";
import { duration, ease, group, inView, press, rise, spring } from "../lib/motion";

/* ---------------- brand ---------------- */

export function LogoMark({ className = "size-8" }: { className?: string }) {
  const id = useId();
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true">
      <defs>
        <linearGradient id={id} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#ddd3ff" />
          <stop offset="1" stopColor="#7c3aed" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="16" fill="#07070a" />
      <path d="M32 9 46 31 32 55 18 31Z" fill={`url(#${id})`} />
      <path d="M32 31v20" stroke="#07070a" strokeWidth="3" strokeLinecap="round" />
      <circle cx="32" cy="29" r="4" fill="#07070a" />
      <circle cx="47" cy="50" r="4.5" fill="#ff6a1f" />
    </svg>
  );
}

export function Logo({ dark = true }: { dark?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark className="size-8" />
      <span className={`text-lg font-bold tracking-tight ${dark ? "text-white" : "text-ink dark:text-white"}`}>
        Glass<span className="text-lavender">Quill</span>
      </span>
    </span>
  );
}

/* ---------------- page header (the homepage hero, for inner pages) ---------------- */

export function PageHeader({ eyebrow, title, accent, children, aside }: {
  eyebrow: string;
  title: string;
  accent: string;
  children?: ReactNode;
  aside?: ReactNode;
}) {
  return (
    <motion.header
      initial="hidden"
      animate="show"
      variants={group(0.09, 0.05)}
      className="relative flex flex-wrap items-end justify-between gap-8 pt-32 pb-10 md:pt-40 md:pb-14"
    >
      <div
        aria-hidden="true"
        className="pointer-events-none absolute top-0 left-1/2 -z-0 h-[420px] w-[min(820px,100%)] -translate-x-1/2 rounded-full bg-violet/20 blur-[120px]"
      />
      <div className="relative max-w-3xl">
        <motion.p variants={rise} className="eyebrow">{eyebrow}</motion.p>
        <h1 className="mt-4 text-[clamp(2.6rem,7vw,5rem)] leading-[0.95] font-extrabold tracking-[-0.04em]">
          <motion.span variants={rise} className="block">{title}</motion.span>{" "}
          <motion.span variants={rise} className="block text-flame">{accent}</motion.span>
        </h1>
        {children && (
          <motion.div variants={rise} className="mt-5 max-w-2xl text-lg leading-relaxed text-white/60">
            {children}
          </motion.div>
        )}
      </div>
      {aside && <motion.div variants={rise} className="relative">{aside}</motion.div>}
    </motion.header>
  );
}

/* ---------------- buttons ---------------- */

type Tone = "black" | "violet" | "light" | "ghost" | "outline";
const tones: Record<Tone, string> = {
  // on dark pages the primary button is white; its hover must stay light, or dark text
  // sits on a dark fill and the label disappears
  black: "bg-ink text-white hover:bg-ink-2 dark:bg-white dark:text-ink dark:hover:bg-lavender-soft",
  violet: "bg-violet text-white hover:bg-violet-deep",
  light: "bg-white text-ink hover:bg-lavender-soft",
  ghost: "text-current hover:bg-black/5 dark:hover:bg-white/10",
  outline: "border border-current/20 text-current hover:border-current/40",
};

interface ButtonProps extends Omit<HTMLMotionProps<"button">, "children"> {
  tone?: Tone;
  icon?: ReactNode;
  loading?: boolean;
  done?: boolean;
  children: ReactNode;
}

/** Pill button with spring press, plus loading and success states that cross-fade. */
export function Button({ tone = "black", icon, loading, done, children, className = "", disabled, ...rest }: ButtonProps) {
  const state = loading ? "loading" : done ? "done" : "idle";
  return (
    <motion.button
      whileHover={disabled || loading ? undefined : { y: -1 }}
      whileTap={disabled || loading ? undefined : press}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={`relative inline-flex h-11 items-center justify-center gap-2 overflow-hidden rounded-full px-5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${tones[tone]} ${className}`}
      {...rest}
    >
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={state}
          className="inline-flex items-center gap-2"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: duration.micro * 1.5, ease: ease.out }}
        >
          {state === "loading" ? (
            <LoaderCircle className="size-4 animate-spin" aria-hidden="true" />
          ) : state === "done" ? (
            <Check className="size-4" aria-hidden="true" />
          ) : (
            icon
          )}
          {children}
        </motion.span>
      </AnimatePresence>
    </motion.button>
  );
}

/* ---------------- scroll reveals ---------------- */

/** Fades and rises once when scrolled into view. */
export function Reveal({ children, className, delay = 0 }: { children: ReactNode; className?: string; delay?: number }) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={inView}
      variants={rise}
      transition={{ delay }}
    >
      {children}
    </motion.div>
  );
}

/** Staggers the reveal of its motion children (give them `variants`). */
export function RevealGroup({
  children,
  className,
  gap,
  as = "div",
}: {
  children: ReactNode;
  className?: string;
  gap?: number;
  as?: "div" | "ul" | "ol" | "section";
}) {
  const Tag = motion[as];
  return (
    <Tag className={className} initial="hidden" whileInView="show" viewport={inView} variants={group(gap)}>
      {children}
    </Tag>
  );
}

/* ---------------- numbers and bars ---------------- */

/** Counts up from 0 when scrolled into view. `format` turns the number into text. */
export function CountUp({ value, format = (v) => v.toFixed(0), className }: {
  value: number;
  format?: (v: number) => string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const seen = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion();
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!seen || reduce) {
      el.textContent = format(seen || reduce ? value : 0);
      return;
    }
    const controls = animate(0, value, {
      duration: 1.2,
      ease: ease.out,
      onUpdate: (v) => (el.textContent = format(v)),
    });
    return () => controls.stop();
  }, [seen, value, format, reduce]);
  return (
    <span ref={ref} className={`num ${className ?? ""}`}>
      {format(0)}
    </span>
  );
}

/**
 * Horizontal bar that grows from the left (scaleX, no layout work).
 * Visibility is watched on the track: the bar itself starts at scaleX(0), i.e. zero width,
 * and a zero-area element is not reliably reported as intersecting.
 */
export function Bar({ value, className = "bg-violet", track = "bg-black/5 dark:bg-white/10", delay = 0, height = "h-2" }: {
  value: number;
  className?: string;
  track?: string;
  delay?: number;
  height?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const seen = useInView(ref, { once: true, amount: 0.5 });
  return (
    <div ref={ref} className={`${height} w-full overflow-hidden rounded-full ${track}`}>
      <motion.div
        className={`h-full origin-left rounded-full ${className}`}
        initial={{ scaleX: 0 }}
        animate={{ scaleX: seen ? Math.max(0, Math.min(1, value)) : 0 }}
        transition={{ duration: 0.9, ease: ease.out, delay }}
      />
    </div>
  );
}

/* ---------------- segmented control (Weekly / Monthly in the reference) ---------------- */

export function Segmented<T extends string>({ options, value, onChange, label }: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  const id = useId();
  return (
    // a toggle group (aria-pressed), not ARIA tabs: there are no tab panels to point at
    <div role="group" aria-label={label} className="inline-flex rounded-full bg-black/5 p-1 dark:bg-white/10">
      {options.map((o) => {
        const active = o.value === value;
        return (
          <button
            key={o.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(o.value)}
            className={`relative rounded-full px-3.5 py-1.5 text-xs font-semibold transition-colors ${
              active ? "text-white dark:text-ink" : "text-black/55 hover:text-black dark:text-white/70 dark:hover:text-white"
            }`}
          >
            {active && (
              <motion.span
                layoutId={`seg-${id}`}
                className="absolute inset-0 rounded-full bg-ink dark:bg-white"
                transition={spring.layout}
              />
            )}
            <span className="relative">{o.label}</span>
          </button>
        );
      })}
    </div>
  );
}

/* ---------------- toast ---------------- */

export function Toast({ message, tone = "error", onClose }: { message: string | null; tone?: "error" | "ok"; onClose: () => void }) {
  // errors stay long enough to read, and never time out while hovered or focused
  const [held, setHeld] = useState(false);
  useEffect(() => {
    if (!message || held) return;
    const t = setTimeout(onClose, tone === "error" ? 8000 : 4000);
    return () => clearTimeout(t);
  }, [message, onClose, held, tone]);
  return (
    <div className="pointer-events-none fixed inset-x-0 bottom-5 z-50 flex justify-center px-4" aria-live="polite">
      <AnimatePresence>
        {message && (
          <motion.div
            key={message}
            role={tone === "error" ? "alert" : "status"}
            onMouseEnter={() => setHeld(true)}
            onMouseLeave={() => setHeld(false)}
            onFocus={() => setHeld(true)}
            onBlur={() => setHeld(false)}
            initial={{ opacity: 0, y: 24, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 12, scale: 0.98 }}
            transition={spring.lift}
            className="pointer-events-auto flex max-w-md items-center gap-3 rounded-2xl bg-ink px-4 py-3 text-sm text-white shadow-2xl ring-1 ring-white/10"
          >
            <span className={`size-2 shrink-0 rounded-full ${tone === "error" ? "bg-ai" : "bg-human"}`} aria-hidden="true" />
            <span>{message}</span>
            <button onClick={onClose} className="ml-2 text-white/60 hover:text-white" aria-label="Dismiss">
              ×
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ---------------- decoration ---------------- */

/** Abstract lavender petals, in place of the reference's 3D flower. Purely decorative. */
export function Petals({ className = "" }: { className?: string }) {
  const id = useId();
  const reduce = useReducedMotion();
  return (
    <motion.svg
      viewBox="0 0 200 200"
      className={className}
      aria-hidden="true"
      initial={{ rotate: -20, opacity: 0, scale: 0.85 }}
      animate={{ rotate: 0, opacity: 1, scale: 1 }}
      transition={{ duration: reduce ? 0 : 1.4, ease: ease.out }}
    >
      <defs>
        <radialGradient id={id} cx="0.5" cy="0.5" r="0.6">
          <stop offset="0" stopColor="#f5f0ff" />
          <stop offset="0.55" stopColor="#c4b5fd" />
          <stop offset="1" stopColor="#8b5cf6" />
        </radialGradient>
      </defs>
      {Array.from({ length: 10 }, (_, i) => (
        <ellipse
          key={i}
          cx="100"
          cy="58"
          rx="20"
          ry="46"
          fill={`url(#${id})`}
          opacity={0.55 + (i % 3) * 0.15}
          transform={`rotate(${i * 36} 100 100)`}
        />
      ))}
      <circle cx="100" cy="100" r="16" fill="#f5f0ff" />
    </motion.svg>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`skeleton ${className}`} aria-hidden="true" />;
}
