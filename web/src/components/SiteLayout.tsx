import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "motion/react";
import { ArrowRight, Menu, X } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Link, NavLink, useLocation } from "react-router-dom";
import { ease, spring } from "../lib/motion";
import { Logo } from "./ui";

const HIDE_CTA = ["/detect", "/results"];

const NAV = [
  { to: "/detect", label: "Detector" },
  { to: "/results", label: "Results" },
  { to: "/method", label: "Method" },
];

/** Marketing layout: cinematic black, sticky header, footer. */
export function SiteLayout({ children }: { children: ReactNode }) {
  const { scrollY } = useScroll();
  const [scrolled, setScrolled] = useState(false);
  const { pathname } = useLocation();
  // the menu remembers the page it was opened on, so navigating anywhere closes it
  const [openOn, setOpenOn] = useState<string | null>(null);
  const open = openOn === pathname;
  const hideCta = HIDE_CTA.includes(pathname);
  const setOpen = (v: boolean | ((o: boolean) => boolean)) =>
    setOpenOn((typeof v === "function" ? v(open) : v) ? pathname : null);
  useMotionValueEvent(scrollY, "change", (y) => setScrolled(y > 12));

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpenOn(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <div className="min-h-dvh bg-ink text-white">
      <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[60] focus:rounded-full focus:bg-white focus:px-4 focus:py-2 focus:text-ink">
        Skip to content
      </a>
      <header
        className={`fixed inset-x-0 top-0 z-40 transition-[background-color,border-color] duration-300 ${
          scrolled || open ? "border-b border-white/10 bg-ink/80 backdrop-blur-xl" : "border-b border-transparent"
        }`}
      >
        <nav aria-label="Main" className="mx-auto flex h-16 max-w-6xl items-center justify-between px-4 sm:px-6">
          <Link to="/" aria-label="GlassQuill home">
            <Logo />
          </Link>
          <ul className="hidden items-center gap-1 md:flex">
            {NAV.map((n) => {
              const active = pathname === n.to;
              return (
                <li key={n.to}>
                  <NavLink
                    to={n.to}
                    aria-current={active ? "page" : undefined}
                    className={`group relative block px-3.5 py-2 text-sm transition-colors ${active ? "text-white" : "text-white/70 hover:text-white"}`}
                  >
                    {n.label}
                    {active ? (
                      <motion.span layoutId="nav-active" transition={spring.layout} className="absolute inset-x-3.5 -bottom-0.5 h-0.5 rounded-full bg-flame" />
                    ) : (
                      <span className="absolute inset-x-3.5 -bottom-0.5 h-px origin-left scale-x-0 bg-white/40 transition-transform duration-300 group-hover:scale-x-100" />
                    )}
                  </NavLink>
                </li>
              );
            })}
          </ul>
          <div className="flex items-center gap-2">
            {/* hidden where the detector is already the point (or one click away). Hiding is a
                class + `inert`, so it takes effect at once even if animations are paused; the
                fade is only CSS, and the slot keeps its width so the nav doesn't shift. */}
            <div
              inert={hideCta}
              aria-hidden={hideCta || undefined}
              className={`hidden transition-[opacity,scale] duration-200 ease-out sm:block ${hideCta ? "pointer-events-none scale-95 opacity-0" : ""}`}
            >
              <Link
                to="/detect"
                className="inline-flex h-10 items-center gap-2 rounded-full bg-white px-4 text-sm font-semibold text-ink transition-[translate,scale] hover:-translate-y-px active:scale-[0.97]"
              >
                Try the detector <ArrowRight className="size-4" aria-hidden="true" />
              </Link>
            </div>
            <button
              className="grid size-10 place-items-center rounded-full text-white md:hidden"
              aria-label={open ? "Close menu" : "Open menu"}
              aria-expanded={open}
              aria-controls="mobile-menu"
              onClick={() => setOpen((o) => !o)}
            >
              <AnimatePresence mode="wait" initial={false}>
                <motion.span
                  key={open ? "x" : "menu"}
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.15 }}
                >
                  {open ? <X className="size-5" /> : <Menu className="size-5" />}
                </motion.span>
              </AnimatePresence>
            </button>
          </div>
        </nav>
        <AnimatePresence>
          {open && (
            <motion.div
              id="mobile-menu"
              className="overflow-hidden border-t border-white/10 md:hidden"
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25, ease: ease.out }}
            >
              <motion.ul
                className="space-y-1 px-4 py-4"
                initial="hidden"
                animate="show"
                variants={{ hidden: {}, show: { transition: { staggerChildren: 0.05 } } }}
              >
                {[{ to: "/", label: "Home" }, ...NAV].map((n) => (
                  <motion.li key={n.to} variants={{ hidden: { opacity: 0, x: -10 }, show: { opacity: 1, x: 0 } }}>
                    <NavLink
                      to={n.to}
                      aria-current={pathname === n.to ? "page" : undefined}
                      className={`flex items-center justify-between rounded-xl px-3 py-3 text-base font-medium hover:bg-white/5 ${pathname === n.to ? "text-white" : "text-white/70"}`}
                    >
                      {n.label}
                      {pathname === n.to && <span className="size-1.5 rounded-full bg-flame" aria-hidden="true" />}
                    </NavLink>
                  </motion.li>
                ))}
              </motion.ul>
            </motion.div>
          )}
        </AnimatePresence>
      </header>
      <main id="main">{children}</main>
      <Footer />
    </div>
  );
}

function Footer() {
  return (
    <footer className="border-t border-white/10">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-14 sm:px-6 md:grid-cols-[1.4fr_1fr_1fr]">
        <div>
          <Logo />
          <p className="mt-4 max-w-sm text-sm leading-relaxed text-white/55">
            Open-source, explainable AI-text detection.
            A GlassQuill score is a second opinion with reasons, never proof on its own.
          </p>
        </div>
        <FooterCol title="Product" links={[["/detect", "Detector"], ["/results", "Results"], ["/method", "Method"]]} />
        <FooterCol title="Learn" links={[["/method#data", "Datasets"], ["/method#features", "The 18 features"], ["/method#limits", "Limits and ethics"]]} />
      </div>
      <div className="mx-auto max-w-6xl px-4 pb-10 text-xs text-white/55 sm:px-6">
        Models: XGBoost + TF-IDF logistic regression, explained with SHAP. Trained on HC3, HAP-E and M4GT.
      </div>
    </footer>
  );
}

function FooterCol({ title, links }: { title: string; links: [string, string][] }) {
  return (
    <div>
      <h2 className="text-sm font-semibold text-white">{title}</h2>
      <ul className="mt-3 space-y-2">
        {links.map(([to, label]) => (
          <li key={to}>
            <Link to={to} className="text-sm text-white/55 transition-colors hover:text-white">
              {label}
            </Link>
          </li>
        ))}
      </ul>
    </div>
  );
}
