import { AnimatePresence, motion } from "motion/react";
import { BarChart3, BookOpen, House, Moon, ScanText, Sun } from "lucide-react";
import { cloneElement, useState, type ReactElement } from "react";
import { Link, NavLink, useLocation, useOutlet } from "react-router-dom";
import { useTheme } from "../lib/hooks";
import { duration, ease, spring } from "../lib/motion";
import { LogoMark } from "./ui";

const APP_NAV = [
  { to: "/detect", label: "Detector", icon: ScanText },
  { to: "/results", label: "Results", icon: BarChart3 },
  { to: "/method", label: "Method", icon: BookOpen },
];

/** Dashboard layout from the reference: light canvas, floating black pill sidebar. */
export function AppShell() {
  const [theme, toggle] = useTheme();
  const { pathname } = useLocation();
  const outlet = useOutlet();

  return (
    <div className={theme === "dark" ? "dark" : ""}>
      <div className="min-h-dvh bg-canvas text-ink transition-colors duration-300 dark:bg-ink dark:text-white">
        <a href="#app-main" className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[60] focus:rounded-full focus:bg-ink focus:px-4 focus:py-2 focus:text-white">
          Skip to content
        </a>

        {/* desktop: floating pill sidebar */}
        <aside className="fixed inset-y-4 left-4 z-30 hidden w-[68px] flex-col items-center rounded-full bg-ink py-5 shadow-2xl ring-1 ring-white/5 lg:flex">
          <Link to="/" aria-label="GlassQuill home" className="rounded-2xl">
            <LogoMark className="size-10" />
          </Link>
          <nav aria-label="App" className="mt-8 flex flex-1 flex-col items-center gap-3">
            {APP_NAV.map((n) => (
              <SideLink key={n.to} {...n} active={pathname === n.to} />
            ))}
          </nav>
          <div className="flex flex-col items-center gap-3">
            <SideButton label={theme === "dark" ? "Light mode" : "Dark mode"} onClick={toggle}>
              {theme === "dark" ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
            </SideButton>
            <SideLink to="/" label="Home" icon={House} active={false} />
          </div>
        </aside>

        {/* mobile: top bar + floating bottom nav (as in the reference's phone frame) */}
        <header className="sticky top-0 z-30 flex h-16 items-center justify-between bg-canvas/85 px-4 backdrop-blur-lg lg:hidden dark:bg-ink/85">
          <Link to="/" aria-label="GlassQuill home" className="flex items-center gap-2">
            <LogoMark className="size-8" />
            <span className="font-bold tracking-tight">GlassQuill</span>
          </Link>
          <button onClick={toggle} aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"} className="grid size-10 place-items-center rounded-full bg-black/5 dark:bg-white/10">
            {theme === "dark" ? <Sun className="size-[18px]" /> : <Moon className="size-[18px]" />}
          </button>
        </header>
        <nav aria-label="App" className="fixed inset-x-4 bottom-4 z-30 mx-auto flex max-w-sm items-center justify-around rounded-full bg-ink p-1.5 shadow-2xl lg:hidden">
          {APP_NAV.map((n) => {
            const active = pathname === n.to;
            return (
              <NavLink key={n.to} to={n.to} aria-current={active ? "page" : undefined} className="relative flex flex-1 items-center justify-center gap-2 rounded-full py-2.5 text-xs font-semibold">
                {active && <motion.span layoutId="bottom-pill" className="absolute inset-0 rounded-full bg-lavender" transition={spring.layout} />}
                <n.icon className={`relative size-[18px] ${active ? "text-ink" : "text-white/70"}`} aria-hidden="true" />
                <span className={`relative ${active ? "text-ink" : "text-white/70"}`}>{n.label}</span>
              </NavLink>
            );
          })}
        </nav>

        <div id="app-main" className="px-4 pt-2 pb-28 sm:px-6 lg:pt-6 lg:pr-8 lg:pb-10 lg:pl-[112px]">
          <AnimatePresence mode="wait" initial={false}>
            {outlet && (
              <motion.div
                key={pathname}
                initial={{ opacity: 0, y: 14 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                transition={{ duration: duration.component, ease: ease.out }}
              >
                {cloneElement(outlet as ReactElement)}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </div>
  );
}

function SideLink({ to, label, icon: Icon, active }: { to: string; label: string; icon: typeof House; active: boolean }) {
  return (
    <Tip label={label}>
      <NavLink to={to} aria-label={label} aria-current={active ? "page" : undefined} className="relative grid size-11 place-items-center rounded-full">
        {active && <motion.span layoutId="side-pill" className="absolute inset-0 rounded-full bg-lavender" transition={spring.layout} />}
        <motion.span whileHover={{ scale: 1.08 }} whileTap={{ scale: 0.92 }} transition={spring.press} className="relative">
          <Icon className={`size-[18px] transition-colors ${active ? "text-ink" : "text-white/60 group-hover:text-white"}`} aria-hidden="true" />
        </motion.span>
      </NavLink>
    </Tip>
  );
}

function SideButton({ label, onClick, children }: { label: string; onClick: () => void; children: ReactElement }) {
  return (
    <Tip label={label}>
      <motion.button
        onClick={onClick}
        aria-label={label}
        whileHover={{ scale: 1.08 }}
        whileTap={{ scale: 0.92, rotate: -15 }}
        transition={spring.press}
        className="grid size-11 place-items-center rounded-full text-white/60 hover:text-white"
      >
        {children}
      </motion.button>
    </Tip>
  );
}

/** Small label that slides out to the right on hover or keyboard focus ("Profile" in the reference). */
function Tip({ label, children }: { label: string; children: ReactElement }) {
  const [show, setShow] = useState(false);
  return (
    <div
      className="group relative"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
      onFocus={() => setShow(true)}
      onBlur={() => setShow(false)}
    >
      {children}
      <AnimatePresence>
        {show && (
          <motion.span
            role="tooltip"
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -6 }}
            transition={{ duration: duration.micro }}
            className="pointer-events-none absolute top-1/2 left-[54px] -translate-y-1/2 rounded-lg bg-ink px-2.5 py-1 text-xs font-medium whitespace-nowrap text-white shadow-lg ring-1 ring-white/10"
          >
            {label}
          </motion.span>
        )}
      </AnimatePresence>
    </div>
  );
}
