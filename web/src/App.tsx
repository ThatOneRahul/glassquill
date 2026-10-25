import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { Component, lazy, Suspense, useEffect, type ReactNode } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { SiteLayout } from "./components/SiteLayout";
import { duration, ease } from "./lib/motion";
import Home from "./pages/Home";

// inner pages load on demand so the landing page ships less JavaScript
const Detect = lazy(() => import("./pages/Detect"));
const Results = lazy(() => import("./pages/Results"));
const Method = lazy(() => import("./pages/Method"));
const NotFound = lazy(() => import("./pages/NotFound"));

/** If a page fails to load (server stopped, or a new build replaced its file), say so
 *  instead of leaving the screen blank. */
class PageErrorBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div role="alert" className="grid min-h-[70dvh] place-items-center px-4 pt-24 text-center">
        <div>
          <p className="text-2xl font-bold">This page could not be loaded.</p>
          <button onClick={() => window.location.reload()} className="mt-6 h-11 rounded-full bg-white px-5 text-sm font-semibold text-ink">
            Reload
          </button>
        </div>
      </div>
    );
  }
}

/** Scroll to an anchor once it exists: the target page may still be fading in or loading. */
function scrollToHash(hash: string, tries = 40) {
  let id: string;
  try {
    id = decodeURIComponent(hash.slice(1));
  } catch {
    return;                                      // malformed link such as "#%": ignore it
  }
  const el = document.getElementById(id);
  if (el) el.scrollIntoView();
  else if (tries > 0) setTimeout(() => scrollToHash(hash, tries - 1), 50);
}

export default function App() {
  const location = useLocation();

  // anchors on the same page, and on first load
  useEffect(() => {
    if (location.hash) scrollToHash(location.hash);
  }, [location.hash]);

  return (
    <MotionConfig reducedMotion="user">
      {/* one layout for every page (header + footer stay put); the page content cross-fades */}
      <SiteLayout>
        <AnimatePresence
          mode="wait"
          initial={false}
          // scroll only once the old page has faded out, so it doesn't jump while leaving
          onExitComplete={() => (window.location.hash ? scrollToHash(window.location.hash) : window.scrollTo(0, 0))}
        >
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: duration.component, ease: ease.out }}
          >
            <PageErrorBoundary>
              <Suspense fallback={<div className="min-h-dvh" />}>
                <Routes location={location}>
                  <Route path="/" element={<Home />} />
                  <Route path="/detect" element={<Detect />} />
                  <Route path="/results" element={<Results />} />
                  <Route path="/method" element={<Method />} />
                  <Route path="*" element={<NotFound />} />
                </Routes>
              </Suspense>
            </PageErrorBoundary>
          </motion.div>
        </AnimatePresence>
      </SiteLayout>
    </MotionConfig>
  );
}
