import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { SiteLayout } from "./components/SiteLayout";
import { duration, ease } from "./lib/motion";
import Home from "./pages/Home";

// inner pages load on demand so the landing page ships less JavaScript
const Detect = lazy(() => import("./pages/Detect"));
const Results = lazy(() => import("./pages/Results"));
const Method = lazy(() => import("./pages/Method"));
const NotFound = lazy(() => import("./pages/NotFound"));

/** Scroll to an anchor once it exists: the target page may still be fading in or loading. */
function scrollToHash(hash: string, tries = 40) {
  const el = document.getElementById(decodeURIComponent(hash.slice(1)));
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
            <Suspense fallback={<div className="min-h-dvh" />}>
              <Routes location={location}>
                <Route path="/" element={<Home />} />
                <Route path="/detect" element={<Detect />} />
                <Route path="/results" element={<Results />} />
                <Route path="/method" element={<Method />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </SiteLayout>
    </MotionConfig>
  );
}
