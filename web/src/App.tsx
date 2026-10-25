import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useLayoutEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { SiteLayout } from "./components/SiteLayout";
import { duration, ease } from "./lib/motion";
import Home from "./pages/Home";

// inner pages load on demand so the landing page ships less JavaScript
const Detect = lazy(() => import("./pages/Detect"));
const Results = lazy(() => import("./pages/Results"));
const Method = lazy(() => import("./pages/Method"));
const NotFound = lazy(() => import("./pages/NotFound"));

function ScrollToTop() {
  const { pathname, hash } = useLocation();
  useLayoutEffect(() => {
    if (hash) document.getElementById(hash.slice(1))?.scrollIntoView();
    else window.scrollTo(0, 0);
  }, [pathname, hash]);
  return null;
}

export default function App() {
  const location = useLocation();
  return (
    <MotionConfig reducedMotion="user">
      <ScrollToTop />
      {/* one layout for every page (header + footer stay put); the page content cross-fades */}
      <SiteLayout>
        <AnimatePresence mode="wait" initial={false}>
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
