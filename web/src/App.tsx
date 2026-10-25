import { AnimatePresence, MotionConfig, motion } from "motion/react";
import { lazy, Suspense, useLayoutEffect } from "react";
import { Route, Routes, useLocation } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { SiteLayout } from "./components/SiteLayout";
import { duration, ease } from "./lib/motion";
import Home from "./pages/Home";

// dashboard pages load on demand so the landing page ships less JavaScript
const Detect = lazy(() => import("./pages/Detect"));
const Results = lazy(() => import("./pages/Results"));
const Method = lazy(() => import("./pages/Method"));
const NotFound = lazy(() => import("./pages/NotFound"));

const APP_PAGES = ["/detect", "/results", "/method"];

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
  // the marketing site and the dashboard swap as a whole; pages inside the dashboard
  // animate within the persistent shell (so the sidebar indicator can glide)
  const area = APP_PAGES.includes(location.pathname) ? "app" : location.pathname === "/" ? "site" : "404";

  return (
    <MotionConfig reducedMotion="user">
      <ScrollToTop />
      <AnimatePresence mode="wait">
        <motion.div
          key={area}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: duration.component, ease: ease.inOut }}
        >
          <Suspense fallback={<div className="min-h-dvh bg-canvas" />}>
            <Routes location={location}>
              <Route path="/" element={<Home />} />
              <Route element={<AppShell />}>
                <Route path="/detect" element={<Detect />} />
                <Route path="/results" element={<Results />} />
                <Route path="/method" element={<Method />} />
              </Route>
              <Route
                path="*"
                element={
                  <SiteLayout>
                    <NotFound />
                  </SiteLayout>
                }
              />
            </Routes>
          </Suspense>
        </motion.div>
      </AnimatePresence>
    </MotionConfig>
  );
}
