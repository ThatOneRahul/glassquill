import { motion } from "motion/react";
import { ArrowLeft } from "lucide-react";
import { Link } from "react-router-dom";
import { group, rise, spring } from "../lib/motion";
import { Seo } from "../lib/seo";

export default function NotFound() {
  return (
    <section className="grid min-h-[80dvh] place-items-center px-4 pt-24 pb-16 text-center">
      <Seo title="Page not found" description="This page does not exist." noindex />
      <motion.div initial="hidden" animate="show" variants={group(0.1)}>
        <motion.p variants={rise} className="eyebrow">Error 404</motion.p>
        <motion.h1 variants={rise} className="mt-3 text-[clamp(3rem,10vw,7rem)] leading-none font-extrabold tracking-[-0.04em]">
          Page <span className="text-flame">not found.</span>
        </motion.h1>
        <motion.p variants={rise} className="mx-auto mt-5 max-w-md text-white/60">
          The page you were looking for does not exist. It may have moved, or the link may be mistyped.
        </motion.p>
        <motion.div variants={rise} className="mt-8 inline-block">
          <motion.div whileHover={{ y: -2 }} whileTap={{ scale: 0.97 }} transition={spring.press}>
            <Link to="/" className="inline-flex h-12 items-center gap-2 rounded-full bg-white px-6 text-sm font-semibold text-ink">
              <ArrowLeft className="size-4" aria-hidden="true" /> Back to home
            </Link>
          </motion.div>
        </motion.div>
      </motion.div>
    </section>
  );
}
