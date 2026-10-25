/**
 * Motion tokens. Every animation on the site uses these, so the timing feels like one system.
 * Only transform and opacity are animated (cheap for the browser), and MotionConfig in App.tsx
 * turns movement off for people who ask their system for reduced motion.
 */
import type { Transition, Variants } from "motion/react";

export const ease = { out: [0.22, 1, 0.36, 1] as const };          // fast start, soft landing

export const duration = { micro: 0.15, component: 0.35, section: 0.6 };

export const spring = {
  press: { type: "spring", stiffness: 500, damping: 30 } satisfies Transition,    // buttons
  lift: { type: "spring", stiffness: 320, damping: 24 } satisfies Transition,     // cards
  layout: { type: "spring", stiffness: 380, damping: 34 } satisfies Transition,   // sliding indicators
};

/** Big entrance (headlines, sections): fade in while rising 24 px. */
export const rise: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: duration.section, ease: ease.out } },
};

/** Smaller entrance (cards, rows): shorter distance, quicker. */
export const pop: Variants = {
  hidden: { opacity: 0, y: 12, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: duration.component, ease: ease.out } },
};

/** A parent that reveals its children one after another. */
export const group = (gap = 0.08, delay = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: gap, delayChildren: delay } },
});

/** Scroll reveals run once, when a fifth of the element is visible. */
export const inView = { once: true, amount: 0.2 } as const;

/** Card hover lift and button press. */
export const lift = { y: -5, transition: spring.lift };
export const press = { scale: 0.97, transition: spring.press };
