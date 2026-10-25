/**
 * Motion tokens. Every animation in the site uses these, so timing feels like one system.
 * Hierarchy: micro (controls) < component (cards, panels) < section/hero (page entrances).
 * Only transform and opacity are animated.
 */
import type { Transition, Variants } from "motion/react";

export const ease = {
  out: [0.22, 1, 0.36, 1] as const,          // expo-out: entrances
  inOut: [0.65, 0, 0.35, 1] as const,        // state changes
};

export const duration = { micro: 0.15, component: 0.35, section: 0.6, hero: 0.9 };

export const spring = {
  press: { type: "spring", stiffness: 500, damping: 30 } satisfies Transition,    // buttons
  lift: { type: "spring", stiffness: 320, damping: 24 } satisfies Transition,     // cards
  layout: { type: "spring", stiffness: 380, damping: 34 } satisfies Transition,   // indicators
  number: { type: "spring", stiffness: 60, damping: 18 } satisfies Transition,    // counters
};

export const stagger = { tight: 0.05, base: 0.08, loose: 0.12 };

/** Section/hero entrance: fade + rise. */
export const rise: Variants = {
  hidden: { opacity: 0, y: 24 },
  show: { opacity: 1, y: 0, transition: { duration: duration.section, ease: ease.out } },
};

/** Component entrance: smaller distance, quicker. */
export const pop: Variants = {
  hidden: { opacity: 0, y: 12, scale: 0.98 },
  show: { opacity: 1, y: 0, scale: 1, transition: { duration: duration.component, ease: ease.out } },
};

export const fade: Variants = {
  hidden: { opacity: 0 },
  show: { opacity: 1, transition: { duration: duration.component, ease: ease.out } },
};

/** Parent that staggers its children's `hidden -> show`. */
export const group = (gap: number = stagger.base, delay = 0): Variants => ({
  hidden: {},
  show: { transition: { staggerChildren: gap, delayChildren: delay } },
});

/** Scroll reveal settings: once, when a fifth of the element is visible. */
export const inView = { once: true, amount: 0.2 } as const;

/** Card hover lift used everywhere (matches the reference `whileHover={{ y: -5 }}`). */
export const lift = { y: -5, transition: spring.lift };
export const press = { scale: 0.97, transition: spring.press };
