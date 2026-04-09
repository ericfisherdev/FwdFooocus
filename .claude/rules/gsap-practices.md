---
paths:
  - "**/*.js"
  - "**/*.html"
  - "**/*.css"
---

# GSAP Animation Practices

Derived from: GSAP official documentation (gsap.com/docs).

These rules apply when working with GSAP animations — tween/timeline code in JS, elements with GSAP-related attributes or classes in HTML, and animation-related CSS.

---

## Core Principles

- Animate transforms (`x`, `y`, `rotation`, `scale`) and `opacity` only — they are GPU-composited and don't trigger layout reflows.
- Use timelines to compose and orchestrate sequences. Individual tweens are building blocks; timelines are the choreography.
- Every tween should have an intentional ease. Linear motion looks mechanical.
- Clean up animations when DOM elements are removed (htmx swaps, Alpine destroy).

## Do's — Tweens

- Use `gsap.to()` for animating to target values, `gsap.from()` for animating from values, `gsap.fromTo()` for explicit start/end, `gsap.set()` for instant changes.
- Always specify `duration` explicitly — don't rely on the default.
- Use GSAP transform shorthands (`x`, `y`, `rotation`, `scale`, `xPercent`, `yPercent`) instead of CSS transform strings.
- Use `autoAlpha` instead of `opacity` — it also sets `visibility: hidden` at 0, removing invisible click targets.
- Use `overwrite: "auto"` to kill only conflicting properties of other tweens on the same target.
- Use `clearProps` after entrance animations to remove inline styles and let CSS take over.
- Use `stagger` for animating multiple targets with sequential delays. Use `stagger: { from: "center", grid: [r,c] }` for grid-based effects.

## Do's — Timelines

- Use `gsap.timeline({ defaults: { duration: 0.5, ease: "power2.out" } })` to avoid repeating common properties.
- Use the position parameter for precise placement: `"<"` (same start), `"<0.2"` (offset from start), `">-0.3"` (overlap with end), `"+=1"` (gap).
- Use `.addLabel('name')` for named positions. Reference labels in position parameters and ScrollTrigger snap.
- Use `yoyo: true` with `repeat: -1` for back-and-forth animations.
- Compose complex animations by building timeline functions that return timelines, then combining with `.add()`.

## Do's — Easing

- `"power2.out"` for decelerating entrances (elements appearing).
- `"power2.in"` for accelerating exits (elements leaving).
- `"power2.inOut"` for smooth state transitions.
- `"elastic.out(1, 0.3)"` for springy/bouncy effects.
- `"back.out(1.7)"` for overshoot-and-settle.
- `"none"` only for constant-speed animations (progress bars, tickers).
- `"steps(N)"` for frame-by-frame sprite animations.

## Do's — ScrollTrigger

- Use `trigger` to specify which element drives the animation.
- Use `start` and `end` with `"trigger viewport"` syntax (e.g., `"top center"`, `"bottom 80%"`).
- Use `scrub: true` to link animation to scroll position. Use `scrub: 1` for 1s of smoothing.
- Use `pin: true` to fix the trigger element while the animation plays.
- Use `snap` to snap to progress values or labels when scrolling stops.
- Use `toggleActions: "play pause resume reverse"` for non-scrubbed scroll-triggered animations.
- Use `markers: true` during development only.
- Use `ScrollTrigger.batch()` for lists of elements — more efficient than individual ScrollTriggers.

## Do's — Responsive & Cleanup

- Use `gsap.matchMedia()` for responsive animations — animations are automatically cleaned up when the media query stops matching.
- Use `gsap.context()` to track all animations created inside, then `ctx.revert()` to kill them all and restore inline styles.
- In Alpine components, create a `gsap.context()` in `init()` and call `ctx.revert()` in `destroy()`.
- For htmx, clean up GSAP contexts in `htmx:beforeSwap` event handlers for elements being replaced.

## Don'ts

- Don't animate layout properties (`left`, `top`, `width`, `height`, `margin`, `padding`) — they trigger expensive reflows. Use transforms.
- Don't use CSS transitions on elements GSAP is animating — they conflict and cause jank.
- Don't set `will-change` on many elements at once — it reserves GPU memory per element. Apply selectively and remove after animation.
- Don't forget to clean up animations when elements are removed. Orphaned tweens cause memory leaks.
- Don't use `onUpdate` for logic that doesn't need 60fps execution — it runs every frame.
- Don't use `immediateRender: true` on `gsap.to()` — it can cause visual jumps.
- Don't leave `markers: true` in production.
- Don't create tweens inside scroll or resize handlers — use ScrollTrigger.
- Don't use `clearProps: "all"` carelessly — it removes all inline styles, potentially breaking layout.

## Performance

- Transforms and opacity are GPU-composited — no layout or paint cost.
- `force3D: true` (GSAP default) promotes elements to their own GPU layer during animation.
- Use `autoAlpha` over `opacity` for both performance and accessibility.
- Use `overwrite: "auto"` to prevent animation pile-up from conflicting tweens.
- Batch ScrollTriggers with `ScrollTrigger.batch()` for element lists.
- Use `gsap.ticker` instead of raw `requestAnimationFrame`.
- Keep durations short: 0.2-0.5s for micro-interactions, 0.5-1s for transitions, 1-2s max for reveals.
- Remove `markers` in production — they are DOM elements that affect performance.
