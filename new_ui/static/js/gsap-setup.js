/**
 * GSAP animation helpers with reduced motion support.
 *
 * Maps design token durations/easings to GSAP parameters.
 * All helpers skip to end state when prefers-reduced-motion is active.
 *
 * Usage from Alpine components:
 *   fwdAnimate(this.$refs.drawer, { x: 0, opacity: 1 }, 'normal', 'out')
 *   fwdSlideIn(this.$refs.panel, 'right', 'normal')
 *   fwdFadeIn(this.$refs.grid.children, 'normal', 0.05)
 */

const FWD_GSAP = (() => {

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
    let reducedMotion = prefersReducedMotion.matches;

    prefersReducedMotion.addEventListener('change', (e) => {
        reducedMotion = e.matches;
    });

    /* -- Duration map (matches CSS token values) -------------------- */

    const DURATIONS = {
        fast:   0.15,
        normal: 0.25,
        slow:   0.4,
    };

    /* -- Easing map (GSAP string easings) --------------------------- */

    const EASINGS = {
        default: 'power2.inOut',
        in:      'power2.in',
        out:     'power2.out',
        spring:  'back.out(1.7)',
    };

    function getDuration(name) {
        if (reducedMotion) return 0;
        return DURATIONS[name] ?? DURATIONS.normal;
    }

    function getEasing(name) {
        return EASINGS[name] ?? EASINGS.default;
    }

    /* -- Core animate ----------------------------------------------- */

    function animate(target, props, duration = 'normal', ease = 'default') {
        return gsap.to(target, {
            ...props,
            duration: getDuration(duration),
            ease: getEasing(ease),
        });
    }

    /* -- Slide in from direction ------------------------------------ */

    function slideIn(target, direction = 'right', duration = 'normal') {
        const axis = (direction === 'left' || direction === 'right') ? 'x' : 'y';
        const offset = (direction === 'right' || direction === 'down') ? 100 : -100;

        gsap.set(target, { [axis]: offset, opacity: 0 });
        return gsap.to(target, {
            [axis]: 0,
            opacity: 1,
            duration: getDuration(duration),
            ease: getEasing('out'),
        });
    }

    /* -- Slide out to direction ------------------------------------- */

    function slideOut(target, direction = 'right', duration = 'normal') {
        const axis = (direction === 'left' || direction === 'right') ? 'x' : 'y';
        const offset = (direction === 'right' || direction === 'down') ? 100 : -100;

        return gsap.to(target, {
            [axis]: offset,
            opacity: 0,
            duration: getDuration(duration),
            ease: getEasing('in'),
        });
    }

    /* -- Fade in (with optional stagger for lists) ------------------ */

    function fadeIn(targets, duration = 'normal', stagger = 0) {
        gsap.set(targets, { opacity: 0, y: 8 });
        return gsap.to(targets, {
            opacity: 1,
            y: 0,
            duration: getDuration(duration),
            ease: getEasing('out'),
            stagger: reducedMotion ? 0 : stagger,
        });
    }

    /* -- Fade out --------------------------------------------------- */

    function fadeOut(targets, duration = 'fast') {
        return gsap.to(targets, {
            opacity: 0,
            duration: getDuration(duration),
            ease: getEasing('in'),
        });
    }

    /* -- Scale pulse (for badges, counts) --------------------------- */

    function pulse(target) {
        if (reducedMotion) return;
        return gsap.fromTo(target,
            { scale: 1 },
            { scale: 1.15, duration: getDuration('fast'), ease: getEasing('spring'), yoyo: true, repeat: 1 }
        );
    }

    return {
        animate,
        slideIn,
        slideOut,
        fadeIn,
        fadeOut,
        pulse,
        getDuration,
        getEasing,
        get reducedMotion() { return reducedMotion; },
    };

})();

// Expose globally for Alpine component access
window.fwdAnimate = FWD_GSAP.animate;
window.fwdSlideIn = FWD_GSAP.slideIn;
window.fwdSlideOut = FWD_GSAP.slideOut;
window.fwdFadeIn = FWD_GSAP.fadeIn;
window.fwdFadeOut = FWD_GSAP.fadeOut;
window.fwdPulse = FWD_GSAP.pulse;
