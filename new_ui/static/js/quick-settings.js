/**
 * Quick Settings — Alpine.js component.
 *
 * Tier 2 controls in the compose pane: performance, aspect ratio,
 * image count, output format, seed, styles.
 *
 * State is read by the Generate button when submitting to /api/generate.
 * All values initialize from $store.config defaults.
 *
 * Performance and aspect-ratio choices are family-aware: performanceModes/
 * aspectRatioOptions prefer $store.model.capabilities (new_ui/static/js/
 * stores.js) and fall back to the global SDXL-equivalent lists until
 * capabilities resolve or if a fetch fails.
 */

function quickSettings() {
    return {
        performance: 'Speed',
        aspectRatio: '',
        imageNumber: 2,
        outputFormat: 'png',
        seed: -1,
        randomSeed: true,
        selectedStyles: [],
        stylesOpen: false,
        styleSearch: '',
        defaultsApplied: false,

        init() {
            const applyDefaults = () => {
                if (this.defaultsApplied) return;
                this.defaultsApplied = true;
                const cfg = Alpine.store('config');
                this.performance = cfg.defaultPerformance || 'Speed';
                this.aspectRatio = cfg.defaultAspectRatio || '';
                this.imageNumber = cfg.defaultImageNumber || 2;
                this.outputFormat = cfg.defaultOutputFormat || 'png';
                this.selectedStyles = [...(cfg.defaultStyles || [])];
                this.seed = -1;
                this.randomSeed = true;
                this._syncToStore();
            };

            if (Alpine.store('config').loaded) {
                applyDefaults();
            } else {
                this.$watch('$store.config.loaded', (loaded) => {
                    if (loaded) applyDefaults();
                });
            }

            // Keep shared store in sync whenever any quick setting changes.
            const watchedKeys = ['performance', 'aspectRatio', 'imageNumber',
                                 'outputFormat', 'seed', 'randomSeed', 'selectedStyles'];
            watchedKeys.forEach((key) => {
                this.$watch(key, () => this._syncToStore());
            });

            // Fall back performance/aspect-ratio to the new family's own
            // default whenever the currently selected value isn't in its
            // allowed list (mirrors the backend's _resolve_performance_
            // selection/_validated_or_default fallbacks in new_ui/app.py).
            this.$watch('$store.model.capabilities', () => this._resetInvalidChoices());
        },

        /** Push current quick-settings values into $store.generation so
         *  the Generate button (outside this component) can read them. */
        _syncToStore() {
            const gen = Alpine.store('generation');
            gen.performance = this.performance;
            gen.aspectRatio = this.aspectRatio;
            gen.imageNumber = this.imageNumber;
            gen.outputFormat = this.outputFormat;
            gen.seed = this.effectiveSeed;
            gen.selectedStyles = [...this.selectedStyles];
        },

        get effectiveSeed() {
            return this.randomSeed ? -1 : this.seed;
        },

        /** Performance modes as {label, ...} descriptors, sourced from the
         *  selected base model's capabilities. Falls back to today's
         *  SDXL-equivalent labels (wrapped the same shape) until
         *  capabilities resolve, so the template can always key off
         *  mode.label regardless of source. */
        get performanceModes() {
            const modes = Alpine.store('model').capabilities?.performance_modes;
            if (modes && modes.length) return modes;
            return ['Speed', 'Quality', 'Extreme Speed'].map((label) => ({ label }));
        },

        get aspectRatioOptions() {
            return Alpine.store('model').capabilities?.aspect_ratios
                || Alpine.store('config').availableAspectRatios;
        },

        get filteredStyles() {
            const q = this.styleSearch.toLowerCase().trim();
            const all = Alpine.store('data').styleList || [];
            if (!q) return all;
            return all.filter(s => s.toLowerCase().includes(q));
        },

        get selectedStyleCount() {
            return this.selectedStyles.length;
        },

        setPerformance(value) {
            this.performance = value;
        },

        toggleStyle(name) {
            const idx = this.selectedStyles.indexOf(name);
            if (idx >= 0) {
                this.selectedStyles.splice(idx, 1);
            } else {
                this.selectedStyles.push(name);
            }
        },

        isStyleSelected(name) {
            return this.selectedStyles.includes(name);
        },

        updateImageNumber(value) {
            const parsed = Number.parseInt(value, 10);
            const max = Alpine.store('config').maxImageNumber || 32;
            this.imageNumber = Number.isFinite(parsed) ? Math.min(Math.max(parsed, 1), max) : 1;
        },

        toggleRandomSeed() {
            this.randomSeed = !this.randomSeed;
            if (this.randomSeed) {
                this.seed = -1;
            } else if (this.seed < 0) {
                this.seed = 0;
            }
        },

        _resetInvalidChoices() {
            const caps = Alpine.store('model').capabilities;
            if (!caps) return;
            const labels = caps.performance_modes.map((mode) => mode.label);
            if (!labels.includes(this.performance)) {
                this.performance = labels[0];
            }
            if (!caps.aspect_ratios.includes(this.aspectRatio)) {
                this.aspectRatio = caps.aspect_ratios[0];
            }
        },
    };
}
