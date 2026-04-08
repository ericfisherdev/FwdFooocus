/**
 * Quick Settings — Alpine.js component.
 *
 * Tier 2 controls in the compose pane: performance, aspect ratio,
 * image count, output format, seed, styles.
 *
 * State is read by the Generate button when submitting to /api/generate.
 * All values initialize from $store.config defaults.
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
    };
}
