/**
 * Settings State — Alpine.js component.
 *
 * Reactive state for Tier 3 Settings Drawer controls.
 * Values initialize from $store.config defaults.
 * Read by the Generate button when submitting to /api/generate.
 *
 * baseModel itself lives in $store.model (new_ui/static/js/stores.js) so
 * sibling components (compose.html's quick settings) can react to it too;
 * this component watches $store.model.baseModel to trigger capability
 * reloads, and $store.model.capabilities to filter/reset the sampler and
 * scheduler choices below.
 */

function settingsState() {
    return {
        refinerModel: 'None',
        refinerSwitch: 0.8,
        guidanceScale: 4.0,
        sharpness: 2.0,
        sampler: 'dpmpp_2m_sde_gpu',
        scheduler: 'karras',
        clipSkip: 2,
        vae: 'Default (model)',
        saveMetadata: true,
        metadataScheme: 'fooocus',

        // Global sampler/scheduler lists (loaded from API), used as a
        // fallback until $store.model.capabilities resolves — see the
        // availableSamplers/availableSchedulers getters below.
        _globalSamplers: [],
        _globalSchedulers: [],

        async init() {
            const cfg = Alpine.store('config');
            if (!cfg.loaded) {
                await cfg.load();
            }
            this.refinerModel = cfg.defaultRefiner || 'None';
            this.refinerSwitch = cfg.defaultRefinerSwitch ?? 0.8;
            this.guidanceScale = cfg.defaultCfgScale ?? 4.0;
            this.sharpness = cfg.defaultSampleSharpness ?? 2.0;
            this.sampler = cfg.defaultSampler || 'dpmpp_2m_sde_gpu';
            this.scheduler = cfg.defaultScheduler || 'karras';

            this._loadSamplers();

            // Reload capabilities whenever the selected base model changes.
            // Registered once here (this component's init() runs once per
            // page load) rather than inside $store.model itself, to keep
            // the store focused on holding/fetching state, not on driving
            // its own reactivity.
            this.$watch('$store.model.baseModel', (name) => {
                Alpine.store('model').loadCapabilities(name);
            });

            // Fall back sampler/scheduler to the new family's own default
            // whenever the currently selected value isn't in its allowed
            // list (mirrors the backend's _validated_choice fallback in
            // new_ui/app.py's _build_generate_args).
            this.$watch('$store.model.capabilities', () => this._resetInvalidChoices());
            // $watch is lazy: capabilities may already have loaded before
            // this component initialized, so reconcile once immediately.
            if (this.$store.model.capabilities) this._resetInvalidChoices();
        },

        async _loadSamplers() {
            try {
                const resp = await fetch('/api/samplers');
                if (resp.ok) {
                    const data = await resp.json();
                    this._globalSamplers = data.samplers || [];
                    this._globalSchedulers = data.schedulers || [];
                }
            } catch (e) { console.warn('[settingsState] Failed to load samplers:', e); }
        },

        get availableSamplers() {
            return Alpine.store('model').capabilities?.sampler_names || this._globalSamplers;
        },

        get availableSchedulers() {
            return Alpine.store('model').capabilities?.scheduler_names || this._globalSchedulers;
        },

        _resetInvalidChoices() {
            const caps = Alpine.store('model').capabilities;
            if (!caps) return;
            if (!caps.sampler_names.includes(this.sampler)) {
                this.sampler = caps.sampler_names[0];
            }
            if (!caps.scheduler_names.includes(this.scheduler)) {
                this.scheduler = caps.scheduler_names[0];
            }
        },
    };
}
