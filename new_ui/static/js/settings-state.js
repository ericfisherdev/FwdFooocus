/**
 * Settings State — Alpine.js component.
 *
 * Reactive state for Tier 3 Settings Drawer controls.
 * Values initialize from $store.config defaults.
 * Read by the Generate button when submitting to /api/generate.
 */

function settingsState() {
    return {
        baseModel: '',
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

        // Available options (loaded from API)
        availableSamplers: [],
        availableSchedulers: [],

        async init() {
            const cfg = Alpine.store('config');
            if (!cfg.loaded) {
                await cfg.load();
            }
            this.baseModel = cfg.defaultModel || '';
            this.refinerModel = cfg.defaultRefiner || 'None';
            this.refinerSwitch = cfg.defaultRefinerSwitch ?? 0.8;
            this.guidanceScale = cfg.defaultCfgScale ?? 4.0;
            this.sharpness = cfg.defaultSampleSharpness ?? 2.0;
            this.sampler = cfg.defaultSampler || 'dpmpp_2m_sde_gpu';
            this.scheduler = cfg.defaultScheduler || 'karras';

            this._loadSamplers();
        },

        async _loadSamplers() {
            try {
                const resp = await fetch('/api/samplers');
                if (resp.ok) {
                    const data = await resp.json();
                    this.availableSamplers = data.samplers || [];
                    this.availableSchedulers = data.schedulers || [];
                }
            } catch (e) { console.warn('[settingsState] Failed to load samplers:', e); }
        },
    };
}
