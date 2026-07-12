/**
 * Alpine.js Global Stores
 *
 * Single source of truth for shared state. Components read from stores,
 * not from their own fetches.
 *
 * Stores:
 *   $store.config   — loaded from /api/config on init
 *   $store.data     — LoRA/model/style lists, refreshed every 60s
 *   $store.ui       — panel open/closed state, persisted to localStorage
 *   $store.generation — generation progress, updated from WebSocket
 *   $store.model    — selected base model + its FamilyCapabilities descriptor
 *
 * Must be loaded before Alpine.js initializes (use alpine:init event).
 */

const UI_STORAGE_KEY = 'fwd_ui_state';
const REFRESH_INTERVAL_MS = 60_000;


document.addEventListener('alpine:init', () => {

    /* ==============================================================
     * $store.config — backend config values (loaded once on init)
     * ============================================================== */

    Alpine.store('config', {
        loaded: false,
        defaultModel: '',
        defaultRefiner: '',
        defaultRefinerSwitch: 0.8,
        defaultPerformance: 'Speed',
        defaultAspectRatio: '1152*896',
        availableAspectRatios: [],
        defaultImageNumber: 2,
        maxImageNumber: 32,
        defaultOutputFormat: 'png',
        defaultPrompt: '',
        defaultPromptNegative: '',
        defaultStyles: [],
        defaultCfgScale: 4.0,
        defaultSampleSharpness: 2.0,
        defaultSampler: 'dpmpp_2m_sde_gpu',
        defaultScheduler: 'karras',
        defaultLoras: [],
        defaultLorasMinWeight: -2,
        defaultLorasMaxWeight: 5,
        defaultMaxLoraNumber: 5,

        async load() {
            try {
                const resp = await fetch('/api/config');
                if (!resp.ok) return;
                const data = await resp.json();
                this.defaultModel = data.default_model || '';
                this.defaultRefiner = data.default_refiner || '';
                this.defaultRefinerSwitch = data.default_refiner_switch ?? 0.8;
                this.defaultPerformance = data.default_performance || 'Speed';
                this.defaultAspectRatio = data.default_aspect_ratio || '1152*896';
                this.availableAspectRatios = data.available_aspect_ratios || [];
                this.defaultImageNumber = data.default_image_number ?? 2;
                this.maxImageNumber = data.max_image_number ?? 32;
                this.defaultOutputFormat = data.default_output_format || 'png';
                this.defaultPrompt = data.default_prompt || '';
                this.defaultPromptNegative = data.default_prompt_negative || '';
                this.defaultStyles = data.default_styles || [];
                this.defaultCfgScale = data.default_cfg_scale ?? 4.0;
                this.defaultSampleSharpness = data.default_sample_sharpness ?? 2.0;
                this.defaultSampler = data.default_sampler || 'dpmpp_2m_sde_gpu';
                this.defaultScheduler = data.default_scheduler || 'karras';
                this.defaultLoras = data.default_loras || [];
                this.defaultLorasMinWeight = data.default_loras_min_weight ?? -2;
                this.defaultLorasMaxWeight = data.default_loras_max_weight ?? 5;
                this.defaultMaxLoraNumber = data.default_max_lora_number ?? 5;
                this.loaded = true;
            } catch (e) {
                console.error('[store.config] Failed to load:', e);
            }
        },
    });


    /* ==============================================================
     * $store.data — LoRA/model/style lists (refreshed every 60s)
     * ============================================================== */

    Alpine.store('data', {
        loraList: [],
        modelList: [],
        styleList: [],
        lastRefresh: null,
        _intervalId: null,

        async refresh() {
            try {
                const [modelsResp, lorasResp, stylesResp] = await Promise.all([
                    fetch('/api/models'),
                    fetch('/api/lora-library-data'),
                    fetch('/api/styles'),
                ]);

                if (modelsResp.ok) {
                    const models = await modelsResp.json();
                    this.modelList = models.checkpoints || [];
                }
                if (lorasResp.ok) {
                    this.loraList = await lorasResp.json();
                }
                if (stylesResp.ok) {
                    const styles = await stylesResp.json();
                    this.styleList = styles.styles || [];
                }
                this.lastRefresh = Date.now();
            } catch (e) {
                console.error('[store.data] Refresh failed:', e);
            }
        },

        startAutoRefresh() {
            this.refresh();
            this._intervalId = setInterval(() => this.refresh(), REFRESH_INTERVAL_MS);
        },

        stopAutoRefresh() {
            if (this._intervalId) {
                clearInterval(this._intervalId);
                this._intervalId = null;
            }
        },
    });


    /* ==============================================================
     * $store.ui — panel state, persisted to localStorage
     * ============================================================== */

    const savedUi = (() => {
        try {
            return JSON.parse(localStorage.getItem(UI_STORAGE_KEY)) || {};
        } catch {
            return {};
        }
    })();

    Alpine.store('ui', {
        settingsDrawerOpen: savedUi.settingsDrawerOpen ?? false,
        devToolsOpen: savedUi.devToolsOpen ?? false,
        devModeEnabled: savedUi.devModeEnabled ?? false,
        activeTab: savedUi.activeTab ?? 'compose',  // for tablet stacked mode
        connectionState: 'disconnected',  // connected | connecting | reconnecting | disconnected

        toggleSettings() {
            this.settingsDrawerOpen = !this.settingsDrawerOpen;
            this._persist();
        },

        toggleDevTools() {
            this.devToolsOpen = !this.devToolsOpen;
            this._persist();
        },

        setDevMode(enabled) {
            this.devModeEnabled = enabled;
            if (!enabled) this.devToolsOpen = false;
            this._persist();
        },

        setActiveTab(tab) {
            this.activeTab = tab;
            this._persist();
        },

        _persist() {
            try {
                localStorage.setItem(UI_STORAGE_KEY, JSON.stringify({
                    settingsDrawerOpen: this.settingsDrawerOpen,
                    devToolsOpen: this.devToolsOpen,
                    devModeEnabled: this.devModeEnabled,
                    activeTab: this.activeTab,
                }));
            } catch { /* localStorage full or unavailable */ }
        },
    });


    /* ==============================================================
     * $store.generation — live generation state
     * ============================================================== */

    Alpine.store('generation', {
        isGenerating: false,
        currentImage: 0,
        totalImages: 0,
        percentage: 0,
        previewImage: null,
        etaSeconds: null,
        progressText: '',

        // Quick-settings values synced from the quickSettings component
        // so the Generate button (outside its scope) can read them.
        performance: 'Speed',
        aspectRatio: '',
        imageNumber: 2,
        outputFormat: 'png',
        seed: -1,
        selectedStyles: [],

        reset() {
            this.isGenerating = false;
            this.currentImage = 0;
            this.totalImages = 0;
            this.percentage = 0;
            this.previewImage = null;
            this.etaSeconds = null;
            this.progressText = '';
        },
    });


    /* ==============================================================
     * $store.model — selected base model + its capability descriptor
     *
     * The cross-component bridge for family-aware UI gating: baseModel
     * (bound from the settings drawer's Base Model select) and the
     * capabilities it resolves to are read by sibling components
     * (compose.html's quick settings, the settings drawer's sampling
     * fields) that cannot see each other's local x-data.
     * ============================================================== */

    Alpine.store('model', {
        baseModel: '',
        family: null,
        capabilities: null,
        _capabilitiesRequestSeq: 0,

        /** Fetch the FamilyCapabilities descriptor for a checkpoint.
         *  Leaves the previously loaded family/capabilities in place on
         *  a failed or non-OK fetch rather than throwing or clearing
         *  them, so gated controls keep their last-known-good state.
         *  A request sequence token discards stale responses: switching
         *  A -> B must never let A's slower response overwrite B's
         *  family/capabilities after the fact. */
        async loadCapabilities(checkpointName) {
            const seq = ++this._capabilitiesRequestSeq;
            try {
                const resp = await fetch(`/api/capabilities?checkpoint=${encodeURIComponent(checkpointName)}`);
                if (seq !== this._capabilitiesRequestSeq) return; // superseded
                if (!resp.ok) return;
                const data = await resp.json();
                if (seq !== this._capabilitiesRequestSeq) return; // superseded
                this.family = data.family;
                this.capabilities = data;
            } catch (e) {
                console.error('[store.model] Failed to load capabilities:', e);
            }
        },
    });


    /* ==============================================================
     * Init: load config, start data refresh
     * ============================================================== */

    Alpine.store('config').load().then(() => {
        // Resolve the default checkpoint's family before compose.html's
        // performance radio / aspect-ratio select settle into their final
        // state, so there is no flash of the wrong control set.
        const defaultModel = Alpine.store('config').defaultModel;
        Alpine.store('model').baseModel = defaultModel;
        Alpine.store('model').loadCapabilities(defaultModel);
    });
    Alpine.store('data').startAutoRefresh();
});
