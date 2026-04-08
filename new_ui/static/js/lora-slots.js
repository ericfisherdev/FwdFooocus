/**
 * LoRA Slots — Alpine.js component.
 *
 * Manages active LoRAs as compact chips. Minimal version for
 * Generation Core — full Picker replaces the dropdown in Phase 3.
 *
 * Spec: component-spec-lora-slot.md
 */

const LORA_COLORS = [
    'var(--lora-color-1)',  'var(--lora-color-2)',  'var(--lora-color-3)',
    'var(--lora-color-4)',  'var(--lora-color-5)',  'var(--lora-color-6)',
    'var(--lora-color-7)',  'var(--lora-color-8)',  'var(--lora-color-9)',
    'var(--lora-color-10)', 'var(--lora-color-11)', 'var(--lora-color-12)',
    'var(--lora-color-13)', 'var(--lora-color-14)', 'var(--lora-color-15)',
    'var(--lora-color-16)',
];

function loraSlots() {
    return {
        slots: [],
        pickerOpen: false,
        maxSlots: 5,

        init() {
            const applyDefaults = () => {
                const config = Alpine.store('config');
                this.maxSlots = config.defaultMaxLoraNumber ?? 5;
                (config.defaultLoras || []).forEach(([filename, weight]) => {
                    if (filename && filename !== 'None') {
                        this.addLora(filename, weight, false);
                    }
                });
            };
            if (Alpine.store('config').loaded) {
                applyDefaults();
            } else {
                this.$watch('$store.config.loaded', (loaded) => {
                    if (loaded && this.slots.length === 0) {
                        applyDefaults();
                    }
                });
            }
        },

        get availableLoras() {
            const active = new Set(this.slots.map(s => s.filename));
            return Alpine.store('data').loraList
                .map(l => l.relative_path || l.filename)
                .filter(f => !active.has(f));
        },

        get canAdd() {
            return this.slots.length < this.maxSlots;
        },

        async addLora(filename, weight = 1.0, skipEvent = false) {
            if (!this.canAdd) return;
            if (this.slots.some(s => s.filename === filename)) return;

            const slot = {
                filename,
                weight,
                color: LORA_COLORS[this.slots.length % LORA_COLORS.length],
                triggerWords: [],
                slotIndex: this.slots.length,
            };
            this.slots.push(slot);

            if (!skipEvent) {
                this._dispatchChanged(slot);
            }

            try {
                const resp = await fetch(`/api/lora-trigger-words?filename=${encodeURIComponent(filename)}`);
                if (resp.ok) {
                    const data = await resp.json();
                    if (!this.slots.includes(slot)) return;
                    slot.triggerWords = data.trigger_words || [];
                    if (!skipEvent) {
                        this._dispatchChanged(slot);
                    }
                }
            } catch { /* non-critical */ }
        },

        removeLora(index) {
            const removed = this.slots.splice(index, 1)[0];
            // Re-index remaining slots
            this.slots.forEach((s, i) => {
                s.slotIndex = i;
                s.color = LORA_COLORS[i % LORA_COLORS.length];
            });
            if (removed) {
                window.dispatchEvent(new CustomEvent('lora-removed', {
                    detail: { index, filename: removed.filename },
                }));
                // Re-dispatch all remaining so colors update
                this.slots.forEach(s => this._dispatchChanged(s));
            }
            // Always notify listeners that LoRA set changed (covers empty-list case)
            window.dispatchEvent(new CustomEvent('lora-changed', {
                detail: { slots: this.slots.map(s => ({ ...s })) },
            }));
        },

        updateWeight(index, weight) {
            const slot = this.slots[index];
            if (!slot) return;
            const config = Alpine.store('config');
            const minWeight = config.defaultLorasMinWeight ?? -2;
            const maxWeight = config.defaultLorasMaxWeight ?? 5;
            const val = parseFloat(weight) || 0;
            slot.weight = Math.min(Math.max(val, minWeight), maxWeight);
            this._dispatchChanged(slot);
        },

        displayName(filename) {
            return filename.replace(/\.safetensors$/i, '').split('/').pop();
        },

        _dispatchChanged(slot) {
            window.dispatchEvent(new CustomEvent('lora-changed', {
                detail: {
                    index: slot.slotIndex,
                    filename: slot.filename,
                    weight: slot.weight,
                    color: slot.color,
                    triggerWords: slot.triggerWords,
                },
            }));
        },
    };
}
