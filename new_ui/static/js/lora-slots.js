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
            this.maxSlots = Alpine.store('config').defaultMaxLoraNumber || 5;

            // Initialize from config defaults
            const defaults = Alpine.store('config').defaultLoras || [];
            defaults.forEach(([filename, weight]) => {
                if (filename && filename !== 'None') {
                    this.addLora(filename, weight, true);
                }
            });
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

            const colorIndex = this.slots.length % LORA_COLORS.length;
            const slot = {
                filename,
                weight,
                color: LORA_COLORS[colorIndex],
                triggerWords: [],
                slotIndex: this.slots.length,
            };

            // Fetch trigger words
            try {
                const resp = await fetch(`/api/lora-trigger-words?filename=${encodeURIComponent(filename)}`);
                if (resp.ok) {
                    const data = await resp.json();
                    slot.triggerWords = data.trigger_words || [];
                }
            } catch { /* non-critical */ }

            this.slots.push(slot);

            if (!skipEvent) {
                this._dispatchChanged(slot);
            }
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
        },

        updateWeight(index, weight) {
            const slot = this.slots[index];
            if (!slot) return;
            slot.weight = parseFloat(weight) || 0;
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
