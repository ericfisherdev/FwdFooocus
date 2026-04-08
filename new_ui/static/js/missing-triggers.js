/**
 * Missing Trigger Detection Bar — Alpine.js component.
 *
 * Shows warnings below the prompt editor when active LoRAs have
 * trigger words not present in the prompt text.
 *
 * Depends on: trigger-highlight.js (TriggerHighlight.findMissingTriggers)
 * Listens for: active-loras-updated, prompt-changed
 *
 * Spec: component-spec-prompt-editor.md (Missing Trigger Bar section)
 */

function missingTriggers() {
    return {
        missing: [],
        activeLoRAs: [],
        promptText: '',

        init() {
            window.addEventListener('active-loras-updated', (e) => {
                this.activeLoRAs = e?.detail?.activeLoRAs || [];
                this._recalculate();
            });

            window.addEventListener('prompt-changed', (e) => {
                if (e?.detail?.mode === 'positive') {
                    this.promptText = e?.detail?.text || '';
                    this._recalculate();
                }
            });
        },

        get hasMissing() {
            return this.missing.length > 0;
        },

        insertTrigger(triggerWord, loraFilename) {
            // Dispatch to the prompt editor's insertAtCursor
            // The prompt editor listens for this custom event
            window.dispatchEvent(new CustomEvent('insert-trigger-word', {
                detail: { word: triggerWord, loraFilename, mode: 'positive' },
            }));

            // Optimistically remove from missing list — filter by both
            // triggerWord AND loraFilename so unrelated rows sharing the
            // same trigger word are not accidentally removed.
            this.missing = this.missing.filter(
                m => !(m.triggerWord === triggerWord && m.loraFilename === loraFilename)
            );
        },

        displayName(filename) {
            return (filename || '').replace(/\.safetensors$/i, '').split('/').pop();
        },

        _recalculate() {
            if (!window.TriggerHighlight) {
                this.missing = [];
                return;
            }
            this.missing = TriggerHighlight.findMissingTriggers(
                this.promptText,
                this.activeLoRAs
            );
        },
    };
}
