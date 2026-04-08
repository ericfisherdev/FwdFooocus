/**
 * Trigger Word Highlighting — extends the Prompt Editor.
 *
 * Scans prompt text for trigger words from active LoRAs and wraps
 * matches in colored <mark> elements. Case-insensitive, whole-word.
 *
 * Spec: component-spec-prompt-editor.md
 */

const TriggerHighlight = (() => {

    /**
     * Build highlighted HTML from plain text and active LoRA data.
     *
     * @param {string} plainText — raw prompt text
     * @param {Array<{color: string, triggerWords: string[]}>} activeLoRAs
     * @returns {string} HTML with <mark> spans for trigger matches
     */
    function highlightTriggers(plainText, activeLoRAs) {
        if (!activeLoRAs || !activeLoRAs.length || !plainText) {
            return escapeHTML(plainText || '');
        }

        // Build map: trigger word (lowercase) → { color, word (original casing) }
        const triggerMap = new Map();
        for (const lora of activeLoRAs) {
            if (!lora.triggerWords) continue;
            for (const word of lora.triggerWords) {
                const lower = word.toLowerCase();
                if (!triggerMap.has(lower)) {
                    triggerMap.set(lower, { color: lora.color, word });
                }
            }
        }

        if (triggerMap.size === 0) return escapeHTML(plainText);

        // Build regex matching all trigger words (whole word, case-insensitive)
        const escaped = [...triggerMap.keys()].map(w =>
            w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        );
        const regex = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');

        // Replace matches with <mark> spans
        return escapeHTML(plainText).replace(regex, (match) => {
            const entry = triggerMap.get(match.toLowerCase());
            if (!entry) return match;
            const bg = hexToRGBA(entry.color, 0.2);
            const border = resolveColor(entry.color);
            return `<mark style="background:${bg};border-bottom:2px solid ${border};border-radius:2px;padding:0 1px;" title="${entry.word} — trigger for LoRA">${match}</mark>`;
        });
    }

    /**
     * Find trigger words from active LoRAs that are missing from the prompt.
     *
     * @param {string} plainText
     * @param {Array<{filename: string, color: string, triggerWords: string[]}>} activeLoRAs
     * @returns {Array<{loraFilename: string, loraColor: string, triggerWord: string}>}
     */
    function findMissingTriggers(plainText, activeLoRAs) {
        if (!activeLoRAs || !activeLoRAs.length) return [];

        const textLower = (plainText || '').toLowerCase();
        const missing = [];

        for (const lora of activeLoRAs) {
            if (!lora.triggerWords || lora.triggerWords.length === 0) continue;

            const hasAny = lora.triggerWords.some(word => {
                const pattern = new RegExp(`\\b${word.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
                return pattern.test(textLower);
            });

            if (!hasAny) {
                missing.push({
                    loraFilename: lora.filename,
                    loraColor: lora.color,
                    triggerWord: lora.triggerWords[0], // first/most common
                });
            }
        }

        return missing;
    }

    /* -- Helpers --------------------------------------------------- */

    function escapeHTML(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /**
     * Resolve a CSS custom property value (e.g. "var(--lora-color-1)")
     * to a computed hex/rgb value.
     */
    function resolveColor(cssValue) {
        if (!cssValue.startsWith('var(')) return cssValue;
        const prop = cssValue.slice(4, -1);
        return getComputedStyle(document.documentElement).getPropertyValue(prop).trim() || cssValue;
    }

    /**
     * Convert a CSS color (hex or var()) to rgba at given opacity.
     */
    function hexToRGBA(cssValue, opacity) {
        const resolved = resolveColor(cssValue);
        // If it's already rgb/rgba, extract values
        const rgbMatch = resolved.match(/^#([0-9a-f]{6})$/i);
        if (rgbMatch) {
            const r = parseInt(rgbMatch[1].slice(0, 2), 16);
            const g = parseInt(rgbMatch[1].slice(2, 4), 16);
            const b = parseInt(rgbMatch[1].slice(4, 6), 16);
            return `rgba(${r},${g},${b},${opacity})`;
        }
        // Fallback: use the color with opacity filter
        return resolved;
    }

    return { highlightTriggers, findMissingTriggers, escapeHTML };

})();

// Expose globally
window.TriggerHighlight = TriggerHighlight;
