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

        // Build map: trigger word (lowercase) → { color, word (original casing), loraName }
        const triggerMap = new Map();
        for (const lora of activeLoRAs) {
            if (!lora.triggerWords) continue;
            const loraName = lora.name || lora.filename || 'LoRA';
            for (const word of lora.triggerWords) {
                const normalized = (word || '').trim();
                if (!normalized) continue;
                const lower = normalized.toLowerCase();
                if (!triggerMap.has(lower)) {
                    triggerMap.set(lower, { color: lora.color, word: normalized, loraName });
                }
            }
        }

        if (triggerMap.size === 0) return escapeHTML(plainText);

        // Build regex matching all trigger words (whole word, case-insensitive)
        const escaped = [...triggerMap.keys()].map(w =>
            w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
        );
        const regex = new RegExp(`\\b(${escaped.join('|')})\\b`, 'gi');

        // Apply regex to plain text first, then escape non-matched segments
        let result = '';
        let lastIndex = 0;
        let m;
        while ((m = regex.exec(plainText)) !== null) {
            const entry = triggerMap.get(m[0].toLowerCase());
            if (!entry) {
                result += escapeHTML(plainText.slice(lastIndex, m.index + m[0].length));
                lastIndex = m.index + m[0].length;
                continue;
            }
            const bg = hexToRGBA(entry.color, 0.2);
            const border = resolveColor(entry.color);
            const safeWord = escapeAttribute(entry.word);
            const safeLoraName = escapeAttribute(entry.loraName);
            result += escapeHTML(plainText.slice(lastIndex, m.index));
            result += `<mark style="background:${bg};border-bottom:2px solid ${border};border-radius:2px;padding:0 1px;" title="${safeWord} — trigger for ${safeLoraName}">${escapeHTML(m[0])}</mark>`;
            lastIndex = m.index + m[0].length;
        }
        result += escapeHTML(plainText.slice(lastIndex));
        return result;
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
                const normalized = (word || '').trim();
                if (!normalized) return false;
                const pattern = new RegExp(`\\b${normalized.toLowerCase().replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`);
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

    function escapeAttribute(text) {
        return String(text)
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    /**
     * Resolve a CSS custom property value (e.g. "var(--lora-color-1)")
     * to a computed hex/rgb value.
     */
    function resolveColor(cssValue) {
        if (!cssValue.startsWith('var(')) return cssValue;
        const inner = cssValue.slice(4, -1);
        const commaIdx = inner.indexOf(',');
        const prop = commaIdx >= 0 ? inner.slice(0, commaIdx).trim() : inner.trim();
        const fallback = commaIdx >= 0 ? inner.slice(commaIdx + 1).trim() : cssValue;
        return getComputedStyle(document.documentElement).getPropertyValue(prop).trim() || fallback;
    }

    /**
     * Convert a CSS color (hex or var()) to rgba at given opacity.
     */
    function hexToRGBA(cssValue, opacity) {
        const resolved = resolveColor(cssValue);
        const hexMatch = resolved.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
        if (hexMatch) {
            const hex = hexMatch[1].length === 3
                ? hexMatch[1].split('').map(ch => ch + ch).join('')
                : hexMatch[1];
            const r = parseInt(hex.slice(0, 2), 16);
            const g = parseInt(hex.slice(2, 4), 16);
            const b = parseInt(hex.slice(4, 6), 16);
            return `rgba(${r},${g},${b},${opacity})`;
        }
        const rgbMatch = resolved.match(/^rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})/i);
        if (rgbMatch) {
            return `rgba(${Number(rgbMatch[1])},${Number(rgbMatch[2])},${Number(rgbMatch[3])},${opacity})`;
        }
        return resolved;
    }

    return { highlightTriggers, findMissingTriggers, escapeHTML };

})();

// Expose globally
window.TriggerHighlight = TriggerHighlight;
