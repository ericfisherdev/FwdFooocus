/**
 * Gallery Comparison Mode — extends the Gallery component.
 *
 * Side-by-side two-image comparison with metadata diff.
 * Activated when exactly 2 images are selected in the gallery.
 *
 * Spec: component-spec-gallery.md (Comparison View section)
 */

const GalleryCompare = (() => {

    /**
     * Compute metadata diff between two images.
     * Returns only keys where the values differ.
     *
     * @param {object} a — image A metadata
     * @param {object} b — image B metadata
     * @returns {Array<{key: string, valueA: any, valueB: any}>}
     */
    function computeDiff(a, b) {
        const keys = [
            'seed', 'prompt', 'negative_prompt', 'cfg_scale',
            'sampler', 'scheduler', 'steps', 'base_model',
            'dimensions', 'width', 'height',
        ];

        return keys
            .map(key => {
                const valueA = a?.[key] ?? null;
                const valueB = b?.[key] ?? null;
                return {
                    key,
                    valueA,
                    valueB,
                    differs: JSON.stringify(valueA) !== JSON.stringify(valueB),
                };
            })
            .filter(row => row.differs);
    }

    /**
     * Format a metadata key for display.
     */
    function formatKey(key) {
        const labels = {
            seed: 'Seed',
            prompt: 'Prompt',
            negative_prompt: 'Negative',
            cfg_scale: 'CFG',
            sampler: 'Sampler',
            scheduler: 'Scheduler',
            steps: 'Steps',
            base_model: 'Model',
            dimensions: 'Dimensions',
            width: 'Width',
            height: 'Height',
        };
        return labels[key] || key;
    }

    /**
     * Format a metadata value for display (truncate long strings).
     */
    function formatValue(val) {
        if (val === null || val === undefined) return '—';
        const str = String(val);
        return str.length > 40 ? str.slice(0, 40) + '…' : str;
    }

    return { computeDiff, formatKey, formatValue };
})();

window.GalleryCompare = GalleryCompare;
