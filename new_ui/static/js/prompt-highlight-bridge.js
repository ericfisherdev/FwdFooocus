/**
 * Prompt-Highlight Bridge
 *
 * Connects the trigger highlighting system to the prompt editor.
 * Listens for lora-changed/lora-removed events, maintains active LoRA
 * list, and re-renders highlights in the contenteditable editor.
 *
 * This runs after both prompt-editor.js and trigger-highlight.js load.
 * Registered via alpine:init so it has access to Alpine stores.
 */

document.addEventListener('alpine:init', () => {

    let activeLoRAs = [];
    let highlightTimeout = null;

    function getEditor(mode) {
        return document.getElementById(`prompt-editor-${mode}`);
    }

    /**
     * Save and restore cursor position across innerHTML updates.
     * Uses character offset via TreeWalker.
     */
    function saveCursorOffset(el) {
        const sel = window.getSelection();
        if (!sel.rangeCount || !el.contains(sel.anchorNode)) return null;

        const range = sel.getRangeAt(0);
        const preRange = document.createRange();
        preRange.setStart(el, 0);
        preRange.setEnd(range.startContainer, range.startOffset);
        return preRange.toString().length;
    }

    function restoreCursorOffset(el, offset) {
        if (offset === null || offset === undefined) return;

        const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
        let charCount = 0;
        let node;

        while ((node = walker.nextNode())) {
            const nodeLen = node.textContent.length;
            if (charCount + nodeLen >= offset) {
                const range = document.createRange();
                range.setStart(node, offset - charCount);
                range.collapse(true);
                const sel = window.getSelection();
                sel.removeAllRanges();
                sel.addRange(range);
                return;
            }
            charCount += nodeLen;
        }
    }

    function applyHighlights(mode) {
        const el = getEditor(mode);
        if (!el) return;

        // Don't re-render while user is selecting text
        const sel = window.getSelection();
        if (sel.rangeCount > 0 && !sel.isCollapsed && el.contains(sel.anchorNode)) {
            return;
        }

        const plainText = el.innerText || '';
        const html = TriggerHighlight.highlightTriggers(plainText, activeLoRAs);

        // Only update if HTML actually changed (avoid cursor flicker)
        if (el.innerHTML !== html) {
            const offset = saveCursorOffset(el);
            el.innerHTML = html;
            restoreCursorOffset(el, offset);
        }
    }

    function scheduleHighlight(delay = 150) {
        clearTimeout(highlightTimeout);
        highlightTimeout = setTimeout(() => {
            applyHighlights('positive');
        }, delay);
    }

    // Listen for LoRA changes
    window.addEventListener('lora-changed', (e) => {
        const detail = e.detail;
        const existing = activeLoRAs.findIndex(l => l.slotIndex === detail.index);
        const entry = {
            slotIndex: detail.index,
            filename: detail.filename,
            color: detail.color,
            triggerWords: detail.triggerWords || [],
        };

        if (existing >= 0) {
            activeLoRAs[existing] = entry;
        } else {
            activeLoRAs.push(entry);
        }

        // Sort by slot index so first LoRA wins color conflicts
        activeLoRAs.sort((a, b) => a.slotIndex - b.slotIndex);
        scheduleHighlight();

        // Dispatch for missing trigger detection (FWDF-105)
        window.dispatchEvent(new CustomEvent('active-loras-updated', {
            detail: { activeLoRAs: [...activeLoRAs] },
        }));
    });

    window.addEventListener('lora-removed', (e) => {
        const detail = e.detail;
        activeLoRAs = activeLoRAs.filter(l => l.slotIndex !== detail.index);
        scheduleHighlight();

        window.dispatchEvent(new CustomEvent('active-loras-updated', {
            detail: { activeLoRAs: [...activeLoRAs] },
        }));
    });

    // Re-highlight when prompt text changes (already debounced upstream)
    window.addEventListener('prompt-changed', (e) => {
        if (e.detail.mode === 'positive' && activeLoRAs.length > 0) {
            scheduleHighlight(0);
        }
    });
});
