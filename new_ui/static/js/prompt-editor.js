/**
 * Prompt Editor — Alpine.js component.
 *
 * contenteditable div supporting trigger word highlighting (wired in FWDF-104).
 * Plain text is the source of truth; HTML is derived for display only.
 *
 * Spec: component-spec-prompt-editor.md
 *
 * Usage:
 *   <div x-data="promptEditor({ mode: 'positive' })"> ... </div>
 */

function promptEditor({ mode = 'positive', value = '' } = {}) {
    return {
        mode,
        plainText: value,
        focused: false,

        /* -- Computed -------------------------------------------------- */

        get label() {
            return this.mode === 'positive' ? 'Prompt' : 'Negative Prompt';
        },

        get placeholder() {
            return this.mode === 'positive'
                ? 'Type prompt here or paste parameters.'
                : 'Describe what you do not want to see.';
        },

        get editorId() {
            return `prompt-editor-${this.mode}`;
        },

        get isEmpty() {
            return this.plainText.trim().length === 0;
        },

        /* -- Lifecycle ------------------------------------------------- */

        init() {
            const el = this._editorEl();
            if (el && this.plainText) {
                el.textContent = this.plainText;
            }
        },

        /* -- Event handlers -------------------------------------------- */

        onInput() {
            const el = this._editorEl();
            if (!el) return;
            this.plainText = el.innerText || '';
            this.$dispatch('prompt-changed', { mode: this.mode, text: this.plainText });
        },

        onPaste(event) {
            event.preventDefault();
            const text = event.clipboardData.getData('text/plain');
            document.execCommand('insertText', false, text);
        },

        onFocus() {
            this.focused = true;
        },

        onBlur() {
            this.focused = false;
        },

        onKeydown(event) {
            if (event.key === 'Escape') {
                this._editorEl()?.blur();
            }
        },

        /* -- Public methods -------------------------------------------- */

        setText(text) {
            this.plainText = text;
            const el = this._editorEl();
            if (el) {
                el.textContent = text;
            }
            this.$dispatch('prompt-changed', { mode: this.mode, text: this.plainText });
        },

        insertAtCursor(text) {
            const el = this._editorEl();
            if (!el) return;

            el.focus();
            const sel = window.getSelection();
            if (sel.rangeCount > 0) {
                const range = sel.getRangeAt(0);
                // Add space before if not at start and previous char isn't space
                const before = range.startOffset > 0 ? ', ' : '';
                const node = document.createTextNode(before + text);
                range.deleteContents();
                range.insertNode(node);
                range.setStartAfter(node);
                range.setEndAfter(node);
                sel.removeAllRanges();
                sel.addRange(range);
            } else {
                el.textContent += (el.textContent.length > 0 ? ', ' : '') + text;
            }

            this.plainText = el.innerText || '';
            this.$dispatch('prompt-changed', { mode: this.mode, text: this.plainText });
        },

        /* -- Private --------------------------------------------------- */

        _editorEl() {
            return document.getElementById(this.editorId);
        },
    };
}
