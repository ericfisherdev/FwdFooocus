/**
 * Split-pane resize component.
 *
 * Handles mouse drag on the separator between compose and gallery panes.
 * Persists the width ratio to localStorage for session continuity.
 * Respects min/max constraints from CSS (360px min, 50vw max).
 *
 * Usage (on the handle element):
 *   x-data="splitPane()"
 *   @mousedown="startDrag"
 *   @keydown="onKeyDown"
 */

const STORAGE_KEY = 'fwd_compose_width';
const MIN_COMPOSE = 360;
const MIN_GALLERY = 320;
const KEYBOARD_STEP = 20;

function splitPane() {
    return {
        dragging: false,

        init() {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                this._applyWidth(parseInt(saved, 10));
            }
        },

        startDrag(event) {
            event.preventDefault();
            this.dragging = true;

            const onMove = (e) => {
                if (!this.dragging) return;
                this._applyWidth(e.clientX);
            };

            const onUp = () => {
                this.dragging = false;
                document.removeEventListener('mousemove', onMove);
                document.removeEventListener('mouseup', onUp);
                document.body.style.cursor = '';
                document.body.style.userSelect = '';
            };

            document.addEventListener('mousemove', onMove);
            document.addEventListener('mouseup', onUp);
            document.body.style.cursor = 'col-resize';
            document.body.style.userSelect = 'none';
        },

        onKeyDown(event) {
            const compose = document.querySelector('.split-pane__compose');
            if (!compose) return;

            const current = compose.getBoundingClientRect().width;
            let next = current;

            if (event.key === 'ArrowLeft') {
                next = current - KEYBOARD_STEP;
                event.preventDefault();
            } else if (event.key === 'ArrowRight') {
                next = current + KEYBOARD_STEP;
                event.preventDefault();
            } else {
                return;
            }

            this._applyWidth(next);
        },

        _applyWidth(px) {
            const container = document.querySelector('.split-pane');
            if (!container) return;

            const totalWidth = container.getBoundingClientRect().width;
            const handleWidth = 6;
            const maxCompose = totalWidth * 0.5;
            const maxByGallery = totalWidth - handleWidth - MIN_GALLERY;

            const clamped = Math.round(
                Math.max(MIN_COMPOSE, Math.min(px, maxCompose, maxByGallery))
            );

            document.querySelector('.split-pane__compose').style.width = clamped + 'px';
            localStorage.setItem(STORAGE_KEY, clamped);
        },
    };
}
