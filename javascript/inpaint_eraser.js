// Brush-based eraser mode for the inpaint mask canvas (FWDF-189).
//
// Gradio 3.41.2's sketch widget has no eraser: tool='sketch' derives its
// mask from stroke ALPHA (gradio/components/image.py preprocess merges the
// alpha channel into an RGB mask) and Sketch.svelte composites every stroke
// with plain source-over, so painting with a black brush adds to the mask
// exactly like any other color -- there is no way to paint transparency
// through Gradio's own API at this version. gr.Eraser only exists in
// Gradio 4+.
//
// Instead we intercept pointer events on the sketch component ourselves
// when Erase mode is active, paint destination-out holes directly onto the
// visible mask canvas for live feedback, and mirror those holes into a
// persistent offscreen canvas. Gradio's own undo/redraw replay its internal
// stroke store (compiled Svelte in site-packages, not patchable from this
// repo) and would otherwise wipe our destination-out holes off the visible
// canvas, so the offscreen canvas -- not the visible one -- is the eraser's
// source of truth. On every pointerup we export it and write it into a
// CSS-hidden Textbox (#inpaint_eraser_data / #inpaint_mask_eraser_data),
// the same JS-to-Python hidden-trigger transport FWDF-186 established for
// #wildcard_scan_trigger; modules/async_worker.py decodes and subtracts it.
//
// Draw-mode strokes carve the same point back OUT of the offscreen eraser
// mask (without touching the visible canvas or intercepting the event --
// Gradio's own Sketch.svelte still owns painting Draw strokes), so a
// redrawn area is not silently re-erased by Python at generation time.
onUiLoaded(function() {
    var BRUSH_RADIUS_SELECTOR = "input[aria-label='Brush radius']";
    var DEFAULT_BRUSH_RADIUS = 20;

    function getComponentRoot(elemId) {
        return gradioApp().querySelector(elemId);
    }

    function getMaskCanvas(root) {
        return root ? root.querySelector("canvas[key='mask']") : null;
    }

    function getInterfaceCanvas(root) {
        return root ? root.querySelector("canvas[key='interface']") : null;
    }

    function getBrushRadius(root) {
        var input = root ? root.querySelector(BRUSH_RADIUS_SELECTOR) : null;
        var value = input ? parseFloat(input.value) : NaN;
        return Number.isFinite(value) && value > 0 ? value : DEFAULT_BRUSH_RADIUS;
    }

    function isErasingActive() {
        var modeInputs = gradioApp().querySelectorAll("#inpaint_brush_mode input[type='radio']");
        for (var i = 0; i < modeInputs.length; i++) {
            if (modeInputs[i].checked) {
                return modeInputs[i].value === 'Erase';
            }
        }
        return false;
    }

    // CSS pixel space (pointer events) to canvas pixel space (drawing/mask
    // data) -- w-full/h-auto canvases are almost never 1:1 with their CSS
    // size, so this scale is required at every zoom level, not just when
    // the user has zoomed in explicitly.
    function canvasPoint(canvas, event) {
        var rect = canvas.getBoundingClientRect();
        return {
            x: (event.clientX - rect.left) * (canvas.width / rect.width),
            y: (event.clientY - rect.top) * (canvas.height / rect.height)
        };
    }

    function writeToTextarea(elemId, value) {
        var textarea = gradioApp().querySelector(`#${elemId} textarea`);
        if (!textarea) {
            return;
        }
        textarea.value = value;
        textarea.dispatchEvent(new Event('input', {bubbles: true}));
    }

    // One eraser "session" per sketch component: its own offscreen mask
    // state and its own textarea transport, sized to the visible mask
    // canvas's current pixel dimensions. A stroke is tracked by mode
    // ('erase' | 'unerase' | null) rather than a boolean so pointerup can
    // always finish the stroke it actually started, regardless of what the
    // mode radio says at release time.
    function createEraserSession(textareaElemId) {
        var offscreen = document.createElement('canvas');
        var offscreenCtx = offscreen.getContext('2d');
        var activeStroke = null;

        function resetSession() {
            offscreenCtx.clearRect(0, 0, offscreen.width, offscreen.height);
            writeToTextarea(textareaElemId, '');
        }

        function ensureSized(maskCanvas) {
            if (offscreen.width !== maskCanvas.width || offscreen.height !== maskCanvas.height) {
                offscreen.width = maskCanvas.width;
                offscreen.height = maskCanvas.height;
                resetSession();
            }
        }

        function eraseAt(maskCanvas, point, radius) {
            var maskCtx = maskCanvas.getContext('2d');
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.beginPath();
            maskCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            maskCtx.fill();
            maskCtx.restore();

            offscreenCtx.save();
            offscreenCtx.globalCompositeOperation = 'source-over';
            offscreenCtx.fillStyle = '#fff';
            offscreenCtx.beginPath();
            offscreenCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            offscreenCtx.fill();
            offscreenCtx.restore();
        }

        // Draw mode: remove this point from the offscreen eraser mask only.
        // The visible mask canvas is Gradio's own responsibility here (plain
        // source-over white, Sketch.svelte) -- painting it ourselves would
        // fight that compositing.
        function uneraseAt(point, radius) {
            offscreenCtx.save();
            offscreenCtx.globalCompositeOperation = 'destination-out';
            offscreenCtx.beginPath();
            offscreenCtx.arc(point.x, point.y, radius, 0, Math.PI * 2);
            offscreenCtx.fill();
            offscreenCtx.restore();
        }

        function replayOntoMask(maskCanvas) {
            if (offscreen.width === 0 || offscreen.height === 0) {
                return;
            }
            var maskCtx = maskCanvas.getContext('2d');
            maskCtx.save();
            maskCtx.globalCompositeOperation = 'destination-out';
            maskCtx.drawImage(offscreen, 0, 0);
            maskCtx.restore();
        }

        function commit() {
            if (offscreen.width === 0 || offscreen.height === 0) {
                return;
            }
            writeToTextarea(textareaElemId, offscreen.toDataURL('image/png'));
        }

        return {
            ensureSized: ensureSized,
            reset: resetSession,
            eraseAt: eraseAt,
            uneraseAt: uneraseAt,
            replayOntoMask: replayOntoMask,
            commit: commit,
            activeStroke: function() { return activeStroke; },
            beginStroke: function(mode) { activeStroke = mode; },
            endStroke: function() { activeStroke = null; }
        };
    }

    // Draws (and clears) the eraser's own circular cursor on the interface
    // canvas. Gradio's built-in brush cursor loop never runs while erasing
    // because we stop pointer events from reaching Svelte's handlers.
    function drawCursor(interfaceCanvas, point, radius, previous) {
        var ctx = interfaceCanvas.getContext('2d');
        if (previous) {
            var pad = previous.radius + 2;
            ctx.clearRect(previous.x - pad, previous.y - pad, pad * 2, pad * 2);
        }
        ctx.save();
        ctx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();
    }

    function clearCursor(interfaceCanvas, previous) {
        if (!interfaceCanvas || !previous) {
            return;
        }
        var ctx = interfaceCanvas.getContext('2d');
        var pad = previous.radius + 2;
        ctx.clearRect(previous.x - pad, previous.y - pad, pad * 2, pad * 2);
    }

    // Handler factory: called once per sketch component (main canvas,
    // advanced masking canvas) rather than sharing one set of closures over
    // a loop variable -- each component needs its own eraser session, mode
    // toggle read, and DOM listeners.
    function attachEraser(canvasElemId, textareaElemId) {
        var componentRoot = getComponentRoot(canvasElemId);
        if (!componentRoot) {
            return;
        }

        var session = createEraserSession(textareaElemId);
        var lastCursor = null;

        function handlePointerDown(event) {
            if (event.button !== 0) {
                // Ignore right-/middle-click -- touch and pen primary
                // contacts report button 0 and are unaffected.
                return;
            }
            var maskCanvas = getMaskCanvas(componentRoot);
            if (!maskCanvas) {
                return;
            }
            session.ensureSized(maskCanvas);

            if (isErasingActive()) {
                event.preventDefault();
                event.stopPropagation();
                session.beginStroke('erase');
                session.eraseAt(maskCanvas, canvasPoint(maskCanvas, event), getBrushRadius(componentRoot));
            } else {
                // Draw mode: do NOT intercept the event -- Gradio must still
                // receive and paint the stroke itself. We only carve this
                // point back out of our own offscreen eraser mask.
                session.beginStroke('unerase');
                session.uneraseAt(canvasPoint(maskCanvas, event), getBrushRadius(componentRoot));
            }
        }

        function handlePointerMove(event) {
            var maskCanvas = getMaskCanvas(componentRoot);
            var interfaceCanvas = getInterfaceCanvas(componentRoot);
            if (!maskCanvas || !interfaceCanvas) {
                return;
            }

            if (isErasingActive()) {
                event.preventDefault();
                event.stopPropagation();
                var radius = getBrushRadius(componentRoot);
                var point = canvasPoint(interfaceCanvas, event);
                drawCursor(interfaceCanvas, point, radius, lastCursor);
                lastCursor = {x: point.x, y: point.y, radius: radius};

                if (session.activeStroke() === 'erase') {
                    session.eraseAt(maskCanvas, canvasPoint(maskCanvas, event), radius);
                }
            } else {
                clearCursor(interfaceCanvas, lastCursor);
                lastCursor = null;

                if (session.activeStroke() === 'unerase') {
                    session.uneraseAt(canvasPoint(maskCanvas, event), getBrushRadius(componentRoot));
                }
            }
        }

        // Bound to window (capture phase), not componentRoot: a stroke that
        // already punched holes must always finish and commit, even when
        // the pointer is released outside the component (drag off-canvas)
        // or the mode radio flips between pointerdown and pointerup. Gating
        // on the stroke recorded at pointerdown -- not a fresh
        // isErasingActive() read here -- makes that immune to a mode change
        // mid-stroke. pointercancel (touch/pen interruption) gets the same
        // treatment so a stuck "drawing" state can't survive it.
        function handlePointerUp(event) {
            var stroke = session.activeStroke();
            if (!stroke) {
                return;
            }
            if (stroke === 'erase') {
                event.preventDefault();
                event.stopPropagation();
            }
            session.endStroke();
            session.commit();
        }

        function handlePointerLeave() {
            clearCursor(getInterfaceCanvas(componentRoot), lastCursor);
            lastCursor = null;
        }

        function resyncAfterRedraw() {
            var maskCanvas = getMaskCanvas(componentRoot);
            if (maskCanvas) {
                session.replayOntoMask(maskCanvas);
            }
        }

        // Capture-phase so we intercept before Svelte's own bubble-phase
        // interface-canvas handlers see the event.
        componentRoot.addEventListener('pointerdown', handlePointerDown, true);
        componentRoot.addEventListener('pointermove', handlePointerMove, true);
        window.addEventListener('pointerup', handlePointerUp, true);
        window.addEventListener('pointercancel', handlePointerUp, true);
        componentRoot.addEventListener('pointerleave', handlePointerLeave, true);

        // Delegated (not bound to Undo/Remove Image at attach time): Gradio
        // only renders those buttons after an image is loaded, so they do
        // not exist yet when attachEraser() runs at onUiLoaded, and Svelte
        // tears down/recreates the file input on every clear+reupload. A
        // listener on componentRoot survives all of that.
        componentRoot.addEventListener('click', function(event) {
            var btn = event.target.closest ? event.target.closest('button[aria-label]') : null;
            if (!btn) {
                return;
            }
            var label = btn.getAttribute('aria-label');
            if (label === 'Undo') {
                requestAnimationFrame(resyncAfterRedraw);
            } else if (label === 'Remove Image') {
                session.reset();
            }
        }, true);

        componentRoot.addEventListener('change', function(event) {
            if (event.target && event.target.matches && event.target.matches("input[type='file']")) {
                session.reset();
            }
        }, true);

        // Gradio-originated canvas repaints come in two shapes, distinguished
        // by mutation type (observed the same way zoom.js's own observer
        // keys off style mutations for resize/redraw):
        //  - attribute 'style' change on the SAME <canvas> node: an in-place
        //    repaint (undo, resize) -- re-apply our offscreen holes.
        //  - childList change that swaps in a different mask <canvas> node,
        //    or a same-size image replacing the previous one via drag-and-
        //    drop (which fires no 'change' event on the file input): treat
        //    as a new image and reset the eraser instead of resyncing stale
        //    holes onto it.
        var lastMaskCanvas = getMaskCanvas(componentRoot);
        var observer = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                var mutation = mutations[i];
                if (mutation.type === 'attributes' && mutation.attributeName === 'style' &&
                        mutation.target.tagName && mutation.target.tagName.toLowerCase() === 'canvas') {
                    requestAnimationFrame(resyncAfterRedraw);
                    return;
                }
                if (mutation.type === 'childList') {
                    var currentMaskCanvas = getMaskCanvas(componentRoot);
                    if (currentMaskCanvas !== lastMaskCanvas) {
                        lastMaskCanvas = currentMaskCanvas;
                        session.reset();
                        return;
                    }
                }
            }
        });
        observer.observe(componentRoot, {attributes: true, childList: true, subtree: true});
    }

    attachEraser('#inpaint_canvas', 'inpaint_eraser_data');
    attachEraser('#inpaint_mask_canvas', 'inpaint_mask_eraser_data');
});
