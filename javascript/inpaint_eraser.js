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
    // canvas's current pixel dimensions.
    function createEraserSession(textareaElemId) {
        var offscreen = document.createElement('canvas');
        var offscreenCtx = offscreen.getContext('2d');
        var drawing = false;

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
            replayOntoMask: replayOntoMask,
            commit: commit,
            isDrawing: function() { return drawing; },
            setDrawing: function(value) { drawing = value; }
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
            var maskCanvas = getMaskCanvas(componentRoot);
            if (!isErasingActive() || !maskCanvas) {
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            session.ensureSized(maskCanvas);
            session.setDrawing(true);
            var point = canvasPoint(maskCanvas, event);
            session.eraseAt(maskCanvas, point, getBrushRadius(componentRoot));
        }

        function handlePointerMove(event) {
            var maskCanvas = getMaskCanvas(componentRoot);
            var interfaceCanvas = getInterfaceCanvas(componentRoot);
            if (!isErasingActive() || !maskCanvas || !interfaceCanvas) {
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            var radius = getBrushRadius(componentRoot);
            var point = canvasPoint(interfaceCanvas, event);
            drawCursor(interfaceCanvas, point, radius, lastCursor);
            lastCursor = {x: point.x, y: point.y, radius: radius};

            if (session.isDrawing()) {
                session.eraseAt(maskCanvas, canvasPoint(maskCanvas, event), radius);
            }
        }

        function handlePointerUp(event) {
            if (!isErasingActive() || !session.isDrawing()) {
                return;
            }
            event.preventDefault();
            event.stopPropagation();
            session.setDrawing(false);
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
        componentRoot.addEventListener('pointerup', handlePointerUp, true);
        componentRoot.addEventListener('pointerleave', handlePointerLeave, true);

        var undoBtn = componentRoot.querySelector("button[aria-label='Undo']");
        if (undoBtn) {
            undoBtn.addEventListener('click', function() {
                requestAnimationFrame(resyncAfterRedraw);
            });
        }

        var clearBtn = componentRoot.querySelector("button[aria-label='Remove Image']");
        if (clearBtn) {
            clearBtn.addEventListener('click', function() {
                session.reset();
            });
        }

        var fileInput = componentRoot.querySelector("input[type='file']");
        if (fileInput) {
            fileInput.addEventListener('change', function() {
                session.reset();
            });
        }

        // Fallback: any Gradio-originated canvas repaint (a style mutation
        // on a <canvas>, the same signal zoom.js's observer keys off of for
        // resize/redraw) gets our destination-out holes re-applied a frame
        // later, after Gradio's own repaint has settled.
        var observer = new MutationObserver(function(mutations) {
            for (var i = 0; i < mutations.length; i++) {
                var mutation = mutations[i];
                if (mutation.type === 'attributes' && mutation.attributeName === 'style' &&
                        mutation.target.tagName && mutation.target.tagName.toLowerCase() === 'canvas') {
                    requestAnimationFrame(resyncAfterRedraw);
                    return;
                }
            }
        });
        observer.observe(componentRoot, {attributes: true, childList: true, subtree: true});
    }

    attachEraser('#inpaint_canvas', 'inpaint_eraser_data');
    attachEraser('#inpaint_mask_canvas', 'inpaint_mask_eraser_data');
});
