// Per-thumbnail multi-select checkboxes for the final gallery (FWDF-191).
//
// Gradio 3.41.2's Gallery renders the grid as
// div.grid-wrap > div.grid-container > button.thumbnail-item.thumbnail-lg,
// one per image, in value order (js/gallery/static/Gallery.svelte in the
// pinned wheel); preview mode adds div.preview with a strip of
// button.thumbnail-item.thumbnail-small, also in value order. There is no
// built-in per-thumbnail checkbox, so this overlays one absolutely
// positioned checkbox per thumbnail button and tracks the checked set
// itself. Checkbox index == DOM order == gallery value order, which is the
// same order webui.py's gallery_paths_state is filled in, so an index here
// maps directly to a gallery_paths_state entry.
//
// The Svelte each-blocks that render the thumbnails are NOT keyed, so
// button elements are REUSED across a gallery value change instead of
// being replaced -- the same identity hazard FWDF-189's inpaint_eraser.js
// resyncAfterRedraw was written against. An overlay attached to button i
// can silently end up positioned over a different image after a redraw.
// To guard against that, every overlay stamps the img src it was built
// for, and any observed mutation batch that leaves a stamped src
// mismatched against its button's live thumbnail forces a full rebuild of
// every overlay plus a checked-state reset -- never a partial resync that
// could carry stale checked state onto a different image.
//
// Checked state reaches Python through a CSS-hidden gr.Textbox
// (#gallery_checked_data), written as a JSON array of checked indices and
// dispatched as a bubbling 'input' event -- the same JS-to-Python hidden
// transport javascript/inpaint_eraser.js uses for #inpaint_eraser_data
// (see css/style.css's #inpaint_eraser_data rule for the CSS-hidden half of
// that precedent).
onUiLoaded(function() {
    var GALLERY_ROOT_SELECTOR = '#final_gallery';
    var TRANSPORT_ELEM_ID = 'gallery_checked_data';
    var THUMBNAIL_SELECTOR = 'button.thumbnail-item';

    function getGalleryRoot() {
        return gradioApp().querySelector(GALLERY_ROOT_SELECTOR);
    }

    function getThumbnailImgSrc(thumbnailButton) {
        var img = thumbnailButton.querySelector('img');
        return img ? img.src : '';
    }

    function writeCheckedIndices(indices) {
        var textarea = gradioApp().querySelector(`#${TRANSPORT_ELEM_ID} textarea`);
        if (!textarea) {
            return;
        }
        textarea.value = JSON.stringify(indices);
        textarea.dispatchEvent(new Event('input', {bubbles: true}));
    }

    // One overlay session per #final_gallery root. Holds the checked-index
    // set and the per-thumbnail-button overlay state (button -> stamped img
    // src) so a later mutation batch can detect drift for any button, in
    // either the grid or the preview strip, against the src it was built
    // for.
    function createOverlaySession(galleryRoot) {
        var checkedIndices = {}; // index -> true, sparse set via plain object
        // WeakMap so overlay bookkeeping never leaks once Gradio actually
        // discards a button node (as opposed to reusing it in place).
        var stampedSrcByButton = new WeakMap();

        function checkedIndexArray() {
            return Object.keys(checkedIndices).map(Number).sort(function(a, b) { return a - b; });
        }

        function isChecked(index) {
            return !!checkedIndices[index];
        }

        function setChecked(index, checked) {
            if (checked) {
                checkedIndices[index] = true;
            } else {
                delete checkedIndices[index];
            }
        }

        function syncOverlayCheckedState(overlayInput, index) {
            overlayInput.checked = isChecked(index);
        }

        function buildOverlay(thumbnailButton, index) {
            var overlay = document.createElement('label');
            overlay.className = 'gallery-checkbox-overlay';

            var input = document.createElement('input');
            input.type = 'checkbox';
            input.className = 'gallery-checkbox-overlay-input';
            syncOverlayCheckedState(input, index);

            // stopPropagation on both click and pointerdown so toggling a
            // checkbox never reaches the thumbnail button's own click
            // handler (which would open the preview / change Gradio's
            // internal selection) -- mirrors inpaint_eraser.js's pattern of
            // intercepting the event at the point it must not bubble from.
            function onToggle(event) {
                event.stopPropagation();
                setChecked(index, input.checked);
                writeCheckedIndices(checkedIndexArray());
            }
            input.addEventListener('pointerdown', function(event) { event.stopPropagation(); });
            input.addEventListener('click', onToggle);

            overlay.appendChild(input);
            thumbnailButton.style.position = thumbnailButton.style.position || 'relative';
            thumbnailButton.appendChild(overlay);
            stampedSrcByButton.set(thumbnailButton, getThumbnailImgSrc(thumbnailButton));
            return input;
        }

        function removeAllOverlays() {
            var overlays = galleryRoot.querySelectorAll('.gallery-checkbox-overlay');
            for (var i = 0; i < overlays.length; i++) {
                overlays[i].remove();
            }
        }

        function resetCheckedState() {
            checkedIndices = {};
            writeCheckedIndices([]);
        }

        // Rebuilds every overlay from scratch across both the grid
        // (thumbnail-lg) and the preview strip (thumbnail-small), which
        // share the same underlying gallery value order and therefore the
        // same index space -- toggling either keeps them in sync because
        // they read/write the same checkedIndices set.
        function rebuildAllOverlays() {
            removeAllOverlays();
            var thumbnailButtons = galleryRoot.querySelectorAll(THUMBNAIL_SELECTOR);
            for (var containerIndex = 0; containerIndex < thumbnailButtons.length; containerIndex++) {
                var button = thumbnailButtons[containerIndex];
                var index = indexWithinOwnContainer(button, thumbnailButtons);
                buildOverlay(button, index);
            }
        }

        // Grid and preview-strip thumbnails are separate containers that
        // each enumerate the full gallery value in order -- index must be
        // computed within the button's own container (grid vs. strip), not
        // across the combined NodeList, or a preview-strip button would get
        // an index offset by however many grid thumbnails precede it in
        // DOM order.
        function indexWithinOwnContainer(button, allThumbnailButtons) {
            var container = button.parentElement;
            var siblingsInContainer = container ? container.querySelectorAll(THUMBNAIL_SELECTOR) : [button];
            for (var i = 0; i < siblingsInContainer.length; i++) {
                if (siblingsInContainer[i] === button) {
                    return i;
                }
            }
            return 0;
        }

        // Detects drift: any live thumbnail button whose current img src no
        // longer matches the src its overlay was stamped against, or any
        // thumbnail button with no overlay at all (a genuinely new button),
        // or an overlay whose button is no longer in the DOM (fewer
        // thumbnails than before). Any of these forces a full rebuild --
        // never a partial resync -- because the reused-button identity race
        // means a partial resync could stamp stale checked state onto a
        // different image (the exact FWDF-189 lesson this mirrors).
        function hasDrifted() {
            var thumbnailButtons = galleryRoot.querySelectorAll(THUMBNAIL_SELECTOR);
            var overlaidCount = galleryRoot.querySelectorAll('.gallery-checkbox-overlay').length;
            if (thumbnailButtons.length !== overlaidCount) {
                return true;
            }
            for (var i = 0; i < thumbnailButtons.length; i++) {
                var button = thumbnailButtons[i];
                var stampedSrc = stampedSrcByButton.get(button);
                if (stampedSrc === undefined || stampedSrc !== getThumbnailImgSrc(button)) {
                    return true;
                }
            }
            return false;
        }

        function reconcile() {
            if (hasDrifted()) {
                resetCheckedState();
                rebuildAllOverlays();
            }
        }

        return {
            reconcile: reconcile,
            rebuildAllOverlays: rebuildAllOverlays,
            resetCheckedState: resetCheckedState
        };
    }

    var galleryRoot = getGalleryRoot();
    if (!galleryRoot) {
        return;
    }

    var session = createOverlaySession(galleryRoot);
    session.rebuildAllOverlays();

    // Coalesce bursts of mutations (Gradio can emit several DOM mutations
    // per redraw) into a single reconcile per animation frame, mirroring
    // inpaint_eraser.js's requestAnimationFrame coalescing.
    var reconcileScheduled = false;
    function scheduleReconcile() {
        if (reconcileScheduled) {
            return;
        }
        reconcileScheduled = true;
        requestAnimationFrame(function() {
            reconcileScheduled = false;
            session.reconcile();
        });
    }

    var observer = new MutationObserver(function() {
        scheduleReconcile();
    });
    observer.observe(galleryRoot, {attributes: true, childList: true, subtree: true});
});
