// Debounced wildcard scan trigger (FWDF-186).
//
// After the user pauses typing in the prompt textarea for 3 seconds, click
// the CSS-hidden #wildcard_scan_trigger Gradio button so its .click handler
// (webui.py's scan_wildcards) runs server-side and refreshes the wildcard
// button pool. Continuous typing keeps resetting the timer, so scans never
// fire mid-typing.
onUiLoaded(function() {
    var wildcardScanTimer = null;
    var WILDCARD_SCAN_DEBOUNCE_MS = 3000;

    function clickWildcardScanTrigger() {
        var trigger = gradioApp().querySelector('#wildcard_scan_trigger');
        if (!trigger) {
            return;
        }
        // Gradio renders gr.Button as a <button>, but be defensive in case
        // the element itself isn't clickable and the real button is nested.
        var clickable = trigger.tagName === 'BUTTON' ? trigger : trigger.querySelector('button');
        if (clickable) {
            clickable.click();
        }
    }

    function scheduleWildcardScan() {
        if (wildcardScanTimer !== null) {
            clearTimeout(wildcardScanTimer);
        }
        wildcardScanTimer = setTimeout(function() {
            wildcardScanTimer = null;
            clickWildcardScanTrigger();
        }, WILDCARD_SCAN_DEBOUNCE_MS);
    }

    var promptTextarea = gradioApp().querySelector('#positive_prompt textarea');
    if (promptTextarea) {
        promptTextarea.addEventListener('input', scheduleWildcardScan);
    }
});
