window.main_viewer_height = 512;

function refresh_grid() {
    let gridContainer = document.querySelector('#final_gallery .grid-container');
    let final_gallery = document.getElementById('final_gallery');

    if (gridContainer) if (final_gallery) {
        let rect = final_gallery.getBoundingClientRect();
        let cols = Math.ceil((rect.width - 16.0) / rect.height);
        if (cols < 2) cols = 2;
        gridContainer.style.setProperty('--grid-cols', cols);
    }
}

function refresh_grid_delayed() {
    refresh_grid();
    setTimeout(refresh_grid, 100);
    setTimeout(refresh_grid, 500);
    setTimeout(refresh_grid, 1000);
}

function resized() {
    let windowHeight = window.innerHeight - 260;
    let elements = document.getElementsByClassName('main_view');

    if (windowHeight > 745) windowHeight = 745;

    for (let i = 0; i < elements.length; i++) {
        elements[i].style.height = windowHeight + 'px';
    }

    window.main_viewer_height = windowHeight;

    refresh_grid();
}

function viewer_to_top(delay = 100) {
    setTimeout(() => window.scrollTo({top: 0, behavior: 'smooth'}), delay);
}

function viewer_to_bottom(delay = 100) {
    let element = document.getElementById('positive_prompt');
    let yPos = window.main_viewer_height;

    if (element) {
        yPos = element.getBoundingClientRect().top + window.scrollY;
    }

    setTimeout(() => window.scrollTo({top: yPos - 8, behavior: 'smooth'}), delay);
}

window.addEventListener('resize', (e) => {
    resized();
});

onUiLoaded(async () => {
    resized();
});

function on_style_selection_blur() {
    let target = document.querySelector("#gradio_receiver_style_selections textarea");
    target.value = "on_style_selection_blur " + Math.random();
    let e = new Event("input", {bubbles: true})
    Object.defineProperty(e, "target", {value: target})
    target.dispatchEvent(e);
}

// Gradio 3.41.2's Radio renders every choice label as an escaped text node, so the
// grey "W:H" suffix in an aspect ratio label shows up as literal <span> markup. The
// label string is also the radio's value, so the markup to render is always
// input.value (a server-generated string from modules.config.add_ratio). The write is
// idempotent (skipped when already equal), which keeps the MutationObserver below from
// re-triggering itself, and it re-runs after every gr.update(choices=...) swap, where
// Gradio recreates or retargets the label text nodes.
function renderAspectRatioLabels() {
    document.querySelectorAll('.aspect_ratios label').forEach(function (label) {
        const input = label.querySelector('input[type=radio]');
        const span = label.querySelector(':scope > span');
        if (input && span && span.innerHTML !== input.value) {
            span.innerHTML = input.value;
        }
    });
}

function observeAspectRatioLabels() {
    const radio = document.querySelector('.aspect_ratios');
    if (!radio) {
        return;
    }
    // childList: the choice list grew or shrank. data-testid: same-length swap, where
    // Gradio only rewrites each label's data-testid to match its new value.
    new MutationObserver(renderAspectRatioLabels).observe(radio, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ['data-testid']
    });
}

onUiLoaded(async () => {
    renderAspectRatioLabels();
    observeAspectRatioLabels();

    document.querySelector('.style_selections').addEventListener('focusout', function (event) {
        setTimeout(() => {
            if (!this.contains(document.activeElement)) {
                on_style_selection_blur();
            }
        }, 200);
    });

    let inputs = document.querySelectorAll('.lora_weight input[type="range"]');

    inputs.forEach(function (input) {
        input.style.marginTop = '12px';
    });
});
