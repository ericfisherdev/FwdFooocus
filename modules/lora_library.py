"""
LoRA Library Page Module

Generates the HTML page for browsing the LoRA library with search and filter functionality.
"""

import html
import json
from typing import Any

from modules.lora_metadata import get_all_library_data, get_distinct_base_models


def generate_library_html() -> str:
    """
    Generate the complete HTML page for the LoRA library.

    Returns:
        Complete HTML document as a string.
    """
    library_data = get_all_library_data()
    base_models = get_distinct_base_models()

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LoRA Library - FwdFooocus</title>
    <style>
        {_get_library_css()}
    </style>
</head>
<body>
    <div class="container">
        <header class="header">
            <h1>LoRA Library</h1>
            <div class="controls">
                <div class="search-box">
                    <input type="text" id="search-input" placeholder="Search LoRAs..."
                           oninput="filterLibrary()">
                    <button onclick="clearSearch()" class="clear-btn" title="Clear search">×</button>
                </div>
                <div class="filter-box">
                    <label for="base-model-filter">Base Model:</label>
                    <select id="base-model-filter" onchange="filterLibrary()">
                        <option value="">All Models</option>
                        {_generate_filter_options(base_models)}
                    </select>
                </div>
                <div class="results-count" id="results-count">
                    {len(library_data)} LoRAs
                </div>
            </div>
        </header>

        <main class="library-content">
            {_generate_empty_state() if not library_data else ''}
            <div class="lora-grid" id="lora-grid">
                {_generate_lora_cards(library_data)}
            </div>
        </main>
    </div>

    <script>
        {_get_library_javascript(library_data)}
    </script>
</body>
</html>'''


def _get_library_css() -> str:
    """Get the CSS styles for the library page."""
    return '''
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #1f2937;
            --bg-tertiary: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent: #3b82f6;
            --accent-hover: #2563eb;
            --border: #4b5563;
            --success: #10b981;
            --warning: #f59e0b;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            margin-bottom: 24px;
        }

        .header h1 {
            font-size: 1.75rem;
            margin-bottom: 16px;
            color: var(--text-primary);
        }

        .controls {
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: center;
        }

        .search-box {
            position: relative;
            flex: 1;
            min-width: 200px;
            max-width: 400px;
        }

        .search-box input {
            width: 100%;
            padding: 10px 36px 10px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 14px;
        }

        .search-box input:focus {
            outline: none;
            border-color: var(--accent);
        }

        .clear-btn {
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 18px;
            cursor: pointer;
            padding: 4px;
        }

        .clear-btn:hover {
            color: var(--text-primary);
        }

        .filter-box {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .filter-box label {
            color: var(--text-secondary);
            font-size: 14px;
        }

        .filter-box select {
            padding: 10px 12px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--bg-secondary);
            color: var(--text-primary);
            font-size: 14px;
            cursor: pointer;
        }

        .filter-box select:focus {
            outline: none;
            border-color: var(--accent);
        }

        .results-count {
            color: var(--text-secondary);
            font-size: 14px;
            margin-left: auto;
        }

        .library-content {
            min-height: 400px;
        }

        .lora-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 16px;
        }

        .lora-card {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            transition: border-color 0.2s, transform 0.2s;
        }

        .lora-card:hover {
            border-color: var(--accent);
        }

        .lora-card:target {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
        }

        .lora-card.hidden {
            display: none;
        }

        .lora-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 12px;
        }

        .lora-filename {
            font-weight: 600;
            font-size: 14px;
            word-break: break-word;
            flex: 1;
        }

        .lora-size {
            color: var(--text-secondary);
            font-size: 12px;
            white-space: nowrap;
            margin-left: 8px;
        }

        .lora-meta {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        .meta-row {
            display: flex;
            font-size: 13px;
        }

        .meta-label {
            color: var(--text-secondary);
            width: 100px;
            flex-shrink: 0;
        }

        .meta-value {
            color: var(--text-primary);
            word-break: break-word;
        }

        .meta-value.truncate {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }

        .trigger-words {
            display: flex;
            flex-wrap: wrap;
            gap: 4px;
        }

        .trigger-tag {
            background: var(--bg-tertiary);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
        }

        .trigger-more {
            color: var(--text-secondary);
            font-size: 12px;
        }

        .copy-triggers-btn {
            background: var(--bg-tertiary);
            border: 1px solid var(--border);
            color: var(--text-primary);
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }

        .copy-triggers-btn:hover {
            background: var(--accent);
        }

        .copy-triggers-btn.copied {
            background: var(--success);
        }

        .base-model-badge {
            display: inline-block;
            background: var(--accent);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }

        .empty-state {
            text-align: center;
            padding: 60px 20px;
            color: var(--text-secondary);
        }

        .empty-state h2 {
            font-size: 1.25rem;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .loading {
            text-align: center;
            padding: 40px;
            color: var(--text-secondary);
        }

        @media (max-width: 768px) {
            .controls {
                flex-direction: column;
                align-items: stretch;
            }

            .search-box {
                max-width: none;
            }

            .results-count {
                margin-left: 0;
            }

            .lora-grid {
                grid-template-columns: 1fr;
            }
        }
    '''


def _get_library_javascript(library_data: list[dict[str, Any]]) -> str:
    """Get the JavaScript code for the library page."""
    # Embed library data as JSON for client-side filtering
    library_json = json.dumps([
        {
            'id': _sanitize_id(item.get('filename', '')),
            'filename': item.get('filename', ''),
            'base_model': item.get('base_model', ''),
            'trigger_words': item.get('trigger_words', []),
            'description': item.get('description', ''),
            'characters': item.get('characters', []),
            'styles': item.get('styles', []),
        }
        for item in library_data
    ])

    return f'''
        const libraryData = {library_json};

        function filterLibrary() {{
            const searchTerm = document.getElementById('search-input').value.toLowerCase().trim();
            const baseModelFilter = document.getElementById('base-model-filter').value;
            const grid = document.getElementById('lora-grid');
            const cards = grid.querySelectorAll('.lora-card');
            let visibleCount = 0;

            cards.forEach((card, index) => {{
                const data = libraryData[index];
                let show = true;

                // Apply base model filter
                if (baseModelFilter && data.base_model !== baseModelFilter) {{
                    show = false;
                }}

                // Apply search filter
                if (show && searchTerm) {{
                    const searchableText = [
                        data.filename,
                        data.base_model,
                        data.description,
                        data.trigger_words.join(' '),
                        data.characters.join(' '),
                        data.styles.join(' ')
                    ].join(' ').toLowerCase();

                    if (!searchableText.includes(searchTerm)) {{
                        show = false;
                    }}
                }}

                if (show) {{
                    card.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    card.classList.add('hidden');
                }}
            }});

            // Update results count
            document.getElementById('results-count').textContent =
                visibleCount + ' ' + (visibleCount === 1 ? 'LoRA' : 'LoRAs');
        }}

        function clearSearch() {{
            document.getElementById('search-input').value = '';
            filterLibrary();
        }}

        function copyTriggerWords(button, words) {{
            const text = words.join(', ');
            navigator.clipboard.writeText(text).then(() => {{
                button.classList.add('copied');
                button.textContent = 'Copied!';
                setTimeout(() => {{
                    button.classList.remove('copied');
                    button.textContent = '📋 Copy';
                }}, 2000);
            }}).catch(err => {{
                console.error('Failed to copy:', err);
                alert('Failed to copy to clipboard');
            }});
        }}

        // Handle deep linking - scroll to target on load
        document.addEventListener('DOMContentLoaded', function() {{
            if (window.location.hash) {{
                const target = document.querySelector(window.location.hash);
                if (target) {{
                    setTimeout(() => {{
                        target.scrollIntoView({{ behavior: 'smooth', block: 'center' }});
                    }}, 100);
                }}
            }}
        }});
    '''


def _generate_filter_options(base_models: list[str]) -> str:
    """Generate option elements for the base model filter."""
    options = []
    for model in base_models:
        escaped = html.escape(model)
        options.append(f'<option value="{escaped}">{escaped}</option>')
    return '\n'.join(options)


def _generate_lora_cards(library_data: list[dict[str, Any]]) -> str:
    """Generate HTML cards for all LoRAs."""
    cards = []
    for item in library_data:
        cards.append(_generate_lora_card(item))
    return '\n'.join(cards)


def _generate_lora_card(metadata: dict[str, Any]) -> str:
    """Generate HTML for a single LoRA card."""
    filename = metadata.get('filename', 'Unknown')
    card_id = _sanitize_id(filename)
    file_size = metadata.get('file_size', 0)
    size_mb = file_size / (1024 * 1024)
    base_model = metadata.get('base_model', 'Unknown')
    trigger_words = metadata.get('trigger_words', [])
    description = metadata.get('description', '')
    characters = metadata.get('characters', [])
    styles = metadata.get('styles', [])

    # Generate trigger words display
    trigger_html = ''
    if trigger_words:
        # Show first 5 tags
        visible_tags = trigger_words[:5]
        tags_html = ''.join(
            f'<span class="trigger-tag">{html.escape(tag)}</span>'
            for tag in visible_tags
        )
        more_count = len(trigger_words) - 5
        more_html = f'<span class="trigger-more">+{more_count} more</span>' if more_count > 0 else ''

        # Create JSON for copy button
        words_json = json.dumps(trigger_words)
        copy_btn = f'''<button class="copy-triggers-btn"
                              onclick="copyTriggerWords(this, {html.escape(words_json)})">
                        📋 Copy
                      </button>'''

        trigger_html = f'''
            <div class="meta-row">
                <span class="meta-label">Triggers:</span>
                <div class="meta-value">
                    <div class="trigger-words">{tags_html}{more_html}</div>
                    {copy_btn}
                </div>
            </div>
        '''
    else:
        trigger_html = '''
            <div class="meta-row">
                <span class="meta-label">Triggers:</span>
                <span class="meta-value" style="color: var(--text-secondary);">No trigger words</span>
            </div>
        '''

    # Generate description
    desc_html = ''
    if description:
        desc_escaped = html.escape(description[:200])
        desc_html = f'''
            <div class="meta-row">
                <span class="meta-label">Description:</span>
                <span class="meta-value truncate">{desc_escaped}</span>
            </div>
        '''

    # Generate characters/styles
    extras = []
    if characters:
        chars = ', '.join(characters[:3])
        if len(characters) > 3:
            chars += f' (+{len(characters) - 3})'
        extras.append(f'''
            <div class="meta-row">
                <span class="meta-label">Characters:</span>
                <span class="meta-value">{html.escape(chars)}</span>
            </div>
        ''')

    if styles:
        style_str = ', '.join(styles[:3])
        if len(styles) > 3:
            style_str += f' (+{len(styles) - 3})'
        extras.append(f'''
            <div class="meta-row">
                <span class="meta-label">Styles:</span>
                <span class="meta-value">{html.escape(style_str)}</span>
            </div>
        ''')

    extras_html = ''.join(extras)

    return f'''
        <div class="lora-card" id="{card_id}">
            <div class="lora-header">
                <span class="lora-filename">{html.escape(filename)}</span>
                <span class="lora-size">{size_mb:.1f} MB</span>
            </div>
            <div class="lora-meta">
                <div class="meta-row">
                    <span class="meta-label">Base Model:</span>
                    <span class="meta-value">
                        <span class="base-model-badge">{html.escape(base_model)}</span>
                    </span>
                </div>
                {trigger_html}
                {desc_html}
                {extras_html}
            </div>
        </div>
    '''


def _generate_empty_state() -> str:
    """Generate HTML for empty library state."""
    return '''
        <div class="empty-state">
            <h2>No LoRAs Found</h2>
            <p>No LoRA files have been scanned yet. LoRAs will appear here once the background scan completes.</p>
        </div>
    '''


def _sanitize_id(filename: str) -> str:
    """
    Sanitize a filename to use as an HTML ID.

    Args:
        filename: The filename to sanitize.

    Returns:
        Sanitized string safe for use as an HTML ID.
    """
    # Remove extension and replace non-alphanumeric chars with hyphens
    import re
    name = filename.rsplit('.', 1)[0] if '.' in filename else filename
    sanitized = re.sub(r'[^a-zA-Z0-9]', '-', name)
    # Remove consecutive hyphens and trim
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    return sanitized.lower() if sanitized else 'lora'
