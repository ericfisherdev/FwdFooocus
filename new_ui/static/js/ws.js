/**
 * WebSocket client for generation progress.
 *
 * Connects to /ws/generation, streams progress to $store.generation,
 * and updates $store.ui connection state. Auto-reconnects with
 * exponential backoff.
 *
 * Must be loaded after stores.js (uses Alpine stores).
 * Registers via alpine:init event.
 */

const WS_MAX_BACKOFF_MS = 30_000;
const WS_INITIAL_BACKOFF_MS = 1_000;

document.addEventListener('alpine:init', () => {

    const gen = Alpine.store('generation');

    let ws = null;
    let backoff = WS_INITIAL_BACKOFF_MS;
    let reconnectTimer = null;
    let hasConnectedBefore = false;
    let connectionState = 'disconnected'; // connected | connecting | reconnecting | disconnected

    function getWsUrl() {
        const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
        return `${proto}//${location.host}/ws/generation`;
    }

    function setConnectionState(state) {
        connectionState = state;
        Alpine.store('ui').connectionState = state;
    }

    function connect() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
            return;
        }

        setConnectionState(hasConnectedBefore ? 'reconnecting' : 'connecting');
        ws = new WebSocket(getWsUrl());

        ws.onopen = () => {
            hasConnectedBefore = true;
            backoff = WS_INITIAL_BACKOFF_MS;
            setConnectionState('connected');
        };

        ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                handleMessage(msg);
            } catch (e) {
                console.error('[ws] Failed to parse message:', e);
            }
        };

        ws.onclose = () => {
            setConnectionState('disconnected');
            scheduleReconnect();
        };

        ws.onerror = () => {
            // onclose will fire after onerror
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            setConnectionState('reconnecting');
            backoff = Math.min(backoff * 2, WS_MAX_BACKOFF_MS);
            connect();
        }, backoff);
    }

    /**
     * Parse "Image X/Y" from progress text to extract current/total image counts.
     * Returns { current, total } or null if the pattern is not found.
     */
    function parseImageProgress(text) {
        if (!text) return null;
        const match = text.match(/Image\s+(\d+)\/(\d+)/i);
        if (!match) return null;
        return { current: parseInt(match[1], 10), total: parseInt(match[2], 10) };
    }

    function handleMessage(msg) {
        switch (msg.type) {
            case 'preview': {
                gen.isGenerating = true;
                gen.percentage = msg.percentage ?? 0;
                gen.progressText = msg.text ?? '';
                if (msg.image) {
                    gen.previewImage = 'data:image/jpeg;base64,' + msg.image;
                }
                const progress = parseImageProgress(msg.text);
                if (progress) {
                    gen.currentImage = progress.current;
                    gen.totalImages = progress.total;
                }
                break;
            }

            case 'results':
                // Intermediate results (images completed so far)
                gen.isGenerating = true;
                window.dispatchEvent(new CustomEvent('generation-results', {
                    detail: { images: msg.images }
                }));
                break;

            case 'finish':
                window.dispatchEvent(new CustomEvent('generation-finish', {
                    detail: { images: msg.images }
                }));
                gen.reset();
                break;

            case 'heartbeat':
                // Server is alive, nothing to update
                break;

            default:
                console.warn('[ws] Unknown message type:', msg.type);
        }
    }

    // Start connection on page load
    connect();
});
