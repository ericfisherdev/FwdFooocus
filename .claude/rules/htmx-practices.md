---
paths:
  - "**/*.html"
  - "**/*.js"
  - "**/*.py"
  - "**/*.css"
---

# htmx & Hypermedia Practices

Derived from: Hypermedia Systems (Carson Gross), htmx official documentation.

These rules apply when working on htmx-driven UI code — templates, JS event handlers, server endpoints returning HTML partials, and related styles.

---

## Core Principles

- The server is the single source of truth for application state. The DOM is a view of server state, not a separate state store.
- Return HTML partials from the server, never JSON for htmx-consumed endpoints. HTML is the API.
- Locality of behavior: put behavior (`hx-get`, `hx-post`, `hx-trigger`) on the element in HTML rather than scattering it across JS files.
- Progressive enhancement: start with plain HTML that works without JavaScript, layer htmx on top.

## Do's — HTML Templates

- Use `hx-boost="true"` on navigation containers to AJAX-ify links and forms with zero JS.
- Use `hx-target` to specify where response HTML is swapped. Default is the triggering element.
- Use `hx-swap` to control insertion: `innerHTML` (default), `outerHTML`, `beforeend`, `afterbegin`, `delete`, `none`.
- Use `hx-swap-oob="true"` to update multiple page regions from a single response (notification badges, status bars, counters).
- Use `hx-trigger` with modifiers for fine-grained control: `changed`, `delay:Nms`, `throttle:Nms`, `once`, `revealed`, `every Ns`, `from:<selector>`.
- Use `hx-indicator` to show loading spinners during requests.
- Use `hx-confirm` before destructive actions (delete, overwrite).
- Use `hx-push-url` on navigations that should be bookmarkable and support back-button.
- Use `hx-disabled-elt` to disable buttons/inputs during in-flight requests.
- Use `hx-sync` to coordinate concurrent requests and prevent race conditions (`abort`, `replace`, `drop`, `queue`).
- Use `hx-preserve` to keep elements (e.g., video players, text inputs) intact across swaps.
- Use `hx-vals` and `hx-include` to send extra data without hidden form fields.
- Design resource-oriented URLs — each URL represents a resource or view, not an RPC action.

## Do's — Server Endpoints

- Check the `HX-Request` header to detect htmx requests vs full-page requests. Return a partial for htmx, a full page otherwise.
- Use response headers for server-driven client control:
  - `HX-Trigger` — fire custom events (toasts, component refreshes).
  - `HX-Redirect` — full-page redirect.
  - `HX-Location` — redirect without full page reload.
  - `HX-Retarget` — override the swap target.
  - `HX-Reswap` — override the swap strategy.
- Return the smallest meaningful HTML partial — don't return a full page when only a table row changed.
- Include CSRF tokens in all mutating requests via `htmx:configRequest` or `hx-headers`.
- Set `hx-disable` on user-generated content to prevent htmx attribute injection.
- Validate and authorize on the server. htmx moves logic server-side — that means thorough server-side checks.

## Do's — JavaScript Event Handlers

- Use `htmx:afterSwap` and `htmx:afterSettle` to initialize JS components on newly swapped content.
- Use `htmx:configRequest` only for presentation concerns (adding headers, tokens). Business logic belongs on the server.
- Use `htmx:beforeSwap` to customize swap behavior for error status codes (e.g., swap 422 validation errors into a form-errors container).
- Listen for `htmx:responseError` and `htmx:sendError` to provide user feedback on failures.

## Don'ts

- Don't return JSON from htmx endpoints. If other consumers need JSON, create separate endpoints.
- Don't build a client-side router. The server controls navigation via `hx-boost`, `hx-push-url`, and `HX-Redirect`.
- Don't manage application state in JavaScript. Store state in the server session, database, or HTML (data attributes, hidden inputs).
- Don't over-fragment partials. Each request should return a meaningful unit of UI, not a single `<span>`. Too many micro-requests create latency.
- Don't ignore the URL. Users must be able to bookmark, share, and use back/forward navigation.
- Don't write custom JavaScript for things htmx attributes handle. Check the attribute reference first.
- Don't use htmx for interactions that must be instant (real-time text editing, drag-and-drop, canvas). Those need client-side JS.
- Don't use `hx-trigger="load"` carelessly — it fires on every swap into the DOM and can create request loops.
- Don't put business logic in JS event handlers. JS handles presentation (headers, spinners, component init); the server handles logic.

## Architecture Patterns

### Partial Swap (Primary Pattern)
User interaction → htmx HTTP request → server returns HTML fragment → htmx swaps into DOM at target.

### Out-of-Band Updates
One response updates multiple independent page regions. Primary content goes to `hx-target`; additional elements with `hx-swap-oob="true"` and matching `id` attributes update their locations. Reduces round trips.

### Active Search
Debounced input with `hx-trigger="keyup changed delay:300ms"`. Server returns rendered results. No client-side state or JSON parsing.

### Lazy Loading
Use `hx-trigger="revealed"` to load content when it enters the viewport. Server returns the content plus a new sentinel for the next chunk.

### Polling
Use `hx-trigger="every Ns"` for simple server-driven polling. Upgrade to WebSocket or SSE extensions for real-time needs.

### Server-Driven Error Handling
Use `htmx:beforeSwap` to handle non-2xx responses — set `shouldSwap = true` and retarget to an error container for 422 validation errors. Use `htmx:responseError` and `htmx:sendError` for network and server failures.

### Request Coordination
Use `hx-sync` to prevent race conditions: `abort` (cancel in-flight), `replace` (swap in-flight for new), `drop` (ignore new if busy), `queue` (FIFO processing).

## Performance

- `hx-boost` is free performance — converts full-page navigations to partial updates with no code changes.
- Lazy load below-the-fold content with `hx-trigger="revealed"`.
- Debounce inputs with `delay:300ms` or `throttle:500ms` to avoid flooding the server.
- Use `hx-swap="none"` when only server-side processing is needed (analytics, logging) with no DOM update.
- Use out-of-band swaps to batch multiple region updates into one response instead of multiple requests.
- Return minimal partials. Smaller responses = faster swaps.
- Use CSS transitions with swap timing (`swap:200ms`, `settle:100ms`) to mask network latency.
- Use `hx-push-url` selectively — only on navigations that should be bookmarkable, not on inline edits.
- Set `hx-history="false"` on elements whose responses should not be cached in browser history.
- Server-rendered HTML is faster than JSON→JS template→DOM for most cases. The HTML is ready to insert.
