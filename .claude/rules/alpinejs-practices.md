---
paths:
  - "**/*.html"
  - "**/*.js"
---

# Alpine.js Practices

Derived from: Alpine.js official documentation (alpinejs.dev).

These rules apply when working with Alpine.js components — inline `x-data` in templates and extracted component definitions in JS files.

---

## Core Principles

- Alpine is for interactive sprinkles (dropdowns, modals, tabs, toggles, search), not full SPAs.
- Declarative over imperative — describe what the UI should look like given state, don't manually manipulate the DOM.
- Locality of behavior — put behavior (`x-data`, `@click`, `:class`) on the element in markup.
- Server-rendered HTML is the starting point. Alpine enhances it.

## Do's — Templates

- Use `x-cloak` with `[x-cloak] { display: none !important; }` to prevent flash of unstyled content.
- Use `x-show` + `x-transition` for smooth show/hide. Use `x-if` only when elements must be removed from the DOM entirely.
- Always use `:key` on `x-for` loops for efficient re-rendering.
- Use `x-bind` object syntax for reusable directive bundles (`x-bind="trigger"`).
- Use `:class` with object syntax for conditional classes: `:class="{ 'active': isActive }"`.
- Use `x-model` with `.debounce.Nms`, `.lazy`, `.number` modifiers as appropriate.
- Use `x-teleport` for modals and overlays to avoid z-index and overflow issues.
- Use `@click.outside` for dismiss-on-click-outside patterns (dropdowns, popovers).
- Use event modifiers (`.prevent`, `.stop`, `.window`, `.once`, `.self`) instead of calling `event.preventDefault()` or `event.stopPropagation()` in handlers.

## Do's — JavaScript

- Extract components with `Alpine.data('name', fn)` when they grow beyond a few lines or are reused across templates.
- Use regular functions (not arrow functions) in `Alpine.data()` when accessing `this` or Alpine magics like `$watch`, `$refs`, `$dispatch`.
- Use `Alpine.store('name', obj)` for state shared across multiple components (theme, auth, notifications).
- Register components and stores inside the `alpine:init` event listener.
- Use `init()` and `destroy()` lifecycle methods in `Alpine.data()` definitions for setup/teardown.
- Use getters (`get propertyName()`) for computed/derived values.
- Use `$watch('property', callback)` for side effects on specific property changes.
- Use `x-effect` for reactive expressions that should re-run when any dependency changes.
- Use `$dispatch('event-name', detail)` for component communication. Use `.window` modifier on listeners for cross-component events.
- Use `$nextTick` when reading DOM state after a reactive update.

## Do's — Plugins

- Use **Persist** (`$persist()`) for user preferences (theme, sidebar state). Use `.as('key')` for custom storage keys.
- Use **Intersect** (`x-intersect`) for lazy loading and scroll-triggered animations. Use `.once` for one-time triggers.
- Use **Focus** (`x-trap`) for trapping focus in modals and dialogs.
- Use **Collapse** (`x-collapse`) for smooth height-based accordion transitions.
- Use **Mask** (`x-mask`) for formatted input fields (phone, date, credit card).

## Don'ts

- Don't use Alpine for complex SPAs — no client-side routing, deep component trees, or heavy state management.
- Don't use `x-html` with untrusted user content — it's an XSS vector.
- Don't put heavy logic in inline `x-data` — extract to `Alpine.data()`.
- Don't use `x-if` when `x-show` suffices — `x-if` destroys/recreates DOM and doesn't support `x-transition`.
- Don't forget `:key` on `x-for` — without it Alpine can't track list items efficiently.
- Don't use `$refs` across component boundaries — refs are scoped to their `x-data`.
- Don't dispatch events without `.window` when communicating between unrelated components.
- Don't manipulate Alpine-managed DOM with vanilla JS — let Alpine's reactivity handle updates.
- Don't store application-critical state only in Alpine — it's gone on page refresh unless persisted.
- Don't use `x-init` for logic depending on other components — use `alpine:initialized` for that.

## Component Communication Patterns

- **Parent → Child:** nested `x-data` scopes inherit outer scope data automatically.
- **Child → Parent:** use `$dispatch('event-name', data)` — DOM events bubble up.
- **Sibling / Unrelated:** use `$dispatch` with `@event.window` listener, or shared `$store`.
- **Global state:** use `Alpine.store()` — reactive and accessible from any component via `$store.name`.

## Performance

- Use `x-show` over `x-if` for frequently toggled elements — `x-show` only changes CSS display.
- Use `.debounce` and `.throttle` modifiers on inputs and event handlers.
- Use `x-intersect.once` for one-time lazy loading — observer disconnects after first trigger.
- Keep `x-data` scopes small and focused. Large monolithic components are harder to reason about.
- Use `$nextTick` sparingly — frequent use often means fighting the reactive model.
- Avoid persisting large or frequently-changing objects with the Persist plugin — it serializes to JSON on every change.
- Keep transitions short (150-300ms) for perceived responsiveness.
