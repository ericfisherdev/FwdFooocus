---
paths:
  - "**/*.py"
  - "**/*.pyx"
  - "**/*.pyi"
---

# Python Coding Practices

Derived from: Fluent Python, Architecture Patterns with Python, High Performance Python, CPython Internals, Robust Python.

Conflict resolution: performance wins over maintainability; Fluent Python idioms are preferred otherwise.

---

## Data Model & Object Design

- Implement dunder methods (`__repr__`, `__eq__`, `__hash__`, `__getitem__`, etc.) so objects integrate naturally with the language — iteration, sorting, `in`, slicing, and context managers should "just work."
- Use `@property` and descriptors for computed or validated attributes — never write Java-style getters/setters.
- Use `@dataclass` (or `attrs`) for data-holding classes. Use `frozen=True` for value objects. Use `slots=True` (Python 3.10+) for memory-heavy instances.
- Use `Enum` instead of string/int constants. `Status.ACTIVE` is self-documenting and typo-proof.
- Use `namedtuple` or `typing.NamedTuple` when you need a lightweight immutable record.
- Use `__slots__` on any class likely to have many instances — it eliminates per-instance `__dict__` and saves ~40% memory.
- Use `__post_init__` validators on dataclasses to enforce invariants at construction time.

## Type System & Contracts

- Add type annotations to all function signatures and module-level variables — they are machine-verifiable documentation with zero runtime cost.
- Use `Protocol` for duck typing contracts (structural subtyping) — preferred over ABCs for defining interfaces.
- Use `collections.abc` ABCs (`Mapping`, `Sequence`, `Iterable`) as mixin bases and for `isinstance` checks — never subclass `dict`/`list`/`str` directly (use `UserDict`, `UserList`, etc.).
- Use `Union`/`|` and `Optional` to explicitly mark variant and nullable types — then handle every branch.
- Use `NewType` to prevent primitive obsession (`UserId`, `EmailAddress` as distinct types).
- Use `TypedDict` for dictionary-shaped data with known keys (API responses, config dicts).
- Use `Final`/`@final` to communicate "do not override or reassign."
- Use `assert_never()` (Python 3.11+) for exhaustiveness checking on union types.
- Don't use `Any` except at true system boundaries (FFI, raw deserialization). Every `Any` is a hole in type safety.
- Don't use `Optional` when `None` is never valid — it pushes null-checking into every caller.
- Don't ignore `mypy`/`pyright` errors. They flag ambiguities that become bugs.

## Composition & Architecture

- Prefer composition over inheritance. Don't create deep hierarchies — use mixins sparingly, favor Protocols.
- Replace strategy classes with plain callables or `Protocol` objects. A `Callable[[Order], Decimal]` beats a `DiscountStrategy` ABC.
- Inject dependencies via constructor or function arguments — no DI framework needed in Python.
- Use decorators for cross-cutting concerns (caching, validation, access control). Use class decorators over metaclasses; use `__init_subclass__` when possible.
- Use context managers (`with`) for resource management — implement `__enter__`/`__exit__` or use `contextlib.contextmanager`. Prefer these over `__del__` destructors.

## Domain & Service Architecture

- Define domain models as plain Python classes with no ORM/framework imports. Separate what the system does from how it's wired.
- Use the Repository pattern to abstract persistence — domain calls `repo.get(id)`, never `session.query()`.
- Use Unit of Work for transaction boundaries — atomic commit/rollback, owns repositories for the transaction's lifetime.
- Keep API/view handlers thin: parse request, call service layer, return response. No business logic in handlers.
- Use Domain Events + Message Bus for cross-aggregate side effects — don't introduce event-driven architecture until you actually have tangled side effects.
- Use CQRS when read and write needs diverge — bypass the ORM for read-heavy endpoints with raw SQL or views.
- Don't let ORM models be domain models. Use classical mapping or separate layers.

## Error Handling

- Catch specific exceptions — never `except Exception` or bare `except`. Let unexpected errors propagate.
- Use exceptions for truly exceptional conditions (network failures, missing files). Use return types (`Result[T, E]`, union types) for expected failure modes (validation, not-found).
- Create domain-specific exception hierarchies — catch `PaymentError`, not `Exception`.
- Never catch and swallow exceptions silently. Log or re-raise.
- Validate at the boundary (API entry points, config loading, deserialization). Once validated, data flows as typed, trustworthy objects.
- Fail fast and loudly — a `ValueError` at startup beats silent corruption in production.

## Generators & Lazy Evaluation

- Prefer generators and `yield` over building large lists in memory. Chain generators for ETL-like pipelines.
- Use generator expressions over list comprehensions when iterating only once — avoids allocating the full list.
- For large datasets, process in fixed-size chunks to balance memory and throughput.

## Concurrency

- I/O-bound: use `asyncio` (cooperative, single-threaded) or `ThreadPoolExecutor` (preemptive).
- CPU-bound: use `ProcessPoolExecutor` or `multiprocessing.Pool` — each process has its own GIL.
- Don't use threads for CPU-bound work in CPython — the GIL serializes Python bytecode execution.
- Don't use `multiprocessing` for small payloads — pickle serialization overhead can exceed computation time.
- Hybrid: use `asyncio` for I/O coordination with `ProcessPoolExecutor` for CPU-intensive stages.

## Testing

- Unit test the domain model (fast, no I/O). Integration test the repository (real DB).
- Use fakes (in-memory repos, fake UoW) for unit tests — don't mock everything. Reserve real infrastructure for integration tests.
- Test behavior (inputs → outputs/side effects), not implementation details.
- If testing is hard, the architecture has coupling problems.

## Performance — General Rules

- Profile before optimizing. Use `cProfile`/`py-spy` for functions, `line_profiler` for hot lines, `memory_profiler`/`tracemalloc` for memory.
- Fix the algorithm first (O(n log n) beats O(n^2) regardless of language), then optimize constant factors.
- Optimize only the hot path (typically 3-5% of code). Stop when performance meets requirements.

## Performance — CPython Specifics

- Local variable access (`LOAD_FAST`) is ~20% faster than global (`LOAD_GLOBAL`). In hot functions, assign globals to locals.
- In hot loops, hoist attribute lookups: `append = my_list.append` before the loop.
- List comprehensions are faster than `for`-loops for building lists — iteration runs in C.
- `str.join()` is O(n); repeated `+=` is O(n^2). Always use `join` for string assembly.
- Use `dict`/`set` for membership tests — O(1) vs O(n) for lists.
- `collections.deque` for O(1) append/popleft (vs O(n) for `list.pop(0)`).
- `bisect` module for sorted-sequence operations — O(log n) lookups.
- `@functools.lru_cache` / `@functools.cache` for memoizing pure functions.
- Use `operator` module (`itemgetter`, `attrgetter`) and `functools` (`partial`, `reduce`) — stdlib C implementations beat equivalent lambdas.
- Tuples are faster to create and use less memory than lists. Use tuples for immutable sequences.

## Performance — Memory

- `__slots__` saves ~40% memory per instance. Use `@dataclass(slots=True)` on Python 3.10+.
- `array.array` or NumPy arrays for homogeneous numeric data — far less memory than `list`.
- `memoryview` for zero-copy slicing of binary data.
- `mmap` for large file access without loading into RAM.
- Generators use O(1) memory regardless of sequence length.

## Performance — I/O

- Batch database operations — one INSERT of 1000 rows beats 1000 individual INSERTs by 10-100x.
- Use `aiohttp`/`httpx` for concurrent HTTP — 50-100x faster than sequential `requests`.
- Buffer file I/O with `buffering=65536` or `io.BufferedReader`.
- Use binary formats (Protobuf, MessagePack, Parquet) over JSON/CSV for large datasets.
- Use eager loading (`joinedload`) in repositories to avoid N+1 query traps from lazy loading.
- Emit events for non-critical side effects (emails, search indexing) — don't block the main request.

## Performance — Numerical

- Use NumPy vectorization instead of Python loops for numerical computation. Don't use `pandas` `.iterrows()`.
- Optimization tiers: pure Python → NumPy/SciPy → Numba JIT → Cython → C/Rust extensions.
- Don't prematurely reach for C extensions — algorithmic fixes + NumPy get you 90% there.

## Common Traps

- Never use mutable default arguments (`def f(x=[])`). The default is shared across calls.
- Never use `is` for value comparison — only for `None` and sentinel checks. Integer/string interning is an implementation detail.
- Don't mutate objects you don't own (function arguments). Aliasing bugs from shared mutable references are a top Python bug source.
- Don't use `isinstance()` as a substitute for polymorphism — use union types or Protocols.
- Don't rely on `__del__` for cleanup — it can prevent GC of cycles and has unpredictable execution order.
- Use `weakref` to break circular references in caches and observer patterns.
