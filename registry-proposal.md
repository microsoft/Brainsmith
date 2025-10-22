Great split. Below is a coherent design that gives you one **uniform registry API** and three **backends** that cover your cases without surprises. You get: (a) eager ergonomics, (b) zero-exec catalog building, and (c) truly lazy, on-demand imports.

I’ll use a single shape for items:

```python
@dataclass(frozen=True)
class CatalogItem:
    key: str                 # stable id
    title: str               # display name
    meta: dict               # arbitrary metadata
    load: Callable[[], type] # imports and returns the class on first call
```

…and a small façade:

```python
class Registry:
    def __init__(self, items: Iterable[CatalogItem]):
        self._items = {i.key: i for i in items}
        self._cache = {}

    def list(self) -> list[dict]:
        # zero-import, static listing
        return [{"key": i.key, "title": i.title, "meta": i.meta} for i in self._items.values()]

    def get(self, key: str) -> type:
        # lazy materialization
        if key not in self._cache:
            self._cache[key] = self._items[key].load()
        return self._cache[key]
```

---

# 1) **Our own components** (package-internal)

Goal: Keep normal eager imports for end-users (`from ourfw import MyComp`) but also be able to build a lazy list without importing implementation modules.

**Pattern:** put a **declarative map** of our components → dotted targets (no imports), and (optionally) PEP 562 for lazy attribute access. If you still want eager default imports, do them in a separate import path (`from ourfw.eager import *`) so your *root* package can remain lazy.

```python
# ourfw/_catalog_own.py  — pure data, no imports
OWN_DECL = {
    "dense":     {"target": "ourfw.nn.dense:Dense", "title": "Dense Layer", "meta": {"family": "nn"}},
    "conv2d":    {"target": "ourfw.nn.conv:Conv2D", "title": "Conv2D",      "meta": {"family": "nn"}},
    "userclass": {"target": "ourfw.builtin.user:UserClass", "title": "User Class", "meta": {}},
}
```

```python
# ourfw/_lazyutil.py
import importlib
def make_loader(target: str):
    mod, _, attr = target.partition(":")
    def _load():
        m = importlib.import_module(mod); return getattr(m, attr)
    return _load
```

```python
# ourfw/__init__.py  — default: lazy, introspection-friendly
from ._catalog_own import OWN_DECL
from ._lazyutil import make_loader
from .registry import CatalogItem, Registry

_own_items = [
    CatalogItem(key=k,
                title=v["title"],
                meta=v["meta"],
                load=make_loader(v["target"]))
    for k, v in OWN_DECL.items()
]
OWN = Registry(_own_items)  # programmatic access

# Optional: expose dotted access lazily (PEP 562)
_lazy_attr = {k: v["target"] for k, v in OWN_DECL.items()}
def __getattr__(name: str):
    try:
        target = _lazy_attr[name]
    except KeyError:
        raise AttributeError(name) from None
    obj = make_loader(target)()
    globals()[name] = obj
    return obj
def __dir__():
    return sorted(set(globals()) | set(_lazy_attr))
```

If you want an explicitly **eager** import surface for “classic” users, ship:

```python
# ourfw/eager.py  — opt-in eager surface
from . import __getattr__  # ensures same names exist
for name in ("Dense", "Conv2D", "UserClass"):  # or derive from OWN_DECL
    globals()[name] = getattr(__import__("ourfw", fromlist=[name]), name)
__all__ = list(globals().keys())
```

Users who prefer eager behavior import from `ourfw.eager`. Everyone else gets lazy by default plus a zero-exec `OWN.list()`.

---

# 2) **Third-party libraries via entry points**

Goal: discover plugins **without importing** them; import only on selection; carry rich metadata.

```python
# ourfw/_catalog_ep.py
from importlib.metadata import entry_points
from .registry import CatalogItem
from ._lazyutil import make_loader

GROUP = "ourfw.components"

def build_entrypoint_items() -> list[CatalogItem]:
    items = []
    for ep in entry_points().select(group=GROUP):
        # Optional rich static metadata file: read dist files if you want more than ep.name
        title = ep.name.replace("-", " ").title()
        meta = {"dist": ep.dist.metadata.get("Name", "")}
        def _load(ep=ep):
            return ep.load()  # import module + getattr symbol
        items.append(CatalogItem(key=ep.name, title=title, meta=meta, load=_load))
    return items
```

Expose it:

```python
# ourfw/plugins.py
from ._catalog_ep import build_entrypoint_items
from .registry import Registry
ENTRYPOINTS = Registry(build_entrypoint_items())
```

**Publisher UX (third-party):**

```toml
# pyproject.toml
[project.entry-points."ourfw.components"]
my-cool-layer = "thirdpkg.layers:CoolLayer"
```

No user code executes until someone calls `ENTRYPOINTS.get("my-cool-layer")`.

---

# 3) **Custom user components** (default eager, optional lazy indexing)

You want both:

* **Default ergonomic path:** users decorate a class and import it normally (eager).
* **Optional optimization:** users can *declare* the component for lazy discovery without importing their module.

### 3a) Eager by default (runtime decorator)

```python
# ourfw/api.py
EAGER = Registry([])  # will be populated at runtime

def component(*, key=None, title=None, **meta):
    def wrap(cls):
        k = key or cls.__name__.lower()
        t = title or cls.__name__
        # runtime registration (module is executing now)
        EAGER._items[k] = CatalogItem(k, t, meta, load=lambda cls=cls: cls)
        return cls
    return wrap
```

User code:

```python
from ourfw.api import component

@component(title="User's Class", category="custom")
class UserClass: ...
```

Importing the module is eager and registers immediately; listing `EAGER.list()` shows it.

### 3b) Optional **lazy** declaration (no import of user code)

Provide a **declarative registry** users can fill (config or entry points). Two easy options:

* **Config-file mode** (internal deployments):

  ```toml
  # ourfw-components.toml (loaded by us; no imports)
  [components."user-class"]
  target = "user_pkg.user_class:UserClass"
  title  = "User's Class"
  category = "custom"
  ```

  Loader:

  ```python
  # ourfw/_catalog_cfg.py
  import tomllib, importlib.resources as res
  from ._lazyutil import make_loader
  from .registry import CatalogItem

  def build_cfg_items() -> list[CatalogItem]:
      data = tomllib.loads(res.files("your_app_config_bundle")
                               .joinpath("ourfw-components.toml").read_text())
      items = []
      for key, rec in data["components"].items():
          items.append(CatalogItem(
              key=key,
              title=rec.get("title", key),
              meta={k:v for k,v in rec.items() if k not in {"target","title"}},
              load=make_loader(rec["target"])
          ))
      return items
  ```

* **User entry-point mode** (public ecosystem): just reuse case #2 and tell users to add an entry point. That’s the cleanest lazy flag.

You can expose these as:

```python
# ourfw/custom.py
from .registry import Registry
from ._catalog_cfg import build_cfg_items
LAZY_CUSTOM = Registry(build_cfg_items())
```

---

## Putting it together (one surface)

Offer a single namespace that merges all three sources:

```python
# ourfw/all_components.py
from .registry import Registry
from . import OWN
from .plugins import ENTRYPOINTS
from .api import EAGER
from .custom import LAZY_CUSTOM

def merged_registry() -> Registry:
    items = {}
    for reg in (OWN, ENTRYPOINTS, EAGER, LAZY_CUSTOM):
        for d in reg.list():
            items.setdefault(d["key"], d)  # leftmost wins
    # Rebuild CatalogItem objects so .get works by delegating to the original registries
    cats = []
    for reg in (OWN, ENTRYPOINTS, EAGER, LAZY_CUSTOM):
        for key in (i["key"] for i in reg.list()):
            if key in items:
                cats.append(CatalogItem(
                    key=key, title=reg._items[key].title, meta=reg._items[key].meta,
                    load=reg._items[key].load))
                items.pop(key, None)
    return Registry(cats)

ALL = merged_registry()
```

Users get:

```python
from ourfw.all_components import ALL

ALL.list()             # zero-exec overview across all sources
Cls = ALL.get("dense") # imports on demand; decorators run on first materialization
```

---

## Notes & traps

* **Your original `__init__.py` sketch (`from user_class import UserClass`) is eager.** Keep that path for ergonomics, but put the lazy story on a *different* surface (as above) so you don’t defeat the optimization.
* **Typing:** provide `.pyi` stubs for the lazy attributes you expose, or use `if TYPE_CHECKING:` to import eagerly only for type checkers.
* **Thread safety:** Python’s import lock guards module execution; memoize returned classes in the registry to avoid duplicate loads in highly concurrent contexts.
* **Metadata fidelity:** decorator-computed metadata only exists after the module executes. For the zero-exec lists, use *static* metadata from your own declarative maps / entry points / config. If you need dynamic metadata, expose a separate call (e.g., `inspect(key)`) that forces a load.

---

### TL;DR

* **Case 1:** Keep a static, dotted-path map for your own components; use PEP 562 for nice dotted access; offer a separate eager surface if desired.
* **Case 2:** Use **entry points** to discover 3P components without imports; import on `get()`.
* **Case 3:** Provide a runtime **decorator** (eager) *and* a **declarative config or entry point** (lazy). Merge all sources behind one `Registry` so callers get one API with lazy semantics on access.
