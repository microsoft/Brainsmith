# Future Action Items: Blueprint/DSE/Runner System

## 1. Design Space Explosion

- [ ] Implement a blueprint linting tool to warn users if their design space exceeds a configurable threshold (e.g., >10,000 combinations), with actionable suggestions.
- [ ] Integrate DSE strategies that can prune the design space before evaluation (sampling, clustering, surrogate models).
- [ ] Allow users to specify progressive exploration phases (coarse → fine), where only a subset of the space is explored initially, and further exploration is conditional on results.

## 2. Entrypoint Mapping Fragility

- [ ] Develop an explicit, versioned mapping layer (YAML or Python dict) that defines how blueprint sections (nodes/transforms) map to FINN entrypoints and steps.
- [ ] Validate the mapping layer at startup and in CI to catch mismatches between blueprint schema and FINN step definitions.
- [ ] Add automated tests that verify the mapping for all supported blueprint constructs and FINN step versions, with clear error messages if mismatches are detected.

---

## Explanation: Entrypoint Mapping Fragility & Explicit Mapping Layer

### What is "Entrypoint Mapping Fragility"?

**Entrypoint Mapping Fragility** refers to the risk that the internal logic translating user-friendly blueprint sections (like `nodes` and `transforms`) into the actual sequence of FINN entrypoint steps is brittle or opaque. If either the blueprint schema or the FINN step definitions change (e.g., new transforms, renamed steps, or altered entrypoint semantics), the mapping logic may silently break, leading to:
- Silent failures (e.g., steps not executed as intended)
- Hard-to-debug errors (e.g., missing or misordered steps)
- Increased maintenance burden as both blueprints and FINN evolve

This fragility is especially problematic in a system where the user interface is intentionally abstracted from the underlying execution details.

### What is an "Explicit Mapping Layer"?

An **Explicit Mapping Layer** is a clear, versioned, and declarative specification (such as a YAML file or Python dictionary) that defines exactly how each blueprint section or keyword maps to the corresponding FINN entrypoint(s) and step(s). For example:

```yaml
mapping_v2:
  nodes:
    canonical_ops: entrypoint_1
    hw_kernels: [entrypoint_3, entrypoint_4]
  transforms:
    model_topology: entrypoint_2
    hw_kernel: entrypoint_5
    hw_graph: entrypoint_6
  step_translation:
    cleanup: step_cleanup
    streamlining: step_streamlining
    matmul_rtl: step_register_matmul_rtl
    # etc.
```

**Benefits:**
- **Transparency:** Anyone can see how blueprint constructs are mapped.
- **Maintainability:** When FINN or blueprint schemas change, only the mapping file needs updating.
- **Testability:** Automated tests can verify that all mappings are valid and up-to-date.
- **Versioning:** Different mapping files can be maintained for different versions of FINN or blueprint schemas.

**In summary:**  
Entrypoint Mapping Fragility is the risk of silent or hard-to-debug failures due to implicit, brittle translation logic between blueprints and FINN steps. An Explicit Mapping Layer mitigates this by making the translation declarative, versioned, and testable.

---