# BrainSmith Foundation Axioms

*Concise principles to prevent over-engineering*

## Core Axioms

### 1. Foundation Over Features
**Build contributor-ready infrastructure, not end-user features.**
- Goal: Enable domain experts to contribute across all libraries
- Success: Can a collaborator add kernels, blueprints, transforms without core changes?
- Anti-pattern: Building end-user polish before foundation is solid

### 2. Cross-Domain Collaboration
**Expect collaborators to work across all libraries simultaneously.**
- Libraries are interconnected, not isolated domains
- Registry patterns enable coordination, not separation
- Anti-pattern: Treating libraries as independent silos

### 3. Dual Interface Strategy  
**Complex collaborative foundation, simple `forge()` interface.**
- Sophisticated infrastructure hidden behind simple API
- Collaborators see registries/libraries, users see functions
- Anti-pattern: Exposing infrastructure complexity to end users

### 4. Minimal Viable Infrastructure
**Build only what's needed for effective collaboration.**
- Registry pattern justified only if it enables collaboration
- Avoid enterprise patterns until explicitly needed
- Anti-pattern: Building orchestration when simple patterns work

### 5. Collaboration Over Orchestration
**Enable distributed expertise, don't orchestrate workflows.**
- Autodiscovery so adding components doesn't require core changes
- Convention over configuration for contributors
- Anti-pattern: Complex workflow engines for simple automation

### 6. Performance Over Purity
**Fast and practical foundation beats architecturally pure but slow.**
- Contributor velocity matters most
- Simple patterns over theoretical completeness
- Anti-pattern: Academic perfection that slows development

## Quick Decision Framework

When tempted to add complexity, ask:
1. **Does this enable collaboration?** (Foundation goal)
2. **Is this minimal viable infrastructure?** (Avoid over-engineering)
3. **Can collaborators work in parallel?** (Cross-domain priority)
4. **Does `forge()` stay simple?** (End-user promise)

## Red Flags

🚨 **Stop if you're building:**
- Enterprise workflow orchestration
- Complex configuration systems
- Abstract base classes for simple operations
- Event systems for basic data flow
- Dependency injection for straightforward tools

✅ **Good signs:**
- Adding a component doesn't require core changes
- New contributors can focus on their domain
- Registry discovery "just works"
- End users still get simple `forge()` interface