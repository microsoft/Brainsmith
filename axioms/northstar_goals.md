# 🎯 **BrainSmith Core Design Axioms & North Star Goals**

*Distilled from API simplification, analysis hooks, and automation transformation work*

---

## 🌟 **North Star Vision**

> **"Make FPGA accelerator design as simple as calling a function."**

BrainSmith should be the **simplest, fastest way** to go from a neural network model to an optimized FPGA accelerator. Users should focus on their algorithms, not on learning complex toolchains.

---

## ⚖️ **Core Design Axioms**

### **Axiom 1: Simplicity Over Sophistication**
> *"A simple solution that works beats a sophisticated solution that confuses."*

- **One function to rule them all**: `forge(model, blueprint)` → optimized accelerator
- **Minimal cognitive load**: Users learn functions, not frameworks
- **No enterprise abstractions**: Direct problem-solving tools, not meta-tools
- **Reduce, don't expand**: Every addition must eliminate more complexity than it creates

**Anti-pattern**: Building enterprise workflow orchestration when users need parameter sweeps

### **Axiom 2: Focus Over Feature Creep**
> *"Do one thing extraordinarily well: FPGA accelerator design space exploration."*

- **Core competency**: Design space exploration (DSE) for FPGA accelerators
- **Not our job**: Custom analysis, workflow orchestration, enterprise features
- **User's job**: Algorithm development, performance analysis, deployment
- **Integration over invention**: Expose data for external tools vs custom implementations

**Anti-pattern**: Building custom statistical analysis when pandas/scipy exist

### **Axiom 3: Hooks Over Implementation**
> *"Expose structured data for external tools rather than reinventing the wheel."*

- **Data exposure**: Provide clean, structured data outputs
- **External integration**: Support pandas, scipy, scikit-learn, etc.
- **User choice**: Let users pick their preferred analysis/visualization tools
- **Maintenance reduction**: External libraries handle complex features

**Success example**: Analysis hooks that expose data for external tools vs custom analysis algorithms

### **Axiom 4: Functions Over Frameworks**
> *"Users want to call functions, not configure frameworks."*

- **Direct function calls**: `parameter_sweep()`, `batch_process()`, `find_best_result()`
- **Composable utilities**: Small functions that work together
- **No configuration objects**: Pass parameters directly to functions
- **Immediate utility**: Functions work without setup or learning curves

**Success example**: Automation helpers that replaced enterprise workflow engine

### **Axiom 5: Performance Over Purity**
> *"Fast and practical beats architecturally pure but slow."*

- **Time to value**: Users should get results in minutes, not hours
- **Parallel execution**: Use all available cores by default
- **Memory efficiency**: Handle large design spaces without exploding memory
- **Progress feedback**: Users should see what's happening

**Implementation**: ThreadPoolExecutor in automation helpers, streaming results

### **Axiom 6: Documentation Over Discovery**
> *"Users shouldn't have to discover how to use the tool."*

- **Clear examples**: Show exactly how to accomplish common tasks
- **Copy-paste ready**: Examples should work with minimal modification
- **Problem-focused**: Organize by what users want to achieve, not by module structure
- **Migration guides**: Help users transition from complex to simple approaches

**Success example**: Comprehensive READMEs with real-world usage examples

---

## 🚫 **Anti-Patterns to Avoid**

### **Enterprise Disease**
- ❌ **Workflow orchestration engines** for simple automation
- ❌ **Configuration objects** with dozens of parameters  
- ❌ **Abstract base classes** for simple operations
- ❌ **Dependency injection** for straightforward tools
- ❌ **Factory patterns** for function calls

### **Academic Over-Engineering**
- ❌ **Custom statistical algorithms** when scipy exists
- ❌ **ML learning systems** for design optimization
- ❌ **Research features** that 99% of users don't need
- ❌ **Theoretical completeness** over practical utility

### **Framework Bloat**
- ❌ **Plugin architectures** for core functionality
- ❌ **Event systems** for simple data flow
- ❌ **Configuration DSLs** for parameter passing
- ❌ **Lifecycle management** for stateless operations

---

## ✅ **Design Patterns to Embrace**

### **Functional Simplicity**
- ✅ **Pure functions** with clear inputs/outputs
- ✅ **Composable utilities** that work together
- ✅ **Immutable data** where practical
- ✅ **Predictable behavior** with minimal side effects

### **Progressive Disclosure**
- ✅ **Simple defaults** that work for 80% of cases
- ✅ **Optional parameters** for customization
- ✅ **Advanced functions** for power users
- ✅ **Escape hatches** to underlying functionality

### **Integration Philosophy**
- ✅ **Data exposure** for external analysis tools
- ✅ **Standard formats** (numpy arrays, pandas DataFrames)
- ✅ **Optional dependencies** for enhanced features
- ✅ **Interoperability** with existing workflows

---

## 🎯 **User Experience Goals**

### **Time to First Success**
- **5 minutes**: Users can run their first DSE
- **15 minutes**: Users understand parameter sweeps and batch processing
- **30 minutes**: Users can integrate with their analysis workflows
- **1 hour**: Users are productive with all major features

### **Cognitive Load Targets**
- **1 primary function**: `forge()` for core DSE
- **12 helper functions**: Automation and analysis utilities
- **0 configuration objects**: Direct parameter passing
- **3 core concepts**: Models, blueprints, parameters

### **Error Experience**
- **Clear error messages** with suggested fixes
- **Graceful degradation** when components fail
- **Progress feedback** for long operations
- **Helpful defaults** that usually work

---

## 🔄 **Evolution Principles**

### **Addition Criteria**
Before adding any new feature, ask:
1. **Does this eliminate more complexity than it adds?**
2. **Is this core to FPGA accelerator DSE?**
3. **Can users accomplish this with existing tools?**
4. **Does this maintain our simplicity promise?**

### **Removal Criteria**
Aggressively remove:
1. **Features used by <10% of users**
2. **Enterprise abstractions** that don't simplify common tasks
3. **Academic features** that duplicate existing tools
4. **Premature optimizations** that complicate the API

### **Refactoring Guidelines**
- **Simplify before optimizing**: Clear code before fast code
- **Reduce before extending**: Cut bloat before adding features
- **Externalize before implementing**: Use existing tools before building custom
- **Document while changing**: Update examples as code evolves

---

## 📏 **Success Metrics**

### **Code Quality Metrics**
- **Lines of code**: Trending down over time
- **API surface area**: Minimal essential functions only
- **Dependency count**: As few as practical
- **Setup complexity**: Single function call to get started

### **User Experience Metrics**
- **Time to first result**: < 5 minutes for new users
- **Documentation examples**: All should copy-paste and work
- **Support questions**: Should decrease as simplicity increases
- **User retention**: Simple tools keep users engaged

### **Performance Metrics**
- **DSE speed**: Continuously improving optimization speed
- **Memory efficiency**: Handle large design spaces
- **Parallel utilization**: Use available cores effectively
- **Result quality**: Maintain optimization quality while simplifying

---

## 🧭 **Decision Framework**

When facing design decisions, use this priority order:

1. **🎯 Simplicity**: Does this make the user's job simpler?
2. **🔧 Functionality**: Does this solve a real FPGA DSE problem?
3. **⚡ Performance**: Does this make DSE faster or more efficient?
4. **🔗 Integration**: Does this work well with user's existing tools?
5. **📚 Maintainability**: Can we support this long-term without complexity growth?

---

## 🎉 **The BrainSmith Promise**

> **"FPGA accelerator design should be as simple as:**
> ```python
> result = brainsmith.forge('model.onnx', 'blueprint.yaml')
> best = brainsmith.find_best_result(results, metric='throughput')
> ```
> **Everything else is optional."**

### **What Success Looks Like**
- **New users** get results in their first 5 minutes
- **Expert users** accomplish complex workflows with simple compositions
- **Tool maintainers** spend time on optimization, not complexity management
- **The ecosystem** thrives because BrainSmith plays well with other tools

### **What We Will Not Become**
- ❌ An enterprise workflow orchestration platform
- ❌ A machine learning research framework  
- ❌ A general-purpose optimization toolkit
- ❌ A complex system that requires extensive training

### **What We Will Always Be**
- ✅ The **simplest** way to explore FPGA accelerator design spaces
- ✅ The **fastest** path from model to optimized accelerator
- ✅ The **most practical** tool for FPGA DSE workflows
- ✅ The **best integrated** solution with existing data science tools

---

*These axioms guide every design decision in BrainSmith. When in doubt, choose simplicity.*