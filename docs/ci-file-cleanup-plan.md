# CI File Cleanup Plan - Post Reusable Workflows Refactoring

## Files That Can Be Eliminated

Based on the successful transition to reusable workflows, several files are now obsolete and can be safely removed.

### ❌ Files to DELETE

#### 1. **`.github/actions/setup-and-test/action.yml`**
**Status**: OBSOLETE
**Reason**: 
- Original composite action approach failed due to checkout requirements
- Functionality completely replaced by reusable workflows
- No longer referenced by any workflow

**Replacement**: The three reusable workflows handle all the functionality this composite action was supposed to provide.

#### 2. **`.github/workflows/ci-with-anchors.yml`** (if exists)
**Status**: OBSOLETE  
**Reason**:
- Appears to be an experimental/alternative implementation
- Not referenced in any documentation
- Main `ci.yml` is the active workflow

#### 3. **Legacy Documentation Files** (candidates for removal)
**Files**:
- `docs/ci-workflow-refactoring-plan.md` - Original planning document
- `docs/ci-workflow-completion-plan.md` - Intermediate implementation plan  
- `docs/ci-refactoring-implementation-summary.md` - Old summary

**Status**: CANDIDATE for archival
**Reason**:
- Historical planning documents no longer needed for operation
- Superseded by `docs/reusable-workflows-implementation-summary.md`
- Could be moved to `docs/archive/` if historical record is desired

### ✅ Files to KEEP

#### Core Active Files
- **`.github/workflows/ci.yml`** - Main orchestrator workflow
- **`.github/workflows/run-smithy-test.yml`** - Test execution workflow
- **`.github/workflows/build-and-push.yml`** - Build and push workflow  
- **`.github/workflows/setup-test-env.yml`** - Environment setup workflow
- **`.github/scripts/ci-common.sh`** - Shell operations library

#### Documentation Files  
- **`.github/workflows/README.md`** - Current usage guide
- **`docs/reusable-workflows-implementation-summary.md`** - Current architecture documentation
- **`docs/ci-security-vulnerability-analysis.md`** - Security analysis
- **`docs/ci-workflow-composability-options.md`** - Architecture decisions

## Impact Analysis

### Removing `.github/actions/setup-and-test/action.yml`

**Safe to remove because**:
1. ✅ No workflows reference it (all use reusable workflows now)
2. ✅ Functionality completely replaced
3. ✅ Was causing checkout issues that led to the reusable workflow approach

**Verification**: Check that no workflows contain `uses: ./.github/actions/setup-and-test`

### Legacy Workflow Files

**Safe to remove because**:
1. ✅ Not referenced in current documentation  
2. ✅ Main `ci.yml` is the active workflow
3. ✅ Would only cause confusion

## Cleanup Commands

```bash
# Remove obsolete composite action
rm -rf .github/actions/setup-and-test/

# Remove legacy workflow files (if they exist)
rm -f .github/workflows/ci-with-anchors.yml

# Archive old documentation (optional)
mkdir -p docs/archive/
mv docs/ci-workflow-refactoring-plan.md docs/archive/ 2>/dev/null || true
mv docs/ci-workflow-completion-plan.md docs/archive/ 2>/dev/null || true  
mv docs/ci-refactoring-implementation-summary.md docs/archive/ 2>/dev/null || true
```

## File Size Reduction

### Before Cleanup
```
.github/actions/setup-and-test/action.yml:     49 lines
.github/workflows/ci.yml:                    150 lines
.github/workflows/run-smithy-test.yml:        80 lines  
.github/workflows/build-and-push.yml:         73 lines
.github/workflows/setup-test-env.yml:         65 lines
.github/scripts/ci-common.sh:               120 lines
Total:                                       537 lines
```

### After Cleanup  
```
.github/workflows/ci.yml:                    150 lines
.github/workflows/run-smithy-test.yml:        80 lines
.github/workflows/build-and-push.yml:         73 lines  
.github/workflows/setup-test-env.yml:         65 lines
.github/scripts/ci-common.sh:               120 lines
Total:                                       488 lines
```

**Reduction**: 49 lines (9% less code to maintain)

## Benefits of Cleanup

### 1. **Reduced Confusion**
- Eliminates obsolete files that could mislead developers
- Clear separation between active and historical files
- Simplified `.github/` directory structure

### 2. **Easier Maintenance**  
- Fewer files to review during updates
- No risk of accidentally referencing obsolete components
- Cleaner git history going forward

### 3. **Better Developer Experience**
- Clear understanding of which files are active
- Reduced cognitive load when exploring the codebase
- Focus on current architecture, not historical experiments

## Recommended Approach

### Phase 1: Safe Removal (Immediate)
1. **Remove `.github/actions/setup-and-test/`** - Definitely obsolete
2. **Remove any `ci-with-anchors.yml`** - Experimental file
3. **Update documentation** to reflect cleanup

### Phase 2: Documentation Archival (Optional)
1. **Create `docs/archive/`** directory
2. **Move planning documents** to archive
3. **Update main README** to reference current docs only

### Phase 3: Verification (Required)
1. **Test CI pipeline** still works after cleanup
2. **Verify no broken references** in documentation  
3. **Update team** on file structure changes

## Risk Assessment

### Low Risk Removals
- ✅ **Composite action**: Not referenced anywhere
- ✅ **Legacy workflows**: Not in use

### Medium Risk Removals  
- ⚠️ **Planning documents**: Might be referenced in team discussions
- **Mitigation**: Archive instead of delete

### Zero Risk
- ✅ All active workflow files remain untouched
- ✅ All operational functionality preserved

## Conclusion

The transition to reusable workflows has made several files obsolete. Removing these files will:

1. **Simplify the codebase** by eliminating 9% of CI-related files
2. **Prevent confusion** by removing non-functional alternatives  
3. **Improve maintainability** with a cleaner file structure
4. **Preserve history** through optional archival of planning documents

The cleanup can be performed safely with minimal risk, as all removed files are either obsolete or superseded by the current implementation.