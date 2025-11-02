# Documentation Redundancy Audit Report

**Date:** 2025-11-02  
**Scope:** All docs except `docs/evaluation/`  
**Status:** ✅ Complete

## Executive Summary

Found **3 major redundancy areas** and **2 minor overlaps**:

1. **Quickstart coverage** - Duplicated across `pypi.md`, `quickstart.mdx`, and `README.md`
2. **Autodiscovery** - Overlaps between `introduction.mdx` and `pypi.md`
3. **Configuration** - Minimal config details in `pypi.md` duplicate comprehensive `configuration.md`

## Detailed Findings

### 🔴 Major Redundancies

#### 1. Quickstart Documentation (3 locations)

**Files:**
- `docs/sdk/quickstart.mdx` (175 lines) - Dedicated quickstart command guide
- `docs/guides/pypi.md` (236 lines) - PyPI README with quickstart section
- `README.md` (259 lines) - Main README with quickstart section

**Overlap:**
- All three document the `atlas quickstart` command
- Installation steps duplicated
- Command options duplicated
- Troubleshooting duplicated

**Recommendation:**
- ✅ Keep `docs/sdk/quickstart.mdx` as the canonical quickstart guide
- ✅ `pypi.md` should reference `quickstart.mdx` instead of duplicating content
- ✅ `README.md` should reference `quickstart.mdx` for detailed usage

**Action:** 
- Remove quickstart details from `pypi.md` (lines 59-67, 96-108) → reference `quickstart.mdx`
- Keep minimal quickstart mention in `README.md` (appropriate for overview)

#### 2. Autodiscovery Documentation (2 locations)

**Files:**
- `docs/guides/introduction.mdx` (41 lines) - Dedicated autodiscovery guide
- `docs/guides/pypi.md` (236 lines) - Brief autodiscovery mention

**Overlap:**
- Both describe `atlas env init` workflow
- Both mention discovery process and generated files

**Recommendation:**
- ✅ Keep `introduction.mdx` as the canonical autodiscovery guide
- ✅ `pypi.md` should reference `introduction.mdx` instead of brief mention

**Action:**
- Remove autodiscovery details from `pypi.md` → add reference to `introduction.mdx`
- Keep minimal mention in `README.md` (appropriate for overview)

#### 3. Configuration Documentation (2 locations)

**Files:**
- `docs/configs/configuration.md` (350 lines) - Comprehensive configuration guide
- `docs/guides/pypi.md` (236 lines) - Minimal config section (lines 47-94)

**Overlap:**
- Basic config structure duplicated
- Example YAML configs duplicated
- LLM/provider setup duplicated

**Recommendation:**
- ✅ Keep `configuration.md` as the canonical configuration reference
- ✅ `pypi.md` should reference `configuration.md` for detailed config options
- ✅ `pypi.md` can keep minimal "getting started" config example

**Action:**
- Reduce config section in `pypi.md` to minimal example → add reference to `configuration.md`
- Keep comprehensive details in `configuration.md`

### 🟡 Minor Overlaps

#### 4. Terminal Telemetry vs Export Traces

**Files:**
- `docs/examples/terminal_telemetry.md` - Terminal output walkthrough
- `docs/examples/export_runtime_traces.md` - Export workflow

**Status:** ✅ **No redundancy** - Different purposes:
- Terminal telemetry = real-time streaming
- Export traces = batch export to JSONL

**Recommendation:** Keep both (complementary)

#### 5. Storage/PostgreSQL Setup

**Files:**
- `docs/guides/pypi.md` - Storage setup (lines 195-208)
- `docs/configs/configuration.md` - Storage config (lines 264-276)
- `docs/sdk/quickstart.mdx` - Storage check (lines 76-89)

**Status:** ✅ **Acceptable overlap** - Each serves different context:
- `pypi.md` = quick setup for getting started
- `configuration.md` = comprehensive config reference
- `quickstart.mdx` = command-specific storage behavior

**Recommendation:** Keep as-is (context-specific)

## Recommended Actions

### Priority 1: Consolidate Quickstart Documentation

1. **Update `docs/guides/pypi.md`:**
   - Remove detailed quickstart section (lines 59-67, 96-108)
   - Add reference: "See [Atlas Quickstart](docs/sdk/quickstart.mdx) for detailed command documentation"
   - Keep minimal mention: "Run `atlas quickstart` to see learning in action"

2. **Update `README.md`:**
   - Keep minimal quickstart mention (already appropriate)
   - Ensure it references `docs/sdk/quickstart.mdx` for details

### Priority 2: Consolidate Autodiscovery Documentation

1. **Update `docs/guides/pypi.md`:**
   - Remove autodiscovery details from "What's New" section
   - Add reference: "See [Autodiscovery Guide](docs/guides/introduction.mdx) for `atlas env init` documentation"

### Priority 3: Consolidate Configuration Documentation

1. **Update `docs/guides/pypi.md`:**
   - Reduce config section to minimal example (keep lines 47-94 but simplify)
   - Add reference: "See [Configuration Guide](docs/configs/configuration.md) for comprehensive configuration options"
   - Remove detailed config explanations (leave those in `configuration.md`)

## Files to Update

1. ✅ `docs/guides/pypi.md` - Remove redundant sections, add references
2. ✅ `README.md` - Verify references are correct (likely minimal changes needed)

## Files to Keep Unchanged

- ✅ `docs/sdk/quickstart.mdx` - Canonical quickstart guide
- ✅ `docs/guides/introduction.mdx` - Canonical autodiscovery guide
- ✅ `docs/configs/configuration.md` - Canonical configuration reference
- ✅ `docs/examples/terminal_telemetry.md` - Unique content
- ✅ `docs/examples/export_runtime_traces.md` - Unique content
- ✅ `docs/operations/guardrails.md` - Unique content
- ✅ `docs/evaluation/*` - Explicitly excluded from audit

## Estimated Impact

**Lines to remove:** ~150 lines of redundant content  
**References to add:** ~6 cross-references  
**Benefit:** 
- Single source of truth for each topic
- Easier maintenance (update once, not multiple places)
- Better user experience (clear navigation between docs)

## Next Steps

1. Review and approve this audit
2. Update `pypi.md` with recommended changes
3. Verify `README.md` references are correct
4. Test documentation links work correctly
5. Commit changes with message: "docs: consolidate redundant documentation"

