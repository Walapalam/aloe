# ALOE Stage 7 / Gate A: xLAM Tool-Disjoint Pilot Split Report

> **Gate A Evaluation: PASS — Zero-leakage tool-disjoint partition constructed across training and validation sets.**

## 1. Dataset Provenance & Metadata
- **Source Dataset**: `Salesforce/xlam-function-calling-60k`
- **Revision SHA**: `26d14ebfe18b1f7b524bd39b404b50af5dc97866`
- **License**: `cc-by-4.0`
- **Total Source Records**: `60,000`
- **Total Source Unique Tools**: `3,605`

## 2. Tool-Disjoint Split Verification

| Partition | Sample Count | Unique Tools Assigned | ID List SHA-256 Digest |
|---|---|---|---|
| **Train** | **800** | 960 tools | `c035d607acb183fd...` |
| **Validation** | **200** | 142 tools | `2511c8b404775eaf...` |
| **Tool Intersection** | — | **0 tools (EMPTY)** | — |
| **Discarded Cross-Tools** | 22,614 | — | — |

## 3. Stratified Category Distribution

### Train Split Category Composition
- `multi_plausible_tools`: **525** (65.6%)
- `obvious_single_tool`: **275** (34.4%)

### Validation Split Category Composition
- `multi_plausible_tools`: **39** (19.5%)
- `obvious_single_tool`: **161** (80.5%)

## 4. Disjointness Guarantee & Research Rigour
- **Zero Signature Memorisation**: Because the intersection of tool schemas between train and validation is mathematically empty, fine-tuned adapters cannot score on validation by recalling specific parameter names, types, or signatures.
- **Generalised Tool Calling**: Validation performance strictly measures the SLM's capability to read an unseen tool declaration in the context prompt, map user intent, and format correct JSON arguments.
- **Deterministic Reproducibility**: The manifest stores sorted ID lists and SHA-256 digests (`data/manifests/xlam_pilot_v1.json`).
