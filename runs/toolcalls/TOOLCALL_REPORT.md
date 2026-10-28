# ALOE Stage 6 / R6: Baseline Tool-Call Output Probe Report

> **Gate 6 Evaluation: PASS — Zero-shot baseline tool-calling capabilities and abstention dynamics audited across 10 hand-written scenarios.**

## Gate 6 Verification Checklist
- [x] 10 diverse test scenarios probed (3 obvious, 2 multi-candidate, 2 missing args, 3 no-tool)
- [x] Model native chat templates applied with JSON schema tool definitions
- [x] Raw model outputs captured verbatim without truncation
- [x] Outputs audited for format parseability, tool name detection, and argument extraction
- [x] Abstention correctly classified on negative/unsupported controls
- [x] Standard Result Schema Envelope (`schema_version: '1.0'`) strictly adhered to

## 1. Summary Comparison

| Model ID | Total Prompts | Parseable Outputs | Tool Calls Emitted | Correct Abstentions | Baseline Parseability | Gate 6 Verdict |
|---|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | 10 | **10/10** | 3/10 | **5/5** | 100% | **PASS** |
| `google/gemma-4-E4B-it` | 10 | **10/10** | 3/10 | **5/5** | 100% | **PASS** |
| `tiiuae/Falcon-H1-1.5B-Instruct` | 10 | **10/10** | 7/10 | **2/5** | 100% | **PASS** |

---

## 2. Detailed Prompt Response Matrix

### Model: `google/gemma-4-E2B-it`

| Prompt ID | Category | Expected Action | Called Tool? | Detected Tool | Format | Abstain? |
|---|---|---|---|---|---|---|
| `T01_obvious_weather` | obvious_single_tool | get_weather | Yes | `get_weather` | json | — |
| `T02_obvious_calc` | obvious_single_tool | calculate_total | Yes | `calculate_total` | json | — |
| `T03_obvious_docs` | obvious_single_tool | search_docs | Yes | `search_docs` | json | — |
| `T04_plausible_travel` | multi_plausible_tools | search_flights_or_book_hotel | No | — | plain_text | ✅ |
| `T05_plausible_network` | multi_plausible_tools | ping_or_dns | No | — | plain_text | ✅ |
| `T06_missing_args_transfer` | missing_required_args | None (Abstain) | No | — | plain_text | ✅ |
| `T07_missing_args_deploy` | missing_required_args | None (Abstain) | No | — | plain_text | ✅ |
| `T08_notool_mutex` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T09_notool_haiku` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T10_notool_capital` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |

### Model: `google/gemma-4-E4B-it`

| Prompt ID | Category | Expected Action | Called Tool? | Detected Tool | Format | Abstain? |
|---|---|---|---|---|---|---|
| `T01_obvious_weather` | obvious_single_tool | get_weather | Yes | `get_weather` | json | — |
| `T02_obvious_calc` | obvious_single_tool | calculate_total | Yes | `calculate_total` | json | — |
| `T03_obvious_docs` | obvious_single_tool | search_docs | Yes | `search_docs` | json | — |
| `T04_plausible_travel` | multi_plausible_tools | search_flights_or_book_hotel | No | — | plain_text | ✅ |
| `T05_plausible_network` | multi_plausible_tools | ping_or_dns | No | — | plain_text | ✅ |
| `T06_missing_args_transfer` | missing_required_args | None (Abstain) | No | — | plain_text | ✅ |
| `T07_missing_args_deploy` | missing_required_args | None (Abstain) | No | — | plain_text | ✅ |
| `T08_notool_mutex` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T09_notool_haiku` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T10_notool_capital` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |

### Model: `tiiuae/Falcon-H1-1.5B-Instruct`

| Prompt ID | Category | Expected Action | Called Tool? | Detected Tool | Format | Abstain? |
|---|---|---|---|---|---|---|
| `T01_obvious_weather` | obvious_single_tool | get_weather | Yes | `get_weather` | json | — |
| `T02_obvious_calc` | obvious_single_tool | calculate_total | Yes | `calculate_total` | json | — |
| `T03_obvious_docs` | obvious_single_tool | search_docs | Yes | `search_docs` | json | — |
| `T04_plausible_travel` | multi_plausible_tools | search_flights_or_book_hotel | Yes | `search_flights` | json | — |
| `T05_plausible_network` | multi_plausible_tools | ping_or_dns | No | — | plain_text | ✅ |
| `T06_missing_args_transfer` | missing_required_args | None (Abstain) | Yes | `transfer_funds` | json | — |
| `T07_missing_args_deploy` | missing_required_args | None (Abstain) | Yes | `deploy_build` | json | — |
| `T08_notool_mutex` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T09_notool_haiku` | no_tool_needed | None (Abstain) | No | — | plain_text | ✅ |
| `T10_notool_capital` | no_tool_needed | None (Abstain) | Yes | `get_weather` | json | — |

---

## 3. Findings & Research Baseline Analysis

- **Base Model Zero-Shot Behavior**: As anticipated by Prompt 6, compact ~1.5B–2B base models without fine-tuning exhibit mixed zero-shot tool invocation, confirming the necessity of ALOE's architecture-typed LoRA optimization.
- **Plumbing Integrity**: Native tokenizers, chat templates, and greedy decoding pipelines function reliably without crashes on Apple Silicon MPS.
- **Baseline Benchmark Established**: These 10 baseline outputs provide the empirical zero-shot reference against which future fine-tuned adapters will be measured.
