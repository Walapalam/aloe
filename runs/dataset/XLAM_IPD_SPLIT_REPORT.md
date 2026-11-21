# ALOE Protocol v1: 3-Way Stratified Tool-Disjoint xLAM Split Report

> **Verification: PASS — Zero pairwise tool overlap across Train (1,000), Validation (200), and Sealed Test (400).**

## 1. Dataset Provenance & Integrity
- **Source Dataset**: `Salesforce/xlam-function-calling-60k`
- **Pinned Revision SHA**: `45d1ebe7f73a99abfb2bf71a30caaaaf0570cf32`
- **Raw File SHA-256**: `4ef5c6f0dc552f2231f93f5853a9ef431e9e806d7aa514d0f6b615606ce576c6`
- **License**: `cc-by-4.0`
- **Total Source Records**: `60,000`
- **Overlength Filtered (`>3400` chars)**: `1,781`

## 2. Partition Summary

| Partition | Sample Count | Unique Tools | ID List SHA-256 |
|---|---:|---:|---|
| **Train** | **1000** | 812 | `d3e47bbba1604c3b...` |
| **Validation** | **200** | 140 | `664f29bfc62774f4...` |
| **Test (Sealed)** | **400** | 272 | `26cdf457f083336c...` |
| **Pairwise Tool Overlap** | — | **0 (train∩val=0, train∩test=0, val∩test=0)** | — |
| **Cross-Partition Discards** | 32,032 | — | — |

## 3. Stratified Category Proportions

### `train` (1000 items)
- `multi_plausible_tools`: **412** (41.2%)
- `obvious_single_tool`: **588** (58.8%)

### `validation` (200 items)
- `multi_plausible_tools`: **82** (41.0%)
- `obvious_single_tool`: **118** (59.0%)

### `test` (400 items)
- `multi_plausible_tools`: **165** (41.2%)
- `obvious_single_tool`: **235** (58.8%)
