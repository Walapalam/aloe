# ALOE Stage 2 / R2: Model Acquisition & Registry Report

> **Gate 2 Evaluation: PASS — Both pilot models successfully acquired, revision SHAs pinned, configs loaded, and native tool-calling chat templates verified.**

## Gate 2 Verification Checklist
- [x] Both models successfully downloaded and local snapshots verified on disk
- [x] AutoConfig loads completely with all architectural fields preserved
- [x] AutoTokenizer loads with verified native chat template
- [x] Exact commit revision SHAs pinned (40-character hex string)
- [x] License terms recorded with exact URLs
- [x] Results adhere strictly to Standard Result Schema Envelope (`schema_version: '1.0'`)

## Pilot Models Registry

| Model ID | Revision SHA | Precision | Total Size | Files | License | Gated | Chat Template | Snapshot Path | Gate 2 |
|---|---|---|---|---|---|---|---|---|---|
| `google/gemma-4-E2B-it` | `3e22461f65e8...` | bf16 | 9.573 GB | 9 | apache-2.0 | No | Present | `/Users/walapalam/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7` | PASS |
| `google/gemma-4-E4B-it` | `ee0ef6023621...` | bf16 | 14.924 GB | 9 | apache-2.0 | No | Present | `/Users/walapalam/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2` | PASS |
| `tiiuae/Falcon-H1-1.5B-Instruct` | `80ebc50d7799...` | bf16 | 2.901 GB | 9 | falcon-llm-license | No | Present | `/Users/walapalam/.cache/huggingface/hub/models--tiiuae--Falcon-H1-1.5B-Instruct/snapshots/80ebc50d7799a440b96c93bb6686a3924a09b0cb` | PASS |

## Model Profiles & Verification Details

### `google/gemma-4-E2B-it`

- **Pinned Commit SHA**: `3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- **Local Snapshot**: `/Users/walapalam/.cache/huggingface/hub/models--google--gemma-4-E2B-it/snapshots/3e22461f65e89153144f8adb70e3b8c2cc9845a7`
- **Total Footprint**: 10,278,849,571 bytes (9.573 GB)
- **Precision**: `bf16` (Quantisation: `none`)
- **License**: `apache-2.0` — [Terms](https://ai.google.dev/gemma/docs/gemma_4_license)
- **Gated Access**: `False`
- **Architecture Class**: `Gemma4ForConditionalGeneration`
- **Decoder Layers**: `35` | **Hidden Size**: `1536`
- **Native Chat Template**: `Verified Present`

#### File Inventory

| File Path | Size (Bytes) | Symlink |
|---|---|---|
| `.gitattributes` | 1,570 | Yes |
| `README.md` | 27,955 | Yes |
| `chat_template.jinja` | 18,569 | Yes |
| `config.json` | 4,954 | Yes |
| `generation_config.json` | 208 | Yes |
| `model.safetensors` | 10,246,621,918 | Yes |
| `processor_config.json` | 1,689 | Yes |
| `tokenizer.json` | 32,169,626 | Yes |
| `tokenizer_config.json` | 3,082 | Yes |

#### Native Chat Template Verification

```jinja
{#
  Template: Google Gemma 4 Canonical Chat Template
  Author: Google Gemma Engineering Team
  Published: 2026-07-09
  Context: Fixed tool-calling loops, turn closures, and thinking content-ordering.
#}
{%- macro format_parameters(properties, required, filter_keys=false) -%}
    {%- set standard_keys = ['description', 'type', 'properties', 'required', 'nullable'] -%}
    {%- set ns = namespace(found_first=false) -%}
    {%- for key, value in properties | dictsort -%}
        {%- set add_comma = false -%}
        {%- if not filter_keys or key not in standard_keys -%}
... [374 additional lines omitted] ...
```

### `google/gemma-4-E4B-it`

- **Pinned Commit SHA**: `ee0ef6023621cff504d758262d4e04895a5af4a2`
- **Local Snapshot**: `/Users/walapalam/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2`
- **Total Footprint**: 16,024,823,729 bytes (14.924 GB)
- **Precision**: `bf16` (Quantisation: `none`)
- **License**: `apache-2.0` — [Terms](https://ai.google.dev/gemma/docs/gemma_4_license)
- **Gated Access**: `False`
- **Architecture Class**: `Gemma4ForConditionalGeneration`
- **Decoder Layers**: `42` | **Hidden Size**: `2560`
- **Native Chat Template**: `Verified Present`

#### File Inventory

| File Path | Size (Bytes) | Symlink |
|---|---|---|
| `.gitattributes` | 1,570 | Yes |
| `README.md` | 27,956 | Yes |
| `chat_template.jinja` | 18,569 | Yes |
| `config.json` | 5,145 | Yes |
| `generation_config.json` | 208 | Yes |
| `model.safetensors` | 15,992,595,884 | Yes |
| `processor_config.json` | 1,689 | Yes |
| `tokenizer.json` | 32,169,626 | Yes |
| `tokenizer_config.json` | 3,082 | Yes |

#### Native Chat Template Verification

```jinja
{#
  Template: Google Gemma 4 Canonical Chat Template
  Author: Google Gemma Engineering Team
  Published: 2026-07-09
  Context: Fixed tool-calling loops, turn closures, and thinking content-ordering.
#}
{%- macro format_parameters(properties, required, filter_keys=false) -%}
    {%- set standard_keys = ['description', 'type', 'properties', 'required', 'nullable'] -%}
    {%- set ns = namespace(found_first=false) -%}
    {%- for key, value in properties | dictsort -%}
        {%- set add_comma = false -%}
        {%- if not filter_keys or key not in standard_keys -%}
... [374 additional lines omitted] ...
```

### `tiiuae/Falcon-H1-1.5B-Instruct`

- **Pinned Commit SHA**: `80ebc50d7799a440b96c93bb6686a3924a09b0cb`
- **Local Snapshot**: `/Users/walapalam/.cache/huggingface/hub/models--tiiuae--Falcon-H1-1.5B-Instruct/snapshots/80ebc50d7799a440b96c93bb6686a3924a09b0cb`
- **Total Footprint**: 3,114,955,068 bytes (2.901 GB)
- **Precision**: `bf16` (Quantisation: `none`)
- **License**: `other` (falcon-llm-license) — [Terms](https://falconllm.tii.ae/falcon-terms-and-conditions.html)
- **Gated Access**: `False`
- **Architecture Class**: `FalconH1ForCausalLM`
- **Decoder Layers**: `24` | **Hidden Size**: `2048`
- **Native Chat Template**: `Verified Present`

#### File Inventory

| File Path | Size (Bytes) | Symlink |
|---|---|---|
| `.gitattributes` | 1,519 | Yes |
| `README.md` | 7,041 | Yes |
| `chat_template.jinja` | 1,151 | Yes |
| `config.json` | 1,599 | Yes |
| `generation_config.json` | 145 | Yes |
| `model.safetensors` | 3,109,773,032 | Yes |
| `special_tokens_map.json` | 7,693 | Yes |
| `tokenizer.json` | 5,062,983 | Yes |
| `tokenizer_config.json` | 99,905 | Yes |

#### Native Chat Template Verification

```jinja
{{bos_token}}
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "You are a function calling AI model. You are provided with function signature within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions.\n<tools>\n" }}
    {%- for tool in tools %}[{{- tool | tojson }}]{%- endfor %}
    {{- "\n</tools>\nFor each function call, return a json object with function name and arguments within <tool_call> </tool_call> tags with the following schema:\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>\n" }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
... [5 additional lines omitted] ...
```
