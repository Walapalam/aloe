#!/usr/bin/env python3
"""Direct automated bridge to Google Colab browser MCP session via colab_mcp.

Opens a Colab scratch notebook with the local MCP WebSocket proxy parameters,
waits for the user to click Connect in the browser, and automatically adds
and runs the ALOE + LiquidAI/LFM3-1.7B verification cells in Google Colab.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from typing import Any
import webbrowser

from colab_mcp.session import ColabProxyClient
from colab_mcp.websocket_server import COLAB, SCRATCH_PATH, ColabWebSocketServer

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_OUT = (
    REPO_ROOT
    / "runs"
    / "gradients"
    / "LiquidAI--LFM3-1.7B"
    / "colab_session_transcript.json"
)

CELLS_TO_RUN = [
    {
        "title": "Cell 1: Colab Runtime & Hardware Inspection",
        "code": (
            "import platform, torch, psutil\n"
            "print(f'Platform: {platform.platform()}')\n"
            "print(f'PyTorch: {torch.__version__}')\n"
            "print(f'CUDA available: {torch.cuda.is_available()}')\n"
            "if torch.cuda.is_available():\n"
            "    print(f'GPU: {torch.cuda.get_device_name(0)}')\n"
            "    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB')\n"
            "print(f'System RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB')\n"
        ),
    },
    {
        "title": "Cell 2: Clone Walapalam/aloe & Install Dependencies in Colab",
        "code": (
            "import os, subprocess, sys\n"
            "if not os.path.exists('/content/aloe'):\n"
            "    subprocess.run(['git', 'clone', 'https://github.com/Walapalam/aloe.git', '/content/aloe'], check=True)\n"
            "else:\n"
            "    subprocess.run(['git', '-C', '/content/aloe', 'pull', '--ff-only'], check=True)\n"
            "os.chdir('/content/aloe')\n"
            "subprocess.run([\n"
            "    sys.executable, '-m', 'pip', 'install', '-q', '-U',\n"
            "    'transformers>=5.15.0', 'peft>=0.13.0', 'accelerate>=1.0.0',\n"
            "    'datasets>=3.0.0', 'botorch>=0.12.0', 'gpytorch>=1.13',\n"
            "    'pydantic>=2.9.0', 'rich>=13.9.0', 'typer>=0.12.5', 'pytest>=8.3.0'\n"
            "], check=True)\n"
            "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.'], check=True)\n"
            "import transformers, peft, botorch\n"
            "print(f'Installed in /content/aloe | transformers={transformers.__version__} peft={peft.__version__} botorch={botorch.__version__}')\n"
        ),
    },
    {
        "title": "Cell 3: Run Full ALOE Pytest Suite (107 Tests incl. LFM3 & Budget Parity)",
        "code": (
            "import os, subprocess, sys\n"
            "os.chdir('/content/aloe')\n"
            "res = subprocess.run([sys.executable, '-m', 'pytest', '-q'], capture_output=True, text=True)\n"
            "print(res.stdout)\n"
            "if res.stderr:\n"
            "    print(res.stderr)\n"
            "assert res.returncode == 0, f'Pytest failed with exit code {res.returncode}'\n"
        ),
    },
    {
        "title": "Cell 4: Live HuggingFace Hub Inspection of LiquidAI/LFM3-1.7B",
        "code": (
            "import json, os, subprocess, sys\n"
            "os.chdir('/content/aloe')\n"
            "res = subprocess.run([\n"
            "    sys.executable, '-m', 'aloe.cli.main', 'inspect', 'LiquidAI/LFM3-1.7B',\n"
            "    '--out', 'runs/grammars/lfm3_1_7b_colab.json'\n"
            "], capture_output=True, text=True)\n"
            "print(res.stdout)\n"
            "grammar = json.load(open('runs/grammars/lfm3_1_7b_colab.json'))\n"
            "print(json.dumps({\n"
            "    'model_id': grammar['model_id'],\n"
            "    'architecture_family': grammar['architecture_family'],\n"
            "    'topology': grammar['topology'],\n"
            "    'groups': {k: len(v) for k, v in grammar['groups'].items()},\n"
            "    'excluded_count': len(grammar['excluded_modules']),\n"
            "    'cardinality': grammar['search_space_cardinality'],\n"
            "}, indent=2))\n"
        ),
    },
    {
        "title": "Cell 5: LFM3-1.7B Pre-Flight Gradient Check, Negative Control & ±5% Budget-Matched SFT Pilot",
        "code": (
            "import json, os, subprocess, sys, torch\n"
            "os.chdir('/content/aloe')\n"
            "steps = '16' if torch.cuda.is_available() else '4'\n"
            "cmd = [\n"
            "    sys.executable, 'scripts/run_colab_lfm3_tests.py',\n"
            "    '--full-weights', '--run-sft-pilot', '--steps', steps,\n"
            "    '--out', 'runs/gradients/LiquidAI--LFM3-1.7B/colab_gpu_report.json'\n"
            "]\n"
            "print('Executing:', ' '.join(cmd))\n"
            "res = subprocess.run(cmd, capture_output=True, text=True)\n"
            "print(res.stdout)\n"
            "if res.stderr:\n"
            "    print('STDERR tail:', res.stderr[-1000:])\n"
            "assert res.returncode == 0, f'LFM3 test script exited with {res.returncode}'\n"
        ),
    },
]


def _extract_tool_json(res: Any) -> dict[str, Any]:
    for item in getattr(res, "content", []):
        txt = getattr(item, "text", None)
        if txt:
            try:
                return json.loads(txt)
            except Exception:
                return {"raw_text": txt}
    if getattr(res, "structured_content", None):
        return dict(res.structured_content)
    return {}


async def main() -> None:
    TRANSCRIPT_OUT.parent.mkdir(parents=True, exist_ok=True)
    async with ColabWebSocketServer() as wss:
        async with ColabProxyClient(wss) as proxy_client:
            url = f"{COLAB}{SCRATCH_PATH}#mcpProxyToken={wss.token}&mcpProxyPort={wss.port}"
            print(f"[ALOE Colab Runner] Opening Google Colab in browser:\n  {url}", flush=True)
            webbrowser.open_new(url)
            print(
                "[ALOE Colab Runner] Waiting up to 180s for you to click Allow/Connect in Colab...",
                flush=True,
            )

            deadline = time.time() + 180.0
            while time.time() < deadline:
                if proxy_client.is_connected():
                    break
                await asyncio.sleep(0.5)

            if not proxy_client.is_connected():
                print("[ALOE Colab Runner] ERROR: Timed out waiting for Colab browser connection.", flush=True)
                return

            client = proxy_client.proxy_mcp_client
            print("[ALOE Colab Runner] Connected to Google Colab runtime!", flush=True)

            tools = await client.list_tools()
            tool_names = [t.name for t in tools]
            print(f"[ALOE Colab Runner] Unlocked Colab MCP tools: {tool_names}", flush=True)

            session_records: list[dict[str, Any]] = []
            for idx, spec in enumerate(CELLS_TO_RUN):
                title = spec["title"]
                code = f"# {title}\n{spec['code']}"
                print(f"\n--- Adding {title} (cellIndex={idx}) ---", flush=True)
                add_res = await client.call_tool(
                    "add_code_cell",
                    {
                        "cellIndex": idx,
                        "language": "python",
                        "code": code,
                    },
                )
                add_payload = _extract_tool_json(add_res)
                cell_id = add_payload.get("newCellId")
                print(f"Created cellId={cell_id}. Executing on Colab runtime...", flush=True)

                t0 = time.time()
                run_res = await client.call_tool(
                    "run_code_cell",
                    {"cellId": cell_id},
                )
                elapsed = round(time.time() - t0, 2)
                run_payload = _extract_tool_json(run_res)

                stdout_lines: list[str] = []
                for out_item in run_payload.get("outputs", []):
                    txt_field = out_item.get("text", [])
                    if isinstance(txt_field, list):
                        stdout_lines.extend(txt_field)
                    elif isinstance(txt_field, str):
                        stdout_lines.append(txt_field)
                joined_output = "".join(stdout_lines)
                print(f"[{title}] finished in {elapsed}s:\n{joined_output}", flush=True)

                session_records.append(
                    {
                        "cellIndex": idx,
                        "cellId": cell_id,
                        "title": title,
                        "elapsed_sec": elapsed,
                        "output": joined_output,
                        "raw_response": run_payload,
                    }
                )
                TRANSCRIPT_OUT.write_text(
                    json.dumps(
                        {
                            "status": "running" if idx < len(CELLS_TO_RUN) - 1 else "completed",
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "colab_tools": tool_names,
                            "cells": session_records,
                        },
                        indent=2,
                    )
                )

            print(
                f"\n[ALOE Colab Runner] All {len(CELLS_TO_RUN)} Colab cells completed! Saved transcript to {TRANSCRIPT_OUT}",
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
