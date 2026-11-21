"""Atomic JSONL Prediction Store for ALOE (Protocol v1).

Persists per-question, per-seed raw model generations and strict AST evaluation
outcomes incrementally so interrupted evaluations resume cleanly and evaluator
updates never require GPU retraining.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aloe.evaluation.tool_calling import EVALUATOR_VERSION, ASTToolCallEvaluator, EvaluationSummary


class PredictionRecord(BaseModel):
    """Single per-question generation and evaluation record."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    split: str
    model_id: str
    model_revision: str
    arm: str
    trial_id: str
    seed: int
    prompt_sha256: str
    raw_output: str
    ground_truth: list[dict[str, Any]] = Field(default_factory=list)
    available_tools: list[dict[str, Any]] | None = None
    evaluator_version: str = EVALUATOR_VERSION
    parse_success: bool
    schema_valid: bool = True
    tool_match: bool
    arg_names_match: bool
    arg_values_match: bool
    exact_match: bool
    lenient_exact_match: bool = False
    parse_error: str | None = None


class PredictionStore:
    """Incremental, crash-resumable JSONL store for trial predictions."""

    def __init__(
        self,
        jsonl_path: Path | str,
        *,
        on_write_callback: Any | None = None,
    ) -> None:
        self.path = Path(jsonl_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._on_write_callback = on_write_callback
        if self.path.is_file():
            self._repair_torn_tail_and_validate()

    def _notify_write(self) -> None:
        if callable(self._on_write_callback) and self.path.is_file():
            self._on_write_callback(self.path)

    def _repair_torn_tail_and_validate(self) -> list[PredictionRecord]:
        """Repair a crash-torn final line on initialization while failing closed on mid-file corruption or duplicates."""
        if not self.path.is_file():
            return []
        valid_records: list[PredictionRecord] = []
        seen: set[str] = set()
        had_corrupt_tail = False
        lines = self.path.read_text(encoding="utf-8").splitlines()
        for idx, raw_line in enumerate(lines):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = PredictionRecord.model_validate_json(line)
            except Exception as err:
                if idx == len(lines) - 1:
                    had_corrupt_tail = True
                    continue
                raise ValueError(f"Mid-file corruption in {self.path}:{idx + 1}: {err}") from err

            if rec.sample_id in seen:
                raise ValueError(f"Duplicate sample_id '{rec.sample_id}' in {self.path}:{idx + 1}")
            seen.add(rec.sample_id)
            valid_records.append(rec)

        if had_corrupt_tail:
            self._rewrite_atomic(valid_records)
        return valid_records

    def load_records(self, *, strict: bool = True) -> list[PredictionRecord]:
        """Load all persisted prediction records, rejecting corrupt lines or duplicate IDs in strict mode."""
        if not self.path.is_file():
            return []

        records: list[PredictionRecord] = []
        seen_ids: set[str] = set()
        with open(self.path, encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    rec = PredictionRecord.model_validate_json(line)
                except Exception as err:
                    if strict:
                        raise ValueError(
                            f"Corrupt prediction JSONL at {self.path}:{line_no}: {err}"
                        ) from err
                    continue

                if rec.sample_id in seen_ids:
                    if strict:
                        raise ValueError(
                            f"Duplicate sample_id '{rec.sample_id}' in {self.path}:{line_no}"
                        )
                    continue
                seen_ids.add(rec.sample_id)
                records.append(rec)
        return records

    def completed_sample_ids(self) -> set[str]:
        """Return set of sample IDs already persisted (enforcing identical duplicate/corruption checks as strict scoring)."""
        recs = self._repair_torn_tail_and_validate()
        return {r.sample_id for r in recs}

    def append(self, record: PredictionRecord) -> None:
        """Append one prediction record and flush + fsync to disk."""
        if self.path.is_file():
            existing = self.completed_sample_ids()
            if record.sample_id in existing:
                raise ValueError(f"Duplicate sample_id '{record.sample_id}' on append to {self.path}")
        line = record.model_dump_json() + "\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        self._notify_write()

    def _rewrite_atomic(self, records: list[PredictionRecord]) -> None:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as handle:
                tmp_path = Path(handle.name)
                for rec in records:
                    handle.write(rec.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.path)
            dir_fd = os.open(str(self.path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        self._notify_write()

    def rescore(
        self,
        evaluator: ASTToolCallEvaluator | None = None,
        *,
        update_trial_envelope: bool = True,
    ) -> EvaluationSummary:
        """Re-score stored raw outputs on CPU without GPU re-execution and atomically synchronize trial.json."""
        eval_engine = evaluator or ASTToolCallEvaluator(strict=True)
        records = self.load_records(strict=True)
        updated: list[PredictionRecord] = []
        for rec in records:
            ev = eval_engine.evaluate_single(
                rec.raw_output,
                rec.ground_truth,
                sample_id=rec.sample_id,
                available_tools=rec.available_tools,
            )
            updated.append(
                rec.model_copy(
                    update={
                        "evaluator_version": EVALUATOR_VERSION,
                        "parse_success": ev.parse_success,
                        "schema_valid": ev.schema_valid,
                        "tool_match": ev.tool_match,
                        "arg_names_match": ev.arg_names_match,
                        "arg_values_match": ev.arg_values_match,
                        "exact_match": ev.exact_match,
                        "lenient_exact_match": ev.lenient_exact_match,
                        "parse_error": ev.parse_error,
                    }
                )
            )
        self._rewrite_atomic(updated)
        summary = self.summarize(updated)

        trial_json_path = self.path.parent / "trial.json"
        if update_trial_envelope and trial_json_path.is_file():
            import json as _json
            from aloe.tracking.trial_store import atomic_write_text

            tdata = _json.loads(trial_json_path.read_text(encoding="utf-8"))
            metrics = tdata.get("metrics")
            if isinstance(metrics, dict) and metrics.get("predictions_jsonl_path") in (
                str(self.path),
                None,
            ):
                new_sha = self.sha256()
                metrics["predictions_sha256"] = new_sha
                metrics["evaluator_version"] = EVALUATOR_VERSION
                metrics["eval_samples"] = summary.total_samples
                metrics["tool_accuracy"] = summary.exact_match_accuracy
                metrics["ast_parse_success_rate"] = summary.parse_success_rate
                metrics["ast_schema_valid_rate"] = summary.schema_valid_rate
                metrics["ast_tool_selection_accuracy"] = summary.tool_selection_accuracy
                metrics["ast_argument_name_accuracy"] = summary.argument_name_accuracy
                metrics["ast_argument_value_accuracy"] = summary.argument_value_accuracy
                metrics["ast_exact_match_accuracy"] = summary.exact_match_accuracy
                metrics["ast_lenient_exact_match_accuracy"] = summary.lenient_exact_match_accuracy
                atomic_write_text(trial_json_path, _json.dumps(tdata, indent=2))

                # Propagate rescored accuracy atomically into parent campaign_ledger.json & campaign_summary.json
                trials_parent = trial_json_path.parent.parent
                if trials_parent.name == "trials":
                    camp_dir = trials_parent.parent
                    ledger_path = camp_dir / "campaign_ledger.json"
                    summary_path = camp_dir / "campaign_summary.json"
                    tid = trial_json_path.parent.name
                    if ledger_path.is_file():
                        ldata = _json.loads(ledger_path.read_text(encoding="utf-8"))
                        for ent in ldata.get("entries", []):
                            if ent.get("trial_id") == tid and isinstance(ent.get("metrics"), dict):
                                ent["metrics"]["exact_match_accuracy"] = summary.exact_match_accuracy
                                ent["metrics"]["tool_accuracy"] = summary.exact_match_accuracy
                        atomic_write_text(ledger_path, _json.dumps(ldata, indent=2))
                        done_ents = [
                            ent for ent in ldata.get("entries", [])
                            if ent.get("status") == "DONE" and isinstance(ent.get("metrics"), dict)
                        ]
                        if summary_path.is_file() and done_ents:
                            sdata = _json.loads(summary_path.read_text(encoding="utf-8"))
                            winner_ent = sorted(
                                done_ents,
                                key=lambda e: (
                                    -float(e["metrics"].get("exact_match_accuracy", e["metrics"]["tool_accuracy"])),
                                    float(e["metrics"]["search_memory_mb"]),
                                    float(e["metrics"]["trainable_parameters"]),
                                ),
                            )[0]
                            sdata["selected_winner_trial_id"] = winner_ent["trial_id"]
                            sdata["selected_winner_metrics"] = winner_ent["metrics"]
                            sdata["selected_winner_config"] = winner_ent["trial_config"]
                            from aloe.optimization.objectives import MultiObjectivePoint
                            from aloe.optimization.pareto import extract_pareto_frontier

                            points = [
                                MultiObjectivePoint(
                                    point_id=ent["trial_id"],
                                    objectives={
                                        "exact_match_accuracy": ent["metrics"].get(
                                            "exact_match_accuracy",
                                            ent["metrics"]["tool_accuracy"],
                                        ),
                                        "trainable_parameters": ent["metrics"]["trainable_parameters"],
                                        "search_memory_mb": ent["metrics"]["search_memory_mb"],
                                    },
                                )
                                for ent in done_ents
                            ]
                            pareto_front = extract_pareto_frontier(
                                points,
                                {
                                    "exact_match_accuracy": False,
                                    "trainable_parameters": True,
                                    "search_memory_mb": True,
                                },
                            )
                            sdata["pareto_trial_ids"] = [pt.point_id for pt in pareto_front]
                            atomic_write_text(summary_path, _json.dumps(sdata, indent=2))

        return summary

    @staticmethod
    def summarize(records: list[PredictionRecord]) -> EvaluationSummary:
        total = len(records)
        if total == 0:
            return EvaluationSummary()
        p_c = sum(1 for r in records if r.parse_success)
        s_c = sum(1 for r in records if r.schema_valid)
        t_c = sum(1 for r in records if r.tool_match)
        n_c = sum(1 for r in records if r.arg_names_match)
        v_c = sum(1 for r in records if r.arg_values_match)
        e_c = sum(1 for r in records if r.exact_match)
        l_c = sum(1 for r in records if r.lenient_exact_match)
        return EvaluationSummary(
            total_samples=total,
            parse_success_count=p_c,
            schema_valid_count=s_c,
            tool_match_count=t_c,
            arg_names_match_count=n_c,
            arg_values_match_count=v_c,
            exact_match_count=e_c,
            lenient_exact_match_count=l_c,
            parse_success_rate=round(p_c / total, 6),
            schema_valid_rate=round(s_c / total, 6),
            tool_selection_accuracy=round(t_c / total, 6),
            argument_name_accuracy=round(n_c / total, 6),
            argument_value_accuracy=round(v_c / total, 6),
            exact_match_accuracy=round(e_c / total, 6),
            lenient_exact_match_accuracy=round(l_c / total, 6),
        )

    def sha256(self) -> str:
        if not self.path.is_file():
            return "UNKNOWN"
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def build_paired_outcome_matrix(
    stores_by_seed: dict[int, PredictionStore],
    *,
    expected_sample_ids: set[str] | None = None,
    metric: str = "exact_match",
) -> dict[str, list[int]]:
    """Aggregate per-seed PredictionStores into `{sample_id: [0/1 per sorted seed]}` for paired_bootstrap.

    Enforces complete, identical question sets across all seeds.
    """
    if not stores_by_seed:
        raise ValueError("stores_by_seed cannot be empty")

    sorted_seeds = sorted(stores_by_seed.keys())
    per_seed_maps: dict[int, dict[str, int]] = {}
    reference_ids: set[str] | None = set(expected_sample_ids) if expected_sample_ids is not None else None

    for seed in sorted_seeds:
        recs = stores_by_seed[seed].load_records(strict=True)
        id_map: dict[str, int] = {}
        for r in recs:
            val = int(bool(getattr(r, metric)))
            id_map[r.sample_id] = val
        if reference_ids is None:
            reference_ids = set(id_map.keys())
        elif set(id_map.keys()) != reference_ids:
            missing = sorted(reference_ids - set(id_map.keys()))[:5]
            extra = sorted(set(id_map.keys()) - reference_ids)[:5]
            raise ValueError(
                f"Sample ID mismatch for seed {seed}: missing={missing}, extra={extra}"
            )
        per_seed_maps[seed] = id_map

    assert reference_ids is not None
    return {
        sid: [per_seed_maps[s][sid] for s in sorted_seeds]
        for sid in sorted(reference_ids)
    }
