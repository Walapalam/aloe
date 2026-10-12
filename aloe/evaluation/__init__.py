"""ALOE Evaluation Engine.

Provides tool-calling AST evaluation, resource monitoring,
non-parametric paired bootstrap testing, and reproducible EvalCard generators.
"""
from __future__ import annotations

from aloe.evaluation.bootstrap import PairedBootstrapResult, paired_bootstrap
from aloe.evaluation.eval_card import (
    EvalCard,
    EvalCardEnvironment,
    EvalCardMetrics,
    EvalCardModel,
    EvalCardOptimizer,
    EvalCardSearchPoint,
)
from aloe.evaluation.tool_calling import (
    ASTToolCallEvaluator,
    EvaluationSummary,
    ToolCallEvaluation,
    ToolCallInstance,
)

__all__ = [
    "ASTToolCallEvaluator",
    "EvalCard",
    "EvalCardEnvironment",
    "EvalCardMetrics",
    "EvalCardModel",
    "EvalCardOptimizer",
    "EvalCardSearchPoint",
    "EvaluationSummary",
    "PairedBootstrapResult",
    "ToolCallEvaluation",
    "ToolCallInstance",
    "paired_bootstrap",
]
