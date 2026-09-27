from aloe.architecture.schema import (ArchitectureDescription, ComponentType,
                                      ModuleRecord, Validity)
from aloe.search_space.grammar import generate


def _desc():
    return ArchitectureDescription(
        model_id="dummy", family="gemma4", num_layers=2, total_params=100,
        topology="interleaved",
        modules=[
            ModuleRecord(name="l.0.self_attn.q_proj", module_class="Linear", layer_index=0,
                         param_count=10, is_linear=True,
                         component_type=ComponentType.LOCAL_ATTENTION, validity=Validity.VALID,
                         validity_reason="Standard nn.Linear target."),
            ModuleRecord(name="l.1.self_attn.q_proj", module_class="Linear", layer_index=1,
                         param_count=10, is_linear=True,
                         component_type=ComponentType.GLOBAL_ATTENTION, validity=Validity.VALID,
                         validity_reason="Standard nn.Linear target."),
            ModuleRecord(name="l.0.conv1d", module_class="Conv1d", layer_index=0,
                         param_count=5, component_type=ComponentType.SSM,
                         validity=Validity.EXCLUDED,
                         validity_reason="Bypasses LoRA forward pass."),
        ])


def test_groups_follow_component_types():
    g = generate(_desc(), backend="cuda_peft")
    assert {x.component_type for x in g.groups} == {
        ComponentType.LOCAL_ATTENTION, ComponentType.GLOBAL_ATTENTION}


def test_excluded_modules_keep_their_reason():
    g = generate(_desc(), backend="cuda_peft")
    assert "l.0.conv1d" in g.excluded
    assert g.excluded["l.0.conv1d"]


def test_every_valid_module_has_a_reason():
    for m in _desc().modules:
        assert m.validity_reason, f"{m.name} has no reason"


def test_alpha_range_is_wider_than_2r():
    """ADR-007 / Gap 2 — alpha must not be tied to the 2r heuristic."""
    g = generate(_desc(), backend="cuda_peft")
    for grp in g.groups:
        assert max(grp.alpha) > 2 * max(grp.rank)


def test_search_grammar_cardinality_collapsed_states():
    """Verify that disabled groups collapse to a single distinct state.
    For 2 groups with 4 ranks and 5 alphas: (1 + 4*5)^2 - 1 = 21^2 - 1 = 440.
    """
    g = generate(_desc(), backend="cuda_peft")
    assert len(g.groups) == 2
    # Without all-disabled: 21^2 - 1 = 440
    assert g.cardinality() == 440
    # With all-disabled: 21^2 = 441
    assert g.cardinality(include_all_disabled=True) == 441

