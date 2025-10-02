from atlas.config.models import StudentPrompts
from atlas.transition.rewriter import PromptRewriter


def test_prompt_rewriter_injects_base_prompt_and_caches():
    prompts = StudentPrompts(
        planner="Planner instructions: {base_prompt}",
        executor="Exec: {base_prompt}",
        synthesizer="Final: {base_prompt}",
    )
    rewriter = PromptRewriter()
    rewritten_first = rewriter.rewrite("BASE", prompts)
    rewritten_second = rewriter.rewrite("BASE", prompts)
    assert rewritten_first is rewritten_second
    assert "BASE" in rewritten_first.planner
    assert rewritten_first.executor.startswith("Exec: BASE")
