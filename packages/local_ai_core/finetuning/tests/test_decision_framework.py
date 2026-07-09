from local_ai_core.finetuning.decision_framework import Approach, recommend_approach


class TestRagWinsOnKnowledgeSignals:
    def test_private_or_changing_knowledge_routes_to_rag(self):
        result = recommend_approach(needs_private_or_changing_knowledge=True)
        assert result.approach == Approach.RAG

    def test_needing_citations_routes_to_rag(self):
        result = recommend_approach(must_cite_sources=True)
        assert result.approach == Approach.RAG

    def test_rag_signal_overrides_fine_tuning_readiness(self):
        # Curriculum's own rule: never fine-tune just to add knowledge that
        # changes often, even if every fine-tuning precondition also holds.
        result = recommend_approach(
            needs_private_or_changing_knowledge=True,
            output_style_is_repetitive=True,
            task_is_narrow_and_stable=True,
            has_enough_labeled_data=True,
            evaluation_proves_improvement=True,
        )
        assert result.approach == Approach.RAG


class TestFineTuningRequiresEveryPrecondition:
    def test_all_four_preconditions_true_recommends_fine_tuning(self):
        result = recommend_approach(
            output_style_is_repetitive=True,
            task_is_narrow_and_stable=True,
            has_enough_labeled_data=True,
            evaluation_proves_improvement=True,
        )
        assert result.approach == Approach.FINE_TUNING

    def test_missing_evaluation_proof_does_not_recommend_fine_tuning(self):
        result = recommend_approach(
            output_style_is_repetitive=True,
            task_is_narrow_and_stable=True,
            has_enough_labeled_data=True,
            evaluation_proves_improvement=False,
        )
        assert result.approach != Approach.FINE_TUNING

    def test_missing_labeled_data_does_not_recommend_fine_tuning(self):
        result = recommend_approach(
            output_style_is_repetitive=True,
            task_is_narrow_and_stable=True,
            has_enough_labeled_data=False,
            evaluation_proves_improvement=True,
        )
        assert result.approach != Approach.FINE_TUNING


class TestPromptingIsTheDefault:
    def test_a_simple_task_recommends_prompting(self):
        result = recommend_approach(task_is_simple=True)
        assert result.approach == Approach.PROMPTING

    def test_no_signals_at_all_defaults_to_prompting(self):
        result = recommend_approach()
        assert result.approach == Approach.PROMPTING


class TestReasonIsTraceable:
    def test_every_recommendation_carries_a_nonempty_reason(self):
        for kwargs in [
            {"needs_private_or_changing_knowledge": True},
            {
                "output_style_is_repetitive": True,
                "task_is_narrow_and_stable": True,
                "has_enough_labeled_data": True,
                "evaluation_proves_improvement": True,
            },
            {},
        ]:
            result = recommend_approach(**kwargs)
            assert len(result.reason) > 0
