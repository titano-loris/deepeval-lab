"""
Tests avec les métriques NATIVES DeepEval — AnswerRelevancy & Faithfulness.

C'est ici que le paradigme LLM-as-Judge entre en jeu :
pour chaque réponse du chatbot, le juge local est interrogé avec
des prompts d'évaluation générés par DeepEval, et retourne un score 0-1.

Différence fondamentale avec RAG-TestKit :
- RAG-TestKit : assert "25" in answer          → binaire, déterministe
- DeepEval-Lab : relevancy_score >= 0.6        → gradué, sémantique

Lancement : pytest tests/test_native_metrics.py -v -m judge
"""
import pytest
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

RELEVANCY_THRESHOLD = 0.6
FAITHFULNESS_THRESHOLD = 0.6


@pytest.mark.judge
@pytest.mark.slow
class TestAnswerRelevancy:
    """La réponse du chatbot répond-elle vraiment à la question posée ?"""

    def test_standard_faq_answer_is_relevant(self, chatbot, judge, eval_cases):
        """CASE001 — réinitialisation mot de passe : cas nominal."""
        case = next(c for c in eval_cases if c["id"] == "CASE001")
        answer = chatbot.answer(case["question"], case["context"])

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
        )
        metric = AnswerRelevancyMetric(
            threshold=RELEVANCY_THRESHOLD, model=judge, async_mode=False
        )
        metric.measure(test_case)

        assert metric.score >= RELEVANCY_THRESHOLD, (
            f"Pertinence insuffisante ({metric.score:.2f})\n"
            f"Question : {case['question']}\n"
            f"Réponse : {answer}\n"
            f"Raison du juge : {metric.reason}"
        )

    def test_sla_answer_is_relevant(self, chatbot, judge, eval_cases):
        """CASE002 — question SLA : la réponse doit traiter le sujet délais."""
        case = next(c for c in eval_cases if c["id"] == "CASE002")
        answer = chatbot.answer(case["question"], case["context"])

        test_case = LLMTestCase(input=case["question"], actual_output=answer)
        metric = AnswerRelevancyMetric(
            threshold=RELEVANCY_THRESHOLD, model=judge, async_mode=False
        )
        metric.measure(test_case)

        assert metric.score >= RELEVANCY_THRESHOLD, (
            f"Score : {metric.score:.2f} — Raison : {metric.reason}"
        )


@pytest.mark.judge
@pytest.mark.slow
class TestFaithfulness:
    """La réponse s'appuie-t-elle fidèlement sur le contexte fourni ?"""

    def test_sla_figures_are_faithful_to_context(self, chatbot, judge, eval_cases):
        """
        CASE002 — les chiffres du SLA (1h, 8h, 48h) doivent venir du
        contexte, pas être inventés ou déformés par le chatbot.
        """
        case = next(c for c in eval_cases if c["id"] == "CASE002")
        answer = chatbot.answer(case["question"], case["context"])

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            retrieval_context=case["context"],
        )
        metric = FaithfulnessMetric(
            threshold=FAITHFULNESS_THRESHOLD, model=judge, async_mode=False
        )
        metric.measure(test_case)

        assert metric.score >= FAITHFULNESS_THRESHOLD, (
            f"Fidélité insuffisante ({metric.score:.2f})\n"
            f"Réponse : {answer}\n"
            f"Contexte : {case['context']}\n"
            f"Raison du juge : {metric.reason}"
        )

    def test_password_reset_faithful_to_context(self, chatbot, judge, eval_cases):
        """CASE001 — la procédure décrite doit correspondre au contexte."""
        case = next(c for c in eval_cases if c["id"] == "CASE001")
        answer = chatbot.answer(case["question"], case["context"])

        test_case = LLMTestCase(
            input=case["question"],
            actual_output=answer,
            retrieval_context=case["context"],
        )
        metric = FaithfulnessMetric(
            threshold=FAITHFULNESS_THRESHOLD, model=judge, async_mode=False
        )
        metric.measure(test_case)

        assert metric.score >= FAITHFULNESS_THRESHOLD, (
            f"Score : {metric.score:.2f} — Raison : {metric.reason}"
        )
