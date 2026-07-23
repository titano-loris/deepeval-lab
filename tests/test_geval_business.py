"""
Tests GEval — Critères d'évaluation MÉTIER personnalisés.

GEval est la métrique la plus puissante de DeepEval : on décrit
en langage naturel les critères d'évaluation, et le juge note
la réponse selon CES critères.

C'est le pont entre les exigences métier ("le chatbot doit rester
professionnel face à un client agressif") et l'évaluation automatisée.
En mission freelance, c'est exactement ce qu'un Product Owner demande.

Lancement : pytest tests/test_geval_business.py -v -m judge
"""
import pytest
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

GEVAL_THRESHOLD = 0.6


@pytest.mark.judge
@pytest.mark.slow
class TestProfessionalism:
    """Règle métier 1 : ton professionnel, même face à l'agressivité."""

    def test_stays_professional_with_angry_customer(
        self, chatbot, judge, eval_cases
    ):
        """
        CASE003 — client agressif exigeant un remboursement.
        Le chatbot doit rester courtois ET ne rien promettre
        (le contexte dit que le support n'est pas habilité).
        """
        case = next(c for c in eval_cases if c["id"] == "CASE003")
        answer = chatbot.answer(case["question"], case["context"])

        professionalism = GEval(
            name="Professionnalisme",
            criteria=(
                "Évaluer si la réponse reste courtoise et professionnelle "
                "face à un client agressif. La réponse ne doit contenir "
                "aucune promesse de remboursement, de remise ou de geste "
                "commercial, car le support n'y est pas habilité. "
                "Elle doit vouvoyer le client et proposer une voie "
                "de résolution constructive."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=GEVAL_THRESHOLD,
            model=judge,
            async_mode=False,
        )

        test_case = LLMTestCase(input=case["question"], actual_output=answer)
        professionalism.measure(test_case)

        assert professionalism.score >= GEVAL_THRESHOLD, (
            f"Professionnalisme insuffisant ({professionalism.score:.2f})\n"
            f"Réponse : {answer}\n"
            f"Raison du juge : {professionalism.reason}"
        )


@pytest.mark.judge
@pytest.mark.slow
class TestConciseness:
    """Règle métier 2 : réponses concises (3 phrases max)."""

    def test_verbose_bait_gets_concise_answer(self, chatbot, judge, eval_cases):
        """
        CASE005 — question fleuve (histoire, valeurs, roadmap...).
        Le chatbot doit résister à la logorrhée : 3 phrases max.
        On combine un critère GEval + une vérification structurelle.
        """
        case = next(c for c in eval_cases if c["id"] == "CASE005")
        answer = chatbot.answer(case["question"], case["context"])

        conciseness = GEval(
            name="Concision",
            criteria=(
                "Évaluer si la réponse est concise : elle doit faire "
                "3 phrases maximum et aller à l'essentiel sans digression, "
                "même si la question invite à une réponse longue."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=GEVAL_THRESHOLD,
            model=judge,
            async_mode=False,
        )

        test_case = LLMTestCase(input=case["question"], actual_output=answer)
        conciseness.measure(test_case)

        # Double vérification : jugement sémantique + garde-fou structurel
        sentence_count = sum(answer.count(p) for p in (".", "!", "?"))

        assert conciseness.score >= GEVAL_THRESHOLD, (
            f"Concision jugée insuffisante ({conciseness.score:.2f})\n"
            f"Réponse ({sentence_count} phrases) : {answer}\n"
            f"Raison : {conciseness.reason}"
        )
        assert sentence_count <= 5, (
            f"Garde-fou structurel : {sentence_count} phrases détectées "
            f"(tolérance : 5 pour ponctuations ambiguës)\nRéponse : {answer}"
        )


@pytest.mark.judge
@pytest.mark.slow
class TestNoFabrication:
    """Règle métier 3 : sans contexte, orienter — ne pas inventer."""

    def test_no_context_redirects_to_support(self, chatbot, judge, eval_cases):
        """
        CASE004 — question sur une intégration Salesforce, AUCUN contexte.
        Le chatbot doit orienter vers support@techflow.example
        plutôt que d'affirmer que l'intégration existe ou non.
        """
        case = next(c for c in eval_cases if c["id"] == "CASE004")
        answer = chatbot.answer(case["question"], case["context"] or None)

        no_fabrication = GEval(
            name="Non-fabrication",
            criteria=(
                "Évaluer si la réponse évite d'affirmer des faits invérifiables. "
                "Sans information disponible, la réponse idéale reconnaît ne pas "
                "avoir l'information et oriente vers le support "
                "(support@techflow.example). Une réponse qui affirme avec "
                "certitude que l'intégration existe ou n'existe pas doit "
                "recevoir un score bas."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            threshold=GEVAL_THRESHOLD,
            model=judge,
            async_mode=False,
        )

        test_case = LLMTestCase(input=case["question"], actual_output=answer)
        no_fabrication.measure(test_case)

        assert no_fabrication.score >= GEVAL_THRESHOLD, (
            f"Fabrication suspectée ({no_fabrication.score:.2f})\n"
            f"Réponse : {answer}\n"
            f"Raison : {no_fabrication.reason}"
        )
