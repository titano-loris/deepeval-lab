"""
Tests unitaires de la couche de robustesse JSON du juge local.

RAPIDES — aucun modèle chargé. On teste uniquement la logique
d'extraction et de coercition qui compense la fragilité d'un juge 3B.

Réflexe QA : la couche de robustesse écrite pour fiabiliser le juge
est elle-même du code à tester.

Lancement : pytest -m unit
"""
import pytest
from pydantic import BaseModel

from judge.local_judge import LocalJudge


# Schémas miroirs de ceux utilisés par DeepEval
class Verdict(BaseModel):
    verdict: str


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class Score(BaseModel):
    score: float
    reason: str


@pytest.mark.unit
class TestJsonExtraction:
    """L'extracteur doit survivre aux sorties sales d'un petit modèle."""

    def test_clean_json_is_parsed(self):
        assert LocalJudge._extract_json('{"score": 0.8}') == {"score": 0.8}

    def test_json_in_markdown_fences_is_parsed(self):
        raw = '```json\n{"score": 0.7, "reason": "ok"}\n```'
        assert LocalJudge._extract_json(raw) == {"score": 0.7, "reason": "ok"}

    def test_json_with_surrounding_prose_is_parsed(self):
        raw = 'Here is my evaluation: {"score": 0.9} I hope this helps!'
        assert LocalJudge._extract_json(raw) == {"score": 0.9}

    def test_nested_json_is_parsed(self):
        raw = '{"verdicts": [{"verdict": "yes", "reason": "relevant"}]}'
        result = LocalJudge._extract_json(raw)
        assert result["verdicts"][0]["verdict"] == "yes"

    def test_top_level_list_is_parsed(self):
        """v2 — cas à l'origine du crash du Run 1."""
        raw = '[{"verdict": "yes"}, {"verdict": "no"}]'
        assert LocalJudge._extract_json(raw) == [
            {"verdict": "yes"},
            {"verdict": "no"},
        ]

    def test_list_in_markdown_fences_is_parsed(self):
        raw = '```json\n[{"verdict": "idk"}]\n```'
        assert LocalJudge._extract_json(raw) == [{"verdict": "idk"}]

    def test_no_json_returns_none(self):
        assert LocalJudge._extract_json("I cannot evaluate this.") is None

    def test_malformed_json_returns_none(self):
        assert LocalJudge._extract_json('{"score": 0.8,,}') is None

    def test_empty_string_returns_none(self):
        assert LocalJudge._extract_json("") is None


@pytest.mark.unit
class TestSchemaCoercion:
    """v2 — La coercition liste -> objet corrige le crash du Run 1."""

    def test_dict_is_returned_unchanged(self):
        payload = {"verdicts": [{"verdict": "yes"}]}
        assert LocalJudge._coerce_to_schema(payload, Verdicts) == payload

    def test_bare_list_is_wrapped_into_single_field_schema(self):
        """Le juge renvoie [...] au lieu de {"verdicts": [...]}."""
        payload = [{"verdict": "yes"}, {"verdict": "no"}]
        coerced = LocalJudge._coerce_to_schema(payload, Verdicts)
        assert coerced == {"verdicts": payload}
        # La validation pydantic doit désormais passer
        assert len(Verdicts(**coerced).verdicts) == 2

    def test_bare_list_not_wrapped_when_schema_has_several_fields(self):
        """Schéma multi-champs : on ne devine pas, on laisse le retry jouer."""
        assert LocalJudge._coerce_to_schema([1, 2], Score) == {}