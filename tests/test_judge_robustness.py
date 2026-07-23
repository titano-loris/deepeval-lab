"""
Tests unitaires de la couche de robustesse JSON du juge local.

RAPIDES — aucun modèle chargé. On teste uniquement la logique
d'extraction JSON qui compense la fragilité d'un juge 3B.

C'est un réflexe QA important : la couche de robustesse qu'on a
écrite pour fiabiliser le juge est elle-même... du code à tester.

Lancement : pytest tests/test_judge_robustness.py -v -m unit
"""
import pytest

from judge.local_judge import LocalJudge


@pytest.mark.unit
class TestJsonExtraction:
    """L'extracteur JSON doit survivre aux sorties sales d'un petit modèle."""

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

    def test_no_json_returns_none(self):
        assert LocalJudge._extract_json("I cannot evaluate this.") is None

    def test_malformed_json_returns_none(self):
        assert LocalJudge._extract_json('{"score": 0.8,,}') is None

    def test_empty_string_returns_none(self):
        assert LocalJudge._extract_json("") is None
