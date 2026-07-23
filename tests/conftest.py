"""
Fixtures partagées — DeepEval-Lab.

Architecture mémoire importante : le chatbot cible ET le juge utilisent
le même modèle de base (Llama-3.2-3B). Sur une machine 32GB, on peut
charger deux instances, mais c'est du gaspillage — les fixtures en
scope session garantissent un seul chargement de chaque rôle pour
toute la suite de tests.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_PATH = Path(__file__).parent.parent / "datasets" / "eval_cases.json"


@pytest.fixture
def eval_cases() -> list[dict]:
    """Les 5 cas d'évaluation du chatbot support."""
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def chatbot():
    """Le système sous test — chargé une fois pour toute la session."""
    from target.chatbot import SupportChatbot

    return SupportChatbot()


@pytest.fixture(scope="session")
def judge():
    """Le juge local DeepEval — chargé une fois pour toute la session."""
    from judge.local_judge import LocalJudge

    return LocalJudge()
