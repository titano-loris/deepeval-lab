# ⚖️ DeepEval-Lab

**LLM-as-Judge en local : évaluer un chatbot avec DeepEval et un juge Llama-3.2-3B — sans API payante**

> Projet de [RAG-TestKit](https://github.com/titano-loris/rag-testkit).
> RAG-TestKit teste avec des assertions **déterministes** ; DeepEval-Lab ajoute
> la dimension **LLM-as-Judge** : un modèle local note la qualité des réponses
> d'un autre modèle selon des métriques natives et des critères métier personnalisés.

## 🎯 Le problème

Un modèle comme DeepEval utilise **GPT-4 par défaut** comme juge d'évaluation. Cela implique :
une clé API payante, un coût par test, et l'envoi des données à un tiers —
souvent rédhibitoire pour des clients européens (RGPD) ou des environnements air-gapped.

**La solution explorée ici** : implémenter l'interface `DeepEvalBaseLLM` pour
utiliser un **Llama-3.2-3B local** comme juge. pour une èvaluation gratuite.

## 🏗️ Architecture

```
┌──────────────────┐           ┌───────────────────┐
│  SYSTÈME TESTÉ   │           │   JUGE LOCAL      │
│  Chatbot support │ réponses  │   Llama-3.2-3B    │
│  (Llama-3.2-3B)  ├─────────→ │ (DeepEvalBaseLLM) │
│                  │           │  + robustesse JSON│
└──────────────────┘           └─────────┬─────────┘
         ↑                               ↓
    5 cas de test                Scores 0-1 + raisons
    (eval_cases.json)            (rapport pytest-html)
```

## 📏 Métriques implémentées

| Métrique                  | Type                      | Question évaluée                                    |
| ------------------------- | ------------------------- | --------------------------------------------------- |
| `AnswerRelevancyMetric`   | Native DeepEval           | La réponse traite-t-elle la question ?              |
| `FaithfulnessMetric`      | Native DeepEval           | La réponse est-elle fidèle au contexte ?            |
| `GEval` Professionnalisme | **Critère métier custom** | adaptation à un client, zéro promesse non autorisée |
| `GEval` Concision         | **Critère métier custom** | 3 phrases max, même sur question fleuve             |
| `GEval` Non-fabrication   | **Critère métier custom** | Sans contexte → orienter, ne pas inventer           |

## 🔧 Le défi technique : fiabiliser un juge 3B

Les métriques DeepEval attendent du **JSON structuré** en sortie du juge.
GPT-4 y excelle ; un modèle 3B est plus fragile. Le projet inclut une
couche de robustesse dans `judge/local_judge.py` :

- extraction JSON tolérante (fences markdown, prose parasite, JSON imbriqué)
- retry unique avec consigne renforcée en cas d'échec de parsing
- validation pydantic contre le schéma attendu

Cette couche est **elle-même testée** (7 tests unitaires, sans modèle) —
tester son outillage de test est un réflexe QA fondamental.

## 🚀 Installation

```bash
git clone https://github.com/titano-loris/deepeval-lab.git
cd deepeval-lab
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # puis renseigner HF_TOKEN
```

Prérequis : licence [Llama-3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
acceptée sur HuggingFace, ~16GB RAM disponibles.

## 💻 Usage

```bash
# Tests rapides — robustesse JSON du juge (~5 s, sans modèle)
pytest -m unit

# Évaluation complète LLM-as-Judge (~20-40 min sur CPU)
pytest -m judge --html=reports/judge-report.html --self-contained-html
```

## 📊 Ce que ce projet démontre

1. **Traduire des exigences métier en critères automatisés** — les règles du
   chatbot (ton, concision, non-fabrication) deviennent des métriques GEval.
2. **Maîtriser les deux philosophies du testing IA** — déterministe
   (RAG-TestKit) et LLM-as-Judge (ce projet), avec leurs forces et limites.
3. **L'ingénierie de fiabilisation** — rendre un petit modèle utilisable
   comme juge là où la documentation suppose GPT-4.

## ⚠️ Limites connues (transparence)

- Un juge 3B est **moins fiable** qu'un GPT-4 : les scores peuvent varier
  entre deux exécutions malgré `temperature=0`.
- La latence CPU (~1-3 min par métrique) réserve ce setup à l'expérimentation
  et aux petites suites — en production, prévoir un GPU ou un juge API.
- Les métriques multi-étapes de DeepEval (qui décomposent en sous-questions)
  multiplient les appels au juge : chaque test = plusieurs inférences.

## 👤 Auteur

**Loris Bartolini** — QA Automation Engineer | AI & LLM Testing
[LinkedIn](https://linkedin.com/in/loris-bartolini) · [GitHub](https://github.com/titano-loris)
