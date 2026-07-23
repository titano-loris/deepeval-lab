"""
LocalJudge — Un LLM local (Llama-3.2-3B) comme juge DeepEval.

DeepEval utilise par défaut
GPT-4 comme juge ou il nécessite une clé OpenAI payante. 
Ici, on implémente l'interface DeepEvalBaseLLM pour utiliser notre modèle
local à la place.

Défi technique : les métriques DeepEval attendent du JSON structuré
en sortie du juge. Un modèle 3B est moins fiable qu'un GPT-4 sur ce
point — on ajoute donc une couche de robustesse (extraction JSON,
retry) pour compenser.
"""
import json
import logging
import re

import torch
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

JUDGE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


class LocalJudge(DeepEvalBaseLLM):
    """
    Juge local pour les métriques DeepEval.

    Usage:
        judge = LocalJudge()
        metric = AnswerRelevancyMetric(model=judge, threshold=0.6)
    """

    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self._pipe = None  # chargement paresseux

    def load_model(self):
        if self._pipe is None:
            logger.info(f"Chargement du juge local : {self.model_name}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            self._pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer
            )
            logger.info("Juge local prêt.")
        return self._pipe

    def generate(self, prompt: str, schema: BaseModel = None) -> str:
        """
        Génère la réponse du juge. Si un schéma pydantic est fourni
        (cas des métriques DeepEval), on force et on valide le JSON.
        """
        pipe = self.load_model()

        system = (
            "You are a precise evaluation judge. "
            "Respond ONLY with valid JSON matching the requested format. "
            "No markdown, no explanation outside the JSON."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        output = pipe(
            messages,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        raw = output[0]["generated_text"][-1]["content"].strip()

        if schema is None:
            return raw

        # --- Couche de robustesse JSON pour petit modèle ---
        parsed = self._extract_json(raw)
        if parsed is not None:
            try:
                return schema(**parsed)
            except Exception as exc:
                logger.warning(f"JSON valide mais schéma non conforme : {exc}")

        # Retry unique avec consigne renforcée
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Your previous answer was not valid JSON for the schema "
                    f"{schema.model_json_schema()}. "
                    "Answer again with ONLY the JSON object."
                ),
            },
        ]
        output = pipe(
            retry_messages,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        raw_retry = output[0]["generated_text"][-1]["content"].strip()
        parsed = self._extract_json(raw_retry)
        if parsed is not None:
            return schema(**parsed)

        raise ValueError(
            f"Le juge local n'a pas produit de JSON valide.\n"
            f"Sortie brute : {raw_retry[:300]}"
        )

    async def a_generate(self, prompt: str, schema: BaseModel = None) -> str:
        """Version async — délègue à la version synchrone (CPU local)."""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"LocalJudge({self.model_name})"

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """
        Extrait le premier objet JSON d'un texte, même entouré de bruit
        (markdown fences, phrases parasites — fréquent avec les petits modèles).
        """
        # Retire les fences markdown éventuelles
        text = re.sub(r"```(?:json)?", "", text).strip()

        # Tentative directe
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Recherche du premier bloc { ... } équilibré
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None
