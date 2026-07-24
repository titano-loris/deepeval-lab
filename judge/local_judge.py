"""
LocalJudge v2 — Un LLM local (Llama-3.2-3B) comme juge DeepEval.

Changements v1 -> v2 :
1. Le schéma JSON attendu est communiqué au juge DÈS le premier appel
   (v1 : uniquement au retry, le modèle devinait le format à l'aveugle).
2. Coercition automatique liste -> objet : quand le juge renvoie
   [{...}] alors que le schéma attend {"verdicts": [{...}]}, on emballe.
   C'est la cause du crash `TypeError: must be a mapping, not list`.
3. Extraction JSON tolérante aux listes de premier niveau.
4. Appel au pipeline factorisé dans _call(), max_new_tokens 512 -> 1024
   (les métriques multi-étapes génèrent des verdicts longs, tronqués à 512).
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
        metric = AnswerRelevancyMetric(model=judge, threshold=0.6, async_mode=False)
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
            logger.info("Juge local prêt (v2).")
        return self._pipe

    # ------------------------------------------------------------------
    # Appel bas niveau
    # ------------------------------------------------------------------
    def _call(self, messages: list[dict]) -> str:
        """Un appel au modèle, en génération déterministe."""
        pipe = self.load_model()
        output = pipe(
            messages,
            max_new_tokens=1024,
            temperature=0.0,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        return output[0]["generated_text"][-1]["content"].strip()

    # ------------------------------------------------------------------
    # Interface DeepEvalBaseLLM
    # ------------------------------------------------------------------
    def generate(self, prompt: str, schema: BaseModel = None):
        system = (
            "You are a precise evaluation judge. "
            "Respond ONLY with valid JSON. No markdown fences, no prose."
        )

        # v2 — le format attendu est annoncé dès le premier appel
        if schema is not None:
            system += (
                "\nYour answer MUST be a single JSON OBJECT (not a list) "
                "matching exactly this JSON schema:\n"
                + json.dumps(schema.model_json_schema())
            )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        raw = self._call(messages)

        if schema is None:
            return raw

        result = self._parse_against_schema(raw, schema)
        if result is not None:
            return result

        # Retry unique, consigne renforcée
        retry_messages = messages + [
            {"role": "assistant", "content": raw},
            {
                "role": "user",
                "content": (
                    "Invalid. Output ONLY the JSON object, starting with '{' "
                    "and matching exactly the schema given above."
                ),
            },
        ]
        raw_retry = self._call(retry_messages)
        result = self._parse_against_schema(raw_retry, schema)
        if result is not None:
            return result

        raise ValueError(
            f"Juge local : JSON invalide après retry.\n"
            f"Sortie brute : {raw_retry[:300]}"
        )

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        """Version async — délègue au synchrone (inférence CPU locale)."""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"LocalJudge-v2({self.model_name})"

    # ------------------------------------------------------------------
    # Couche de robustesse
    # ------------------------------------------------------------------
    def _parse_against_schema(self, raw: str, schema: BaseModel):
        """Extrait, coerce, puis valide contre le schéma pydantic."""
        parsed = self._extract_json(raw)
        if parsed is None:
            return None
        parsed = self._coerce_to_schema(parsed, schema)
        try:
            return schema(**parsed)
        except Exception as exc:
            logger.warning(f"Schéma non conforme : {exc}")
            return None

    @staticmethod
    def _coerce_to_schema(parsed, schema: BaseModel) -> dict:
        """
        v2 — Coercition liste -> objet.

        Les petits modèles renvoient souvent [{...}, {...}] là où DeepEval
        attend {"verdicts": [{...}, {...}]}. Si le schéma n'a qu'un seul
        champ, on emballe automatiquement la valeur dedans.
        """
        if isinstance(parsed, dict):
            return parsed
        fields = list(schema.model_fields.keys())
        if len(fields) == 1:
            return {fields[0]: parsed}
        return {}  # échouera à la validation -> déclenche le retry

    @staticmethod
    def _extract_json(text: str):
        """
        Extrait le premier JSON d'un texte bruité.
        v2 : accepte aussi une liste de premier niveau.
        """
        text = re.sub(r"```(?:json)?", "", text).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for opener, closer in (("{", "}"), ("[", "]")):
            start = text.find(opener)
            if start == -1:
                continue
            depth = 0
            for i, ch in enumerate(text[start:], start):
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])
                        except json.JSONDecodeError:
                            break
        return None