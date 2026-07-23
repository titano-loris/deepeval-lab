"""
Chatbot cible — Le système sous test.

Un assistant support client simple, avec un system prompt qui définit
son comportement attendu : ton professionnel, réponses concises,
pas de promesses non autorisées.

C'est CE comportement que DeepEval va évaluer via le juge local :
- Les réponses sont-elles pertinentes ? (AnswerRelevancy)
- Sont-elles fidèles au contexte fourni ? (Faithfulness)
- Respectent-elles nos critères métier ? (GEval : professionnalisme, concision)
"""
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

TARGET_MODEL = "meta-llama/Llama-3.2-3B-Instruct"

SYSTEM_PROMPT = """Tu es l'assistant support client de la société TechFlow,
un éditeur de logiciels SaaS de gestion de projet.

Règles de comportement :
1. Ton professionnel et courtois, tutoiement interdit.
2. Réponses CONCISES : 3 phrases maximum.
3. Si un contexte documentaire est fourni, s'appuyer uniquement dessus.
4. Ne JAMAIS promettre de remboursement, de remise ou de délai
   sans que le contexte l'autorise explicitement.
5. Si l'information manque, orienter vers support@techflow.example."""


class SupportChatbot:
    """Assistant support — le système dont on évalue la qualité."""

    def __init__(self, model_name: str = TARGET_MODEL):
        logger.info(f"Chargement du chatbot cible : {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        logger.info("Chatbot cible prêt.")

    def answer(self, question: str, context: list[str] | None = None) -> str:
        """Répond à une question client, avec contexte documentaire optionnel."""
        user_content = question
        if context:
            ctx = "\n\n".join(context)
            user_content = f"CONTEXTE :\n{ctx}\n\nQUESTION CLIENT : {question}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        output = self.pipe(
            messages,
            max_new_tokens=200,
            temperature=0.1,
            do_sample=True,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        return output[0]["generated_text"][-1]["content"].strip()
