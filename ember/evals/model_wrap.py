"""Model wrapper for generation-based evaluation."""
from __future__ import annotations

from typing import Any, Dict, List

import torch


class WrappedHFModel:
    """Unified HF generation wrapper for Gemma-2 and Llama-3-Instruct."""

    def __init__(self, model: Any, tokenizer: Any, it: bool = True) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.it = it
        # Decoder-only models must pad on the left for batched generation.
        self.tokenizer.padding_side = "left"

    # ------------------------------------------------------------------ #
    # Identity helpers                                                    #
    # ------------------------------------------------------------------ #

    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path

    def is_it(self) -> bool:
        return self.it

    def _is_llama(self) -> bool:
        return "llama" in self.tokenizer_name().lower()

    def _is_gemma(self) -> bool:
        return "gemma" in self.tokenizer_name().lower()

    def _device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ------------------------------------------------------------------ #
    # Chat-template wrapping                                              #
    # ------------------------------------------------------------------ #

    def wrap_prompt(self, prompt: str) -> str:
        """Apply the model's chat template to a raw user prompt.

        Gemma-2: strip the leading '<bos>' text (5 chars) and rely on
            add_special_tokens=True so the tokenizer prepends BOS as a
            token - single BOS.

        Llama-3: keep the template intact and tokenize with
            add_special_tokens=False to prevent a second BOS prepend.
        """
        if not self.it:
            return prompt
        s = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if self._is_gemma():
            return s[5:]   # strip '<bos>'; tokenizer prepends it as a token
        return s

    # ------------------------------------------------------------------ #
    # Tokenization                                                        #
    # ------------------------------------------------------------------ #

    def _tokenize(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a list of already-wrapped (or raw) prompts.

        Gemma: add_special_tokens=True - wrap_prompt strips the '<bos>'
            text so the tokenizer prepends BOS as token 2.
        Llama: add_special_tokens=False - the chat template already
            contains '<|begin_of_text|>' which encodes to BOS 128000.
        """
        return self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=self._is_gemma(),
        )

    # ------------------------------------------------------------------ #
    # Response extraction                                                 #
    # ------------------------------------------------------------------ #

    def _extract_response(self, decoded: str) -> str:
        """Strip the prompt header from a full decoded generation string.

        After decode(skip_special_tokens=True):
          Gemma-2: 'user\\n{question}\\nmodel\\n{response}'
          Llama-3: 'user\\n\\n{question}\\n\\nassistant\\n\\n{response}'
        """
        if not self.it:
            return decoded
        if self._is_gemma():
            idx = decoded.find("model\n")
            if idx != -1:
                return decoded[idx + len("model\n"):].strip()
            idx = decoded.find("model")
            return decoded[idx + len("model"):].strip() if idx != -1 else decoded
        if self._is_llama():
            idx = decoded.find("assistant\n\n")
            return decoded[idx + len("assistant\n\n"):].strip() if idx != -1 else decoded
        raise AssertionError(f"Unknown tokenizer: {self.tokenizer_name()}")

    # ------------------------------------------------------------------ #
    # Generation                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, prompt: Any, max_new_tokens: int = 50,
                 temperature: float = 0.1, do_sample: bool = False) -> str:
        """Single-item generation. Does NOT wrap the prompt internally.

        mc.py calls model.wrap_prompt(raw) then model.generate(wrapped).
        """
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        inputs = {k: v.to(self._device()) for k, v in self._tokenize([prompt]).items()}
        output_ids = self.model.generate(**inputs, **gen_kwargs)
        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return self._extract_response(decoded)

    def generate_multiple(self, prompts: List[str], max_new_tokens: int = 200,
                          do_sample: bool = False, batch_size: int = 50,
                          verbose: bool = False, **_kwargs) -> List[str]:
        """Batched generation. Wraps each prompt internally via wrap_prompt.

        Pass RAW (unwrapped) prompts.
        """
        wrapped = [self.wrap_prompt(p) for p in prompts]
        outputs: List[str] = []

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = 0.1

        iterator = range(0, len(wrapped), batch_size)
        if verbose:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Generating")
            except ImportError:
                pass

        for start in iterator:
            batch = wrapped[start:start + batch_size]
            inputs = {k: v.to(self._device())
                      for k, v in self._tokenize(batch).items()}
            gen_ids = self.model.generate(**inputs, **gen_kwargs)
            for seq_ids in gen_ids:
                decoded = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
                outputs.append(self._extract_response(decoded))

        return outputs

    def get_token_ids(self, prompt: str, wrap: bool = True) -> List[int]:
        """Return the token ID sequence for a prompt (diagnostic helper)."""
        text = self.wrap_prompt(prompt) if wrap else prompt
        return self._tokenize([text])["input_ids"][0].tolist()


def ensure_wrapped_model(model: Any, tokenizer: Any = None) -> WrappedHFModel:
    """Return a WrappedHFModel, or pass through if already wrapped."""
    if isinstance(model, WrappedHFModel):
        return model
    if tokenizer is not None:
        return WrappedHFModel(model, tokenizer, it=True)
    raise TypeError(
        f"Unsupported model type: {type(model).__name__}. "
        "Pass an HF AutoModelForCausalLM with a tokenizer."
    )


__all__ = ["WrappedHFModel", "ensure_wrapped_model"]
