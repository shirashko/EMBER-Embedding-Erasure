"""Map a frozen dictionary's activation site onto this repo's hook names.

The dictionaries under ``mlp_intermediate`` were trained on the input to the
MLP down-projection. This repo calls that same tensor ``mlp`` and reads it at
``blocks.{layer}.mlp.hook_post``. ``mlp_output`` is a different tensor (the
MLP block output) and must not be scored with that hook.
"""
from __future__ import annotations

# Dictionary site name -> EMBER ActivationGenerator mode.
DICTIONARY_SITE_TO_EMBER_MODE = {
    "mlp_intermediate": "mlp",
    "mlp": "mlp",
}

EMBER_MODE_HOOK = {
    "mlp": "blocks.{layer}.mlp.hook_post",
    "mlp_out": "blocks.{layer}.hook_mlp_out",
    "residual": "blocks.{layer}.hook_resid_post",
}


def ember_mode_for_site(site: str) -> str:
    """Return the EMBER mode for a dictionary site, or exit if it is not MLP-intermediate."""
    mode = DICTIONARY_SITE_TO_EMBER_MODE.get(site)
    if mode is None:
        known = ", ".join(sorted(DICTIONARY_SITE_TO_EMBER_MODE))
        raise SystemExit(
            f"Dictionary site {site!r} is not MLP-intermediate. "
            f"Only {known} map to EMBER mode 'mlp' ({EMBER_MODE_HOOK['mlp']}). "
            "mlp_output and residual are different tensors and are not scored here."
        )
    return mode


def hook_for_layer(ember_mode: str, layer: int) -> str:
    try:
        template = EMBER_MODE_HOOK[ember_mode]
    except KeyError as exc:
        raise SystemExit(f"No hook registered for EMBER mode {ember_mode!r}") from exc
    return template.format(layer=layer)
