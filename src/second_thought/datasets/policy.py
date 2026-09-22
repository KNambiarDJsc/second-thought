"""Hard boundary: which providers' decisions may ever leave this project as training data.

TypeSafe's Master Customer Agreement (§2.3(b), https://typesafe.ai/legal/mca)
prohibits using Jev output "to perform model distillation, train a model to
imitate the output of the Services, or develop ... a similar or competing
product." A capture → correct → dataset → fine-tune loop that included
Jev-sourced decisions in its dataset export would do exactly that.

This module is the single place that boundary is enforced. It is
deliberately not a config flag: a provider is blocked here, in code, not in
a settings file a caller could edit or forget to set. The capture and
storage layers accept Jev decisions freely (observability/calibration
review of your own Jev usage is not restricted) — only ``datasets.export``
consults this module, and it does so unconditionally.
"""

from __future__ import annotations

TRAINING_EXPORT_BLOCKED_PROVIDERS: frozenset[str] = frozenset({"jev"})


class BlockedProviderError(RuntimeError):
    """Raised when a caller explicitly asks to export a policy-blocked provider's decisions."""


def is_exportable(provider: str) -> bool:
    return provider.strip().lower() not in TRAINING_EXPORT_BLOCKED_PROVIDERS


def assert_not_explicitly_blocked(providers: set[str]) -> None:
    """Fail loudly if a caller explicitly asked for a blocked provider by name.

    Called with the caller's requested provider filter (if any), not with
    every provider present in storage — silently omitting Jev records from a
    mixed-provider export is the normal, quiet path; explicitly asking for
    ``providers=["jev"]`` on an export is a mistake worth stopping for.
    """
    blocked = {p for p in providers if not is_exportable(p)}
    if blocked:
        raise BlockedProviderError(
            f"provider(s) {sorted(blocked)!r} may not be included in a dataset export. "
            "TypeSafe's MCA (§2.3(b)) prohibits training or distilling a model on Jev "
            "output, or using it to build a similar/competing product. Jev decisions "
            "can be captured and reviewed for your own observability, but this "
            "project will not include them in any training/eval dataset export."
        )
