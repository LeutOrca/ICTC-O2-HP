"""Apply isolated, frozen MACE changes while retaining unmodified native layers."""
VARIANTS = ("native", "base", "prefix", "scalar", "receiver", "projection", "receiver-projection", "final", "final-input")


def apply(model, variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    if variant in ("prefix", "final", "final-input"):
        from mace_frozen_prefix import install
        install(model)
    if variant == "scalar":
        from mace_scalar_tp import install
        install(model)
    if variant in ("projection", "receiver-projection", "final", "final-input"):
        from mace_species_projection import install
        install(model)
    if variant in ("receiver", "receiver-projection", "final", "final-input"):
        from mace_receiver_tp import install
        install(model)


def evaluator(variant, prior):
    if variant in ("receiver", "receiver-projection", "final", "final-input"):
        from mace_receiver_tp import evaluate
        return evaluate
    return prior.graph_evaluate
