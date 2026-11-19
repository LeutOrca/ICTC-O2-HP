"""Pack the live element weights for the residual selector as a separate step."""
from compact_species import PackedSpeciesProjection, install as install_first


def install(bench, profile=False):
    install_first(bench, profile=profile)
    base_model = bench.model_for
    evaluate = bench.evaluate

    def model_for(backend, device):
        if backend == "hp-uvu-compact-both":
            model = base_model("hp-uvu-compact", device)
            residual = model.interactions[1]
            residual.skip_tp = PackedSpeciesProjection(residual.skip_tp)
            return model
        return base_model(backend, device)

    def evaluate_both(model, data):
        selector = model.interactions[1].skip_tp
        if not isinstance(selector, PackedSpeciesProjection):
            return evaluate(model, data)
        selector.groups = data["_selector_groups"]
        try:
            return evaluate(model, data)
        finally:
            selector.groups = None

    bench.model_for = model_for
    bench.evaluate = evaluate_both
