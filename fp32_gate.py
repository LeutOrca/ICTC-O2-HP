"""Original numerical thresholds with exclusively FP32 floating error statistics."""


def gate(torch, got, ref, counts):
    results = {}
    for key in ("energy", "forces"):
        assert got[key].dtype == ref[key].dtype == torch.float32
        actual, expected = got[key].detach().cpu(), ref[key].detach().cpu()
        assert actual.shape == expected.shape
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        if key == "energy":
            atoms = counts.detach().cpu().to(torch.float32)
            actual, expected = actual.reshape(-1) / atoms, expected.reshape(-1) / atoms
        delta = (actual - expected).abs()
        reference_norm = expected.square().sum(dtype=torch.float32).sqrt()
        rms = expected.square().mean(dtype=torch.float32).sqrt()
        relative = delta.square().sum(dtype=torch.float32).sqrt() / reference_norm.clamp_min(1e-30)
        atol, rtol, relative_limit = (expected.new_tensor(value) for value in (2e-5, 2e-5, 1e-5))
        near_zero = rms < expected.new_tensor(1e-3) if key == "forces" else reference_norm < expected.new_tensor(1e-15)
        violations = int((delta > atol + rtol * expected.abs()).sum())
        results[key] = dict(violations=violations, rms_ref=float(rms), max_abs=float(delta.max()),
                            relative_l2=float(relative), relative_applies=not bool(near_zero),
                            unit="eV/atom" if key == "energy" else "eV/angstrom",
                            passed=violations == 0 and bool(near_zero | (relative <= relative_limit)),
                            metrics_dtype="torch.float32")
    return results
