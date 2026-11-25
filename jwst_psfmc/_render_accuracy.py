"""Assess fixed-shape response templates against an independent reference."""

import math
import warnings

import numpy as np

from .models import ConstantBackground, PointSource, SersicSource, _COMPONENT_FIELDS


def assess_rendering(fit, references, label, budget):
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("render_epsilon_max must be finite and positive")
    base = {"status": "unvalidated", "epsilon_max": float(budget)}
    components = [c for c in fit.model.components if type(c) is not ConstantBackground]
    if not components:
        return {**base, "status": "within_budget", "method": "constant_background_exact",
                "upper_epsilon": 0.0, "witness_epsilon": 0.0}
    if references is None:
        return {**base, "reason": "no independently validated reference for this fit"}
    if not isinstance(label, str) or not label.strip():
        raise ValueError("render references require a nonempty provenance label")
    if set(references) != {component.name for component in components}:
        raise ValueError("provide one unit-flux reference for each non-background component")
    base["reference"] = label
    roots = fit.parameters._roots
    values = fit.parameters.decode(fit.parameters.initial)
    errors, amplitudes = {}, {}
    for component in components:
        kind, name = type(component), component.name
        reference = np.asarray(references[name])
        if (reference.shape != fit.data.shape or reference.dtype.kind not in "iuf"
                or not np.all(np.isfinite(reference))):
            raise ValueError(f"{name}: render reference must be a finite real stamp")
        if kind not in (PointSource, SersicSource):
            return {**base, "reason": "reference assessment requires built-in source components"}
        if any(roots[f"{name}.{field}"].prior is not None
               for field in _COMPONENT_FIELDS[kind] if field != "flux"):
            return {**base, "reason": "unit-flux references do not cover free geometry"}
        key = f"{name}.flux"
        root = roots[key]
        unit = component.render({**values, key: 1.0}, fit.context)
        error = fit.noise.whiten(unit - reference)
        if root.name in errors:
            errors[root.name] += error
        else:
            errors[root.name] = error
        amplitudes[root.name] = root
    # Center a bounding ball on the midpoint error in whitened image space.
    initial = sum(errors[key] * values[key] for key in errors)
    witness = float(np.linalg.norm(initial))
    mid = np.zeros_like(initial)
    radius = 0.0
    for key, error in errors.items():
        root = amplitudes[key]
        lo, hi = root.prior._support if root.prior is not None else (root.fixed, root.fixed)
        if not (math.isfinite(lo) and math.isfinite(hi)):
            if np.any(error):
                radius = math.inf
            continue
        mid += (lo/2 + hi/2) * error
        radius += (hi/2 - lo/2) * np.linalg.norm(error)
        for endpoint in (lo, hi):
            witness = max(witness, float(np.linalg.norm(initial + (endpoint - values[key])*error)))
    upper = float(np.linalg.norm(mid) + radius)
    if witness > budget:
        status = "over_budget"
        warnings.warn(f"Render accuracy exceeds the image-error budget before fitting "
                      f"(epsilon={witness:.4g}, budget={budget:.4g}, reference={label})",
                      RuntimeWarning, stacklevel=3)
    else:
        status = "within_budget" if upper <= budget else "unvalidated"
    return {**base, "status": status, "method": "fixed_geometry_whitened_image_error",
            "upper_epsilon": upper if math.isfinite(upper) else None,
            "witness_epsilon": witness}
