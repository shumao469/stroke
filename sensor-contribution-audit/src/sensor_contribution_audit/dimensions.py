"""Bookkeeping of declared model dimensions, not evidence of model efficacy."""


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def dimension_audit(d_x, d_z, h_x=None, h_z=None, quality=0, availability=0):
    """Audit an affine score and, optionally, a declared encoder/readout input.

    s = w_x.T x + w_z.T z + b has d_x + d_z + 1 scalar parameters.
    A nonlinear encoder/fusion parameter count requires an architecture.
    Quality and availability are observed inputs, not model parameters.
    """
    d_x = _integer(d_x, "d_x")
    d_z = _integer(d_z, "d_z")
    quality = _integer(quality, "quality", 0)
    availability = _integer(availability, "availability", 0)
    result = {
        "existing_feature_dimension": d_x,
        "added_feature_dimension": d_z,
        "combined_feature_dimension": d_x + d_z,
        "coefficient_dimensions": {"w_x": d_x, "w_z": d_z},
        "affine_intercept_parameter_count": 1,
        "affine_total_parameter_count": d_x + d_z + 1,
        "score_dimension": 1,
        "scope": "Declared feature dimensions and an affine scalar score only",
    }
    if (h_x is None) != (h_z is None):
        raise ValueError("provide both latent dimensions together")
    if h_x is not None:
        h_x = _integer(h_x, "h_x")
        h_z = _integer(h_z, "h_z")
        result["encoder_fusion_input"] = {
            "existing_latent_dimension": h_x,
            "added_latent_dimension": h_z,
            "observed_quality_dimension": quality,
            "observed_availability_dimension": availability,
            "declared_concatenated_readout_input_dimension": h_x + h_z + quality + availability,
            "encoder_and_fusion_parameter_count": None,
            "parameter_count_scope": "Architecture-dependent; inferred neither from feature dimension nor from input count",
        }
    elif quality or availability:
        raise ValueError("latent dimensions are required to audit additional readout inputs")
    return result
