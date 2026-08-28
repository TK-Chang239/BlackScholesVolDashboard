"""Environment smoke test: deps importable, src package resolvable."""


def test_imports():
    import numpy
    import requests  # noqa: F401
    import scipy.optimize
    import scipy.stats
    import yaml  # noqa: F401

    import src.models  # noqa: F401

    assert numpy.asarray([1.0]).dtype == numpy.float64
    assert callable(scipy.stats.norm.cdf)
    assert callable(scipy.optimize.brentq)


def test_config_loads():
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg["symbol"] == "SPY"
    assert cfg["chain_filter"]["moneyness_min"] == 0.70
    assert cfg["rates"]["dividend_yield_fallback"] > 0
