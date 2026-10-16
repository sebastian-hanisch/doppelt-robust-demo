"""Generator der Linie (wortgleich in allen Repos): Strom, Normalverteilung, Strukturmodell, Intervention, geschlossene Wirkung."""

import math

import numpy as np
import pytest

import kausal_generator as gen

MASK = (1 << 64) - 1


def splitmix_reference(seed, count):
    """Unabhängige Python-Fassung des SplitMix64-Stroms (Steele, Lea, Flood 2014)."""
    state, out = seed & MASK, []
    for _ in range(count):
        state = (state + 0x9E3779B97F4A7C15) & MASK
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        out.append(z ^ (z >> 31))
    return out


@pytest.mark.parametrize("seed", [0, 1, 35, 123456789, 2**63 + 5])
def test_splitmix_stream_matches_the_reference(seed):
    assert gen.SplitMix64(seed).u64(25).tolist() == splitmix_reference(seed, 25)


def test_the_stream_does_not_depend_on_how_it_is_drawn():
    a, b = gen.SplitMix64(7), gen.SplitMix64(7)
    assert np.concatenate([a.uniform(3), a.uniform(4)]).tolist() == b.uniform(7).tolist()


def test_uniform_is_in_the_half_open_unit_interval():
    u = gen.SplitMix64(3).uniform(50000)
    assert u.min() >= 0.0 and u.max() < 1.0 and abs(u.mean() - 0.5) < 0.01


def test_normal_has_mean_zero_and_unit_variance():
    z = gen.SplitMix64(11).normal(200000)
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.02


def test_normal_cdf_known_values_and_scalar_input():
    assert float(gen.normal_cdf(0.0)) == pytest.approx(0.5)
    assert float(gen.normal_cdf(1.959964)) == pytest.approx(0.975, abs=1e-6)
    assert float(gen.normal_cdf(-1.0)) == pytest.approx(0.15865525393145707, abs=1e-12)
    assert gen.normal_cdf(np.array([0.0, 1.0])).shape == (2,)


def test_normal_pdf_at_zero():
    assert float(gen.normal_pdf(0.0)) == pytest.approx(1.0 / math.sqrt(2.0 * math.pi))


@pytest.mark.parametrize("p", [1e-9, 1e-6, 0.001, 0.02, 0.0243, 0.1, 0.5, 0.9, 0.9757, 0.999, 1 - 1e-9])
def test_ppf_inverts_the_cdf(p):
    x = gen.normal_ppf(np.array([p]))
    assert float(gen.normal_cdf(x)[0]) == pytest.approx(p, rel=1e-6, abs=1e-12)


def test_ppf_is_odd_and_zero_at_half():
    assert float(gen.normal_ppf(np.array([0.5]))[0]) == pytest.approx(0.0, abs=1e-12)
    p = np.array([0.01, 0.2, 0.7])
    assert gen.normal_ppf(1.0 - p).tolist() == pytest.approx((-gen.normal_ppf(p)).tolist(), abs=1e-9)


@pytest.mark.parametrize("kind", gen.NOISE_KINDS)
def test_every_noise_kind_has_mean_zero_and_unit_variance(kind):
    z = gen._noise_block(kind, gen.SplitMix64(5), 300000)
    assert abs(z.mean()) < 0.01 and abs(z.var() - 1.0) < 0.02


def test_noise_kinds_differ_in_tails():
    kurt = {k: float(np.mean(gen._noise_block(k, gen.SplitMix64(5), 300000) ** 4)) for k in gen.NOISE_KINDS}
    assert kurt["uniform"] < kurt["gauss"] < kurt["laplace"]


def chain(c_xm=0.5, c_my=2.0, c_xy=0.0):
    return gen.Scm(names=("X", "M", "Y"), parents=((), (0,), (0, 1)), coefs=((), (c_xm,), (c_xy, c_my)),
                   intercepts=(0.0, 0.0, 0.0), noise_sd=(1.0, 0.5, 0.5), binary=(False, False, False))


def test_scm_rejects_inconsistent_inputs():
    with pytest.raises(ValueError):
        gen.Scm(names=("A", "B"), parents=((), (1,)), coefs=((), (1.0,)), intercepts=(0.0, 0.0), noise_sd=(1.0, 1.0), binary=(False, False))
    with pytest.raises(ValueError):
        gen.Scm(names=("A",), parents=((),), coefs=((),), intercepts=(0.0,), noise_sd=(1.0,), binary=(False,), noise_kind="cauchy")
    with pytest.raises(ValueError):
        gen.Scm(names=("A", "B"), parents=((), (0,)), coefs=((), ()), intercepts=(0.0, 0.0), noise_sd=(1.0, 1.0), binary=(False, False))


def test_sample_shapes_and_unknown_do_node():
    d = gen.sample(chain(), 50, gen.SplitMix64(1), reps=3)
    assert set(d) == {"X", "M", "Y"} and all(v.shape == (3, 50) for v in d.values())
    with pytest.raises(ValueError):
        gen.sample(chain(), 5, gen.SplitMix64(1), do={"Q": 1.0})


def test_same_seed_gives_the_same_sample_and_another_seed_a_different_one():
    a = gen.sample(chain(), 20, gen.SplitMix64(4))["Y"]
    assert a.tolist() == gen.sample(chain(), 20, gen.SplitMix64(4))["Y"].tolist()
    assert a.tolist() != gen.sample(chain(), 20, gen.SplitMix64(5))["Y"].tolist()


def test_intervention_leaves_upstream_nodes_alone_and_shifts_downstream_by_the_path_sum():
    scm = chain(0.5, 2.0, 0.3)
    base = gen.sample(scm, 200, gen.SplitMix64(9))
    one = gen.sample(scm, 200, gen.SplitMix64(9), do={"X": 1.0})
    zero = gen.sample(scm, 200, gen.SplitMix64(9), do={"X": 0.0})
    assert (one["X"] == 1.0).all() and (zero["X"] == 0.0).all()
    assert np.allclose(one["M"] - zero["M"], 0.5)
    assert np.allclose(one["Y"] - zero["Y"], 0.3 + 0.5 * 2.0)
    assert base["X"].shape == (1, 200)


def test_total_effect_of_a_chain_a_direct_edge_and_a_diamond_by_hand():
    assert gen.total_effect(chain(0.5, 2.0, 0.0), "X", "Y") == pytest.approx(1.0)
    assert gen.total_effect(chain(0.5, 2.0, 0.3), "X", "Y") == pytest.approx(1.3)
    assert gen.total_effect(chain(0.5, 2.0, 0.3), "M", "Y") == pytest.approx(2.0)
    assert gen.total_effect(chain(0.5, 2.0, 0.3), "Y", "X") == pytest.approx(0.0)
    diamond = gen.Scm(names=("A", "B", "C", "D"), parents=((), (0,), (0,), (1, 2)), coefs=((), (2.0,), (3.0,), (0.5, -1.0)),
                      intercepts=(0.0,) * 4, noise_sd=(1.0,) * 4, binary=(False,) * 4)
    assert gen.total_effect(diamond, "A", "D") == pytest.approx(2.0 * 0.5 + 3.0 * -1.0)


def test_total_effect_matches_the_simulated_intervention_in_the_standorte_world():
    scm = gen.standorte_confounded(1.0, 0.0, 4.0, 3.5, 5.0)
    one = gen.sample(scm, 500, gen.SplitMix64(2), do={"Faktor": 1.0})["Servicegrad"]
    zero = gen.sample(scm, 500, gen.SplitMix64(2), do={"Faktor": 0.0})["Servicegrad"]
    assert gen.total_effect(scm, "Faktor", "Servicegrad") == pytest.approx(3.5)
    assert (one - zero).mean() == pytest.approx(3.5, abs=1e-9)


def test_total_effect_of_a_node_on_itself_is_zero_and_scm_rejects_negative_parent_index():
    assert gen.total_effect(chain(0.5, 2.0, 0.3), "X", "X") == pytest.approx(0.0)
    with pytest.raises(ValueError):
        gen.Scm(names=("A", "B"), parents=((), (-1,)), coefs=((), (1.0,)), intercepts=(0.0, 0.0), noise_sd=(1.0, 1.0), binary=(False, False))


def test_total_effect_refuses_a_binary_node_on_the_path():
    scm = gen.Scm(names=("A", "B", "C"), parents=((), (0,), (1,)), coefs=((), (1.0,), (1.0,)), intercepts=(0.0,) * 3, noise_sd=(1.0,) * 3,
                  binary=(False, True, False))
    with pytest.raises(ValueError):
        gen.total_effect(scm, "A", "C")
    assert gen.total_effect(scm, "B", "C") == pytest.approx(1.0)


def test_binary_node_is_a_probit_threshold():
    scm = gen.Scm(names=("Z", "X"), parents=((), (0,)), coefs=((), (1.0,)), intercepts=(0.0, 0.5), noise_sd=(1.0, 1.0), binary=(False, True))
    d = gen.sample(scm, 200000, gen.SplitMix64(8))
    assert set(np.unique(d["X"])) == {0.0, 1.0}
    assert d["X"].mean() == pytest.approx(float(gen.normal_cdf(0.5 / math.sqrt(2.0))), abs=0.005)
    near_zero = np.abs(d["Z"][0]) < 0.05
    assert d["X"][0][near_zero].mean() == pytest.approx(float(gen.normal_cdf(0.5)), abs=0.03)


def test_coin_version_makes_the_factor_independent_of_the_fluctuation_with_the_same_share():
    obs = gen.standorte_confounded(1.5, 0.0, 4.0, 3.5, 5.0)
    coin = gen.standorte_confounded(1.5, 0.0, 4.0, 3.5, 5.0, coin_p=0.5)
    d = gen.sample(coin, 100000, gen.SplitMix64(3))
    assert abs(np.corrcoef(d["Schwankung"][0], d["Faktor"][0])[0, 1]) < 0.02
    assert d["Faktor"].mean() == pytest.approx(gen.sample(obs, 100000, gen.SplitMix64(3))["Faktor"].mean(), abs=0.01)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            gen.standorte_confounded(1.0, 0.0, 4.0, 3.5, 5.0, coin_p=bad)


def test_generator_has_no_module_prefix_and_a_version():
    assert gen.GENERATOR_VERSION == 1
