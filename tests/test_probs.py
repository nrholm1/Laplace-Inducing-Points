import pytest

from src.train_map import nl_posterior_fun, nl_likelihood_fun, nl_prior_fun


# todo 1. Simple synthetic distributions
    # todo 1a: small synthetic model (known params)
    # todo 1b: small synthetic dataset from known model
    # todo 1c: compute log likelihood with nl_likelihood_fun
    # todo 1d: compare to manually computed log likelihood

# todo 2. Toy posterior setup
    # todo 2a: compute log prior manually
    # todo 2b: combine with log likelihood
    # todo 2c: check nl_posterior_fun = nl_likelihood + nl_prior


# todo 3. Verify learned variance term!
    # todo 3a: how??

@pytest.fixture
def test_nl_posterior_fun():
    ...


@pytest.fixture
def test_nl_likelihood_fun():
    ...