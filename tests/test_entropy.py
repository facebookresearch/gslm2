"""Tests for entropy computation."""

import math

import pytest
import torch

from gslm2.models.speech.metrics import compute_entropy


class TestComputeEntropy:
    """Test compute_entropy function."""

    def test_uniform_distribution(self) -> None:
        """Uniform distribution should have maximum entropy = log(K)."""
        k = 64
        batch_size = 100

        probs = torch.ones(batch_size, k) / k
        log_probs = torch.log(probs)

        predictive, codebook = compute_entropy(probs, log_probs)

        expected_entropy = math.log(k)
        assert predictive.item() == pytest.approx(expected_entropy, rel=1e-5)
        assert codebook.item() == pytest.approx(expected_entropy, rel=1e-5)

    def test_deterministic_one_hot(self) -> None:
        """One-hot (deterministic) distribution should have zero entropy."""
        k = 64
        batch_size = 10

        # All waveforms choose class 0
        probs = torch.zeros(batch_size, k)
        probs[:, 0] = 1.0
        log_probs = torch.log(probs + 1e-8)

        predictive, codebook = compute_entropy(probs, log_probs)

        assert predictive.item() == pytest.approx(0.0, abs=1e-5)
        assert codebook.item() == pytest.approx(0.0, abs=1e-5)

    def test_mixed_one_hot(self) -> None:
        """Different one-hot samples: zero predictive entropy, nonzero codebook entropy."""
        k = 32

        # Each sample picks a different class
        probs = torch.eye(k)  # Implicitly batch_size = 32
        log_probs = torch.log(probs + 1e-8)

        predictive, codebook = compute_entropy(probs, log_probs)

        # Predictive: each sample has 0 entropy → mean is 0
        assert predictive.item() == pytest.approx(0.0, abs=1e-5)
        # Codebook: avg distribution is uniform → entropy is log(K)
        assert codebook.item() == pytest.approx(math.log(k), rel=1e-5)
