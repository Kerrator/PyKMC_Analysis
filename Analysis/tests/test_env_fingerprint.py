"""Tests for the lattice-environment fingerprint module.

Verifies D4 group closure, permutation correctness, canonicalisation
invariance under rotation/mirror, and determinism of fingerprint strings.
"""
from __future__ import annotations

import numpy as np
import pytest

from Analysis.env_fingerprint import (
    D4_OPS,
    INV_PERMS_BULK,
    INV_PERMS_SURFACE,
    PERMS_BULK,
    PERMS_SURFACE,
    SHELL_REF_POSITIONS_BULK,
    SHELL_REF_POSITIONS_SURFACE,
    canonicalise,
    fingerprint_string,
)


# ---------------------------------------------------------------------------
# Test 6.1 — D4 group closure
# ---------------------------------------------------------------------------

def test_d4_has_eight_elements():
    assert len(D4_OPS) == 8


def test_d4_is_closed_under_composition():
    """For every pair (R1, R2), R1 @ R2 must also be in D4_OPS."""
    op_set = {tuple(R.flatten().astype(int).tolist()) for R in D4_OPS}
    for i, R1 in enumerate(D4_OPS):
        for j, R2 in enumerate(D4_OPS):
            R12 = R1 @ R2
            key = tuple(R12.flatten().astype(int).tolist())
            assert key in op_set, f"D4_OPS[{i}] @ D4_OPS[{j}] not in group"


def test_d4_inverse_exists_in_group():
    """Every R has an inverse R^-1 also in D4_OPS."""
    op_set = {tuple(R.flatten().astype(int).tolist()) for R in D4_OPS}
    for i, R in enumerate(D4_OPS):
        inv = np.linalg.inv(R).astype(int)
        key = tuple(inv.flatten().tolist())
        assert key in op_set, f"D4_OPS[{i}]^-1 not in group"


# ---------------------------------------------------------------------------
# Test 6.2 — Permutations are valid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("perms,n_slots", [
    (PERMS_SURFACE, 13),
    (PERMS_BULK, 18),
])
def test_perms_are_valid_permutations(perms, n_slots):
    assert perms.shape == (8, n_slots)
    for op_idx in range(8):
        assert sorted(perms[op_idx].tolist()) == list(range(n_slots))


# ---------------------------------------------------------------------------
# Test 6.3 — Canonicalisation: rotated environments give SAME fingerprint
# ---------------------------------------------------------------------------

def test_rotation_invariance_simple():
    """A vacancy at slot 0 with hop into the vacancy should canonicalise
    identically under rotation by 90, 180, or 270 degrees.

    Active rotation of env by R: rotated_env[R(p)] = env[p], i.e.,
    rotated_env = env[INV_PERMS[R]]. Hop rotation: R @ hop. Both must
    use the SAME R to give a physically equivalent environment.
    """
    n = len(SHELL_REF_POSITIONS_SURFACE)
    # Original: vacant at slot 0 = (2, 0, 0); hop = (2, 0, 0) (into vacancy)
    env0 = np.ones(n, dtype=np.int8)
    env0[0] = 0
    canon0 = canonicalise(env0, (2, 0, 0), PERMS_SURFACE)

    # Rotate (env, hop) consistently — same R applied to both
    for op_idx in range(1, 8):
        env_rot = env0[INV_PERMS_SURFACE[op_idx]]
        hop_rot = tuple((D4_OPS[op_idx] @ np.array([2, 0, 0])).tolist())
        canon_r = canonicalise(env_rot, hop_rot, PERMS_SURFACE)
        assert canon_r == canon0, (
            f"rotation by op {op_idx} broke canonicalisation: "
            f"orig {canon0} vs rotated {canon_r}"
        )


def test_distinct_environments_give_distinct_fingerprints():
    """Two environments with vacancies in geometrically inequivalent
    positions (relative to the hop direction) must yield different
    fingerprints, even after canonicalisation."""
    n = len(SHELL_REF_POSITIONS_SURFACE)
    # A: vacant at slot 0 = (2,0,0), hop (2,0,0) — vacancy IS the destination
    env_a = np.ones(n, dtype=np.int8)
    env_a[0] = 0
    canon_a = canonicalise(env_a, (2, 0, 0), PERMS_SURFACE)
    fp_a = fingerprint_string(canon_a[0], canon_a[1])

    # B: vacant at slot 8 = (2,2,0), hop (2,0,0) — vacancy is a 2NN
    # diagonal, NOT the destination. Different geometric relationship.
    env_b = np.ones(n, dtype=np.int8)
    env_b[8] = 0
    canon_b = canonicalise(env_b, (2, 0, 0), PERMS_SURFACE)
    fp_b = fingerprint_string(canon_b[0], canon_b[1])

    assert fp_a != fp_b, (
        f"distinct environments collapsed to same fingerprint: "
        f"both = {fp_a}"
    )


# ---------------------------------------------------------------------------
# Test 6.4 — Fingerprint string determinism
# ---------------------------------------------------------------------------

def test_fingerprint_string_format():
    """Output string has expected layout."""
    n = len(SHELL_REF_POSITIONS_SURFACE)
    env = np.zeros(n, dtype=np.int8)
    env[0] = 1
    fp = fingerprint_string(env, (1, 2, 3), kind="occ")
    assert fp.startswith("occ:1NN+2NN/")
    assert "|h:1,2,3" in fp
    # 13 bits → 4-4-4-1
    parts = fp.split("/")[1].split("|")[0]
    runs = parts.split("-")
    assert len(runs) == 4
    assert len(runs[0]) == 4
    assert len(runs[3]) == 1


def test_fingerprint_string_deterministic():
    """Same inputs → same output, no dict-iteration leakage."""
    env = (0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1)
    hop = (1, 0, -1)
    s1 = fingerprint_string(env, hop)
    s2 = fingerprint_string(env, hop)
    assert s1 == s2


# ---------------------------------------------------------------------------
# Edge case: full vacuum environment
# ---------------------------------------------------------------------------

def test_empty_environment_canonicalises():
    """An all-vacant environment + hop produces a canonical form without crashing."""
    n = len(SHELL_REF_POSITIONS_SURFACE)
    env = np.zeros(n, dtype=np.int8)
    canon = canonicalise(env, (2, 0, 0), PERMS_SURFACE)
    assert canon[0] == tuple([0] * n)
    # Hop should be canonical lex-min over D4 orbit of (2,0,0)
    # Orbit: (2,0,0), (0,2,0), (-2,0,0), (0,-2,0). Lex-min = (-2, 0, 0)
    assert canon[1] == (-2, 0, 0)


def test_full_environment_canonicalises():
    """Fully-occupied environment + hop → canonical form."""
    n = len(SHELL_REF_POSITIONS_SURFACE)
    env = np.ones(n, dtype=np.int8)
    canon = canonicalise(env, (2, 0, 0), PERMS_SURFACE)
    assert canon[0] == tuple([1] * n)
    assert canon[1] == (-2, 0, 0)
