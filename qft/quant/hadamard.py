"""Orthogonal rotation matrices for block-input rotation (QServe App. B.2.1, QuaRot).

A Hadamard matrix H_n (entries +-1, H H^T = n I) exists for n = 1, 2 and (conjecturally) every
multiple of 4.  We construct
  * Sylvester matrices for powers of two,
  * Paley-I matrices of order q+1 for prime powers q = 3 (mod 4)  (e.g. 12, 20, 28, 44, ...),
  * Kronecker products of the two.
Qwen2.5-0.5B has hidden size 896 = 28 * 32, so H_896 = H_28 (Paley, q = 27 = 3^3) (x) H_32.
If no construction is found we fall back to a seeded random orthogonal matrix.
"""
from __future__ import annotations

import itertools
import math

import torch


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    return all(n % d for d in range(2, int(math.isqrt(n)) + 1))


def _prime_power(q: int):
    for p in range(2, q + 1):
        if _is_prime(p):
            k, m = 0, q
            while m % p == 0:
                m //= p
                k += 1
            if m == 1 and k > 0:
                return p, k
            if k > 0:
                return None
    return None


class _GF:
    """Tiny finite field GF(p^k) (k <= 3) with elements encoded as ints in [0, p^k)."""

    def __init__(self, p: int, k: int):
        self.p, self.k, self.q = p, k, p ** k
        self.mod = self._find_irreducible() if k > 1 else None
        self.elems = [self._decode(i) for i in range(self.q)]
        self.index = {e: i for i, e in enumerate(self.elems)}

    def _decode(self, i):
        coeffs = []
        for _ in range(self.k):
            coeffs.append(i % self.p)
            i //= self.p
        return tuple(coeffs)

    def _find_irreducible(self):
        p, k = self.p, self.k
        if k > 3:
            raise NotImplementedError("GF(p^k) only implemented for k <= 3")
        for tail in itertools.product(range(p), repeat=k):
            poly = list(tail) + [1]          # monic, low-order first
            # degree <= 3: irreducible <=> no root in GF(p)
            if all(sum(c * pow(x, j, p) for j, c in enumerate(poly)) % p for x in range(p)):
                return poly
        raise RuntimeError("no irreducible polynomial found")

    def sub(self, a: int, b: int) -> int:
        ea, eb = self.elems[a], self.elems[b]
        return self.index[tuple((x - y) % self.p for x, y in zip(ea, eb))]

    def mul(self, a: int, b: int) -> int:
        ea, eb = self.elems[a], self.elems[b]
        if self.k == 1:
            return (ea[0] * eb[0]) % self.p
        prod = [0] * (2 * self.k - 1)
        for i, x in enumerate(ea):
            for j, y in enumerate(eb):
                prod[i + j] = (prod[i + j] + x * y) % self.p
        for d in range(len(prod) - 1, self.k - 1, -1):   # reduce mod irreducible
            c = prod[d]
            if c:
                for j in range(self.k + 1):
                    prod[d - self.k + j] = (prod[d - self.k + j] - c * self.mod[j]) % self.p
        return self.index[tuple(prod[: self.k])]


def paley1(q: int) -> torch.Tensor:
    """Paley construction I: Hadamard matrix of order q+1 for prime power q = 3 (mod 4)."""
    pk = _prime_power(q)
    if pk is None or q % 4 != 3:
        raise ValueError(f"Paley-I needs a prime power q = 3 mod 4, got {q}")
    F = _GF(*pk)
    squares = {F.mul(x, x) for x in range(1, q)}

    def chi(a):
        return 0 if a == 0 else (1 if a in squares else -1)

    Qm = torch.tensor([[chi(F.sub(i, j)) for j in range(q)] for i in range(q)], dtype=torch.float64)
    n = q + 1
    S = torch.zeros(n, n, dtype=torch.float64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Qm
    H = S + torch.eye(n, dtype=torch.float64)
    return H


def sylvester(n: int) -> torch.Tensor:
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def _is_hadamard(H: torch.Tensor) -> bool:
    n = H.shape[0]
    return torch.allclose(H @ H.T, n * torch.eye(n, dtype=H.dtype))


def hadamard_matrix(n: int) -> torch.Tensor | None:
    """Unnormalized Hadamard matrix of order n (or None if no construction available)."""
    if n & (n - 1) == 0:
        return sylvester(n)
    m, pow2 = n, 1
    while m % 2 == 0:
        m //= 2
        pow2 *= 2
    # try every split n = base * 2^a where base = q + 1 is Paley-constructible
    a_pow = pow2
    while a_pow >= 1:
        base = n // a_pow
        q = base - 1
        if base % 4 == 0 and _prime_power(q) is not None and q % 4 == 3:
            H = torch.kron(paley1(q), sylvester(a_pow))
            if _is_hadamard(H):
                return H
        a_pow //= 2
    return None


def random_orthogonal(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    Q, R = torch.linalg.qr(A)
    return Q * torch.sign(torch.diagonal(R)).unsqueeze(0)


def get_rotation(n: int, kind: str = "hadamard", seed: int = 0) -> torch.Tensor:
    """Orthogonal n x n matrix (float64).

    kind: ``hadamard`` (H / sqrt(n)), ``random_hadamard`` (H diag(+-1) / sqrt(n), QuaRot) or
    ``orthogonal`` (Haar-random orthogonal matrix).
    """
    if kind in ("hadamard", "random_hadamard"):
        H = hadamard_matrix(n)
        if H is None:
            print(f"[rotation] no Hadamard construction for n={n}; using random orthogonal matrix")
            return random_orthogonal(n, seed)
        H = H / math.sqrt(n)
        if kind == "random_hadamard":
            g = torch.Generator().manual_seed(seed)
            signs = torch.randint(0, 2, (n,), generator=g).to(torch.float64) * 2 - 1
            H = signs.unsqueeze(1) * H          # diag(s) H  (still orthogonal)
        return H
    if kind == "orthogonal":
        return random_orthogonal(n, seed)
    raise ValueError(f"unknown rotation kind {kind}")
