from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from core import config


class SpectralMap:
    def __init__(self, dim: int = config.EMBED_DIM):
        self.dim = dim
        self.V2: np.ndarray | None = None
        self.singular_values: np.ndarray | None = None
        self.eta = 0.0
        self.n_rows = 0

    def fit_blocks(self, blocks: Iterable[np.ndarray]) -> SpectralMap:
        G = np.zeros((self.dim, self.dim), dtype=np.float64)
        n = 0
        for B in blocks:
            B = np.asarray(B, dtype=np.float64)
            G += B.T @ B
            n += B.shape[0]
        if n == 0:
            raise ValueError("cannot fit on an empty corpus")
        eigvals, eigvecs = np.linalg.eigh(G)
        order = np.argsort(eigvals)[::-1][: min(n, self.dim)]
        lam = np.clip(eigvals[order], 0.0, None)
        V = eigvecs[:, order[:2]]
        V *= np.sign(V[np.abs(V).argmax(axis=0), [0, 1]])
        self.V2 = np.ascontiguousarray(V, dtype=np.float32)
        self.singular_values = np.sqrt(lam)
        self.eta = float(lam[:2].sum() / lam.sum()) if lam.sum() > 0 else 0.0
        self.n_rows = n
        return self

    def fit(self, D: np.ndarray, rows: int = config.SCORE_BLOCK_ROWS) -> SpectralMap:
        return self.fit_blocks(D[s:s + rows] for s in range(0, D.shape[0], rows))

    def project(self, X: np.ndarray) -> np.ndarray:
        if self.V2 is None:
            raise RuntimeError("fit() first")
        return np.asarray(X, dtype=np.float32) @ self.V2

    def to_dict(self) -> dict:
        return {"V2": self.V2.tolist(), "singular_values": self.singular_values.tolist(),
                "eta": self.eta, "n_rows": self.n_rows}

    @classmethod
    def from_dict(cls, d: dict) -> SpectralMap:
        m = cls(len(d["V2"]))
        m.V2 = np.asarray(d["V2"], dtype=np.float32)
        m.singular_values = np.asarray(d["singular_values"])
        m.eta, m.n_rows = d["eta"], d["n_rows"]
        return m


def _self_check() -> None:
    from core.dense import normalize

    rng = np.random.default_rng(0)
    latent = rng.normal(size=(2_000, 384)) * np.r_[8.0, 5.0, np.full(382, 1.0)]
    D = normalize(latent @ np.linalg.qr(rng.normal(size=(384, 384)))[0])

    print("Gram-eigh vs np.linalg.svd")
    m = SpectralMap().fit(D, rows=300)
    _, S, Vt = np.linalg.svd(D.astype(np.float64), full_matrices=False)
    align = np.abs(np.sum(m.V2 * Vt[:2].T, axis=0))
    assert np.all(align > 1 - 1e-5), align
    assert np.allclose(m.singular_values, S, atol=1e-6)
    eta_svd = (S[0] ** 2 + S[1] ** 2) / (S ** 2).sum()
    assert abs(m.eta - eta_svd) < 1e-9 and 0 < m.eta <= 1
    print(f"  ok  |<v_i, v_i_svd>| = {np.round(align, 7)}, sigma match, eta = {m.eta:.4f}")

    print("Eckart-Young-Mirsky")
    V2 = m.V2.astype(np.float64)
    err = np.linalg.norm(D - D @ V2 @ V2.T, "fro") ** 2
    assert abs(err - (S[2:] ** 2).sum()) < 1e-6 * err
    rand = np.linalg.qr(rng.normal(size=(384, 2)))[0]
    assert np.linalg.norm(D - D @ rand @ rand.T, "fro") ** 2 > err
    print(f"  ok  ||D - D2||_F^2 = sum_(j>2) sigma_j^2 = {err:.4f}; a random rank-2 basis does worse")

    print("projection + persistence")
    X2 = m.project(D)
    q2 = m.project(D[0])
    assert X2.shape == (2_000, 2) and q2.shape == (2,) and np.allclose(X2[0], q2)
    assert np.allclose(SpectralMap.from_dict(m.to_dict()).project(D[:5]), m.project(D[:5]), atol=1e-7)
    small = SpectralMap().fit(D[:3])
    assert small.singular_values.shape == (3,), "only min(N, d) singular values exist"
    print("  ok  X2D (N, 2), q2D (2,), N < d handled, round-trips")
    print("core.svd_map: all checks passed")


if __name__ == "__main__":
    _self_check()
