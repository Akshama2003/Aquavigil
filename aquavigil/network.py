"""A small synthetic water network and its leak sensitivity matrix.

The network is an n x n grid of junctions fed from a reservoir at node 0.
Pressure sensors sit on a subset of junctions. The sensitivity matrix S gives,
for a unit leak on pipe p, the pressure drop seen at sensor s. It decays with
the distance between the sensor and the pipe, and is larger far from the
reservoir (where there is less head to spare).

This is a *linearized surrogate*, not a full hydraulic solver. See README
("What is simulated") for what that means for the claims you can make.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Network:
    n: int
    decay: float
    coords: np.ndarray  # (n*n, 2) junction (row, col)
    sensors: list  # junction indices that carry a pressure sensor
    pipes: list  # (u, v) junction index pairs
    pipe_mid: np.ndarray  # (n_pipes, 2) pipe midpoints
    S: np.ndarray  # (n_sensors, n_pipes) nominal sensitivity

    @property
    def n_pipes(self) -> int:
        return len(self.pipes)

    @property
    def n_sensors(self) -> int:
        return len(self.sensors)

    def pipe_label(self, p: int) -> str:
        u, v = self.pipes[p]
        r1, c1 = divmod(u, self.n)
        r2, c2 = divmod(v, self.n)
        return f"P{p:02d} (junction {r1},{c1} - junction {r2},{c2})"

    def pipe_distance(self, p: int, q: int) -> float:
        """Grid distance (in pipe lengths) between the midpoints of two pipes."""
        return float(np.abs(self.pipe_mid[p] - self.pipe_mid[q]).sum())

    def dist_to_reservoir(self, node: int) -> float:
        return float(np.abs(self.coords[node] - self.coords[0]).sum())


def build_grid(n: int = 6, n_sensors: int = 8, decay: float = 1.5) -> Network:
    pipes = []
    for r in range(n):
        for c in range(n):
            i = r * n + c
            if c + 1 < n:
                pipes.append((i, i + 1))
            if r + 1 < n:
                pipes.append((i, i + n))

    coords = np.array([divmod(i, n) for i in range(n * n)], dtype=float)
    pipe_mid = np.array([(coords[u] + coords[v]) / 2 for u, v in pipes])

    # Spread sensors out: greedy farthest-point selection, starting far from the reservoir.
    chosen = [n * n - 1]
    while len(chosen) < n_sensors:
        d = np.min(
            [np.abs(coords - coords[c]).sum(axis=1) for c in chosen], axis=0
        )
        d[chosen] = -1
        chosen.append(int(np.argmax(d)))

    S = np.zeros((len(chosen), len(pipes)))
    for i, s in enumerate(chosen):
        d = np.abs(pipe_mid - coords[s]).sum(axis=1)
        w = 1.0 + 0.1 * np.abs(coords[s] - coords[0]).sum()
        S[i] = w * np.exp(-d / decay)
    S /= S.max()

    return Network(
        n=n,
        decay=decay,
        coords=coords,
        sensors=chosen,
        pipes=pipes,
        pipe_mid=pipe_mid,
        S=S,
    )
