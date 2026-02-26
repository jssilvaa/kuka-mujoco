from __future__ import annotations

from dataclasses import dataclass
import numpy as np 


#T0D0
# Change actuators to acc type, implement double integrator for fun 
@dataclass
class Integrator:
    F: np.ndarray   # (2n, 2n)
    G: np.ndarray   # (2n, n)

    @staticmethod
    def make(n: int, h: float) -> Integrator:
        I = np.eye(n)
        F = np.block([[I, h * I],
                      [np.zeros((n, n)), I]])
        G = np.block([[0.5 * h**2 * I],
                      [h * I]])
        return Integrator(F=F, G=G)

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self.F @ x + self.G @ u