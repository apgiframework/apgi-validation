"""
TIER DESIGNATION: Tier 1 (thermodynamic)

"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.apgi_config import get_apgi_settings
from utils.apgi_settings import APGISettings


class APGIPreProcessor:
    """Section 1: Signal Preprocessing"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.errors: List[float] = []

    def compute_prediction_error(self, x: float, x_hat: float) -> float:
        """Eq 1.1: epsilon = x - x_hat"""
        return x - x_hat

    def update_statistics(self, epsilon: float) -> Tuple[float, float]:
        """Eq 1.2: Running Mean / Variance"""
        self.errors.append(epsilon)
        if len(self.errors) > self.window_size:
            self.errors.pop(0)

        mu = float(np.mean(self.errors)) if self.errors else 0.0
        sigma2 = float(np.var(self.errors)) if self.errors else 1.0
        return mu, sigma2

    def standardize(self, epsilon: float, mu: float, sigma: float) -> float:
        """Eq 1.3: Z-Score Standardization (z = (epsilon - mu) / sigma)"""
        return (epsilon - mu) / (sigma + 1e-9)


class APGIPrecisionSystem:
    """Section 2: Precision System"""

    def __init__(self, cfg: APGISettings, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        self.cfg = cfg
        # Single recorded random stream (APGI-REP-002); no global NumPy state.
        self.seed = seed
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)
        self.precision = cfg.Pi_e_init

    def compute_precision(self, sigma2: float) -> float:
        """Eq 2.1: Pi = 1 / sigma^2"""
        return 1.0 / (sigma2 + 1e-9)

    def effective_interoceptive_precision(self, pi_baseline: float, beta: float, m_ca: float) -> float:
        """Eq 2.2: Pi_eff = Pi_baseline * exp(beta * M(c,a))"""
        return float(pi_baseline * np.exp(beta * m_ca))

    def precision_ode(
        self,
        pi: float,
        epsilon: float,
        pi_next: float,
        pi_prev_psi: float,
        alpha: float,
        tau: float,
        c_down: float,
        c_up: float,
    ) -> float:
        """Eq 2.3: Precision Dynamics (ODE)"""
        # dPi/dt = -Pi/tau + alpha|epsilon| + C_down(Pi_l+1 - Pi_l) + C_up * psi(epsilon_l-1)
        return (-pi / tau) + (alpha * abs(epsilon)) + (c_down * (pi_next - pi)) + (c_up * pi_prev_psi)


class APGICoreSignal:
    """Section 3: Core APGI Signal"""

    @staticmethod
    def accumulated_signal(pi_e: float, ze: float, pi_i_eff: float, zi: float) -> float:
        """Eq 3.1: St = Pi_e |ze| + Pi_i_eff |zi|"""
        return pi_e * abs(ze) + pi_i_eff * abs(zi)


class APGIIgnitionMechanism:
    """Section 4: Ignition Mechanism"""

    @staticmethod
    def logistic_ignition(s_t: float, theta_t: float, alpha: float) -> float:
        """Eq 4.1: Bt = 1 / (1 + exp(-alpha * (St - theta_t)))"""
        # Use clip to avoid overflow in exp
        val = -alpha * (s_t - theta_t)
        return float(1.0 / (1.0 + np.exp(np.clip(val, -100, 100))))

    @staticmethod
    def hard_ignition(s_t: float, theta_t: float) -> bool:
        """Eq 4.2: Ignition = I(St > theta_t)"""
        return bool(s_t > theta_t)

    @staticmethod
    def ignition_margin(s_t: float, theta_t: float) -> float:
        """Eq 4.3: Delta_t = St - theta_t"""
        return s_t - theta_t


class APGISystemDynamics:
    """Section 5 & 8: Continuous Dynamics and SDE"""

    def __init__(self, cfg: APGISettings, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        self.cfg = cfg
        # Single recorded random stream (APGI-REP-002); no global NumPy state.
        self.seed = seed
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)

    def signal_dynamics(
        self,
        s_t: float,
        pi_e: float,
        ze: float,
        pi_i: float,
        zi: float,
        beta: float,
        tau_s: float,
        dt: float,
    ) -> float:
        """Eq 5.1 & 8.1: Signal Dynamics (ODE/SDE)"""
        # dS/dt = -S/tau_s + Pi_e|ze| + beta * Pi_i|zi| + eta_s(t)
        noise = float(self.rng.normal(0, np.sqrt(dt)))
        ds_dt = (-s_t / tau_s) + (pi_e * abs(ze)) + (beta * pi_i * abs(zi))
        return float(s_t + ds_dt * dt + noise)

    def threshold_dynamics(
        self,
        theta_t: float,
        theta_0: float,
        b_prev: float,
        ds_dt: float,
        gamma: float,
        delta: float,
        lambda_urg: float,
        dt: float,
    ) -> float:
        """Eq 5.2: Threshold Dynamics"""
        # dtheta/dt = gamma(theta_0 - theta_t) + delta * B_t-1 - lambda * |dS/dt|
        dtheta_dt = gamma * (theta_0 - theta_t) + (delta * b_prev) - (lambda_urg * abs(ds_dt))
        return theta_t + dtheta_dt * dt


class APGIAllostaticLayer:
    """Section 6 & 7: Discrete Allostatic and Energy Layer"""

    @staticmethod
    def threshold_update(theta_t: float, eta: float, c_met: float, v_inf: float) -> float:
        """Eq 6.1: Cost-Value Threshold Update"""
        return theta_t + eta * (c_met - v_inf)

    @staticmethod
    def metabolic_cost(bits_erased: float, kappa: float) -> float:
        """Eq 7.1: Metabolic Cost"""
        return kappa * bits_erased

    @staticmethod
    def landauer_limit(k: float, t: float) -> float:
        """Eq 7.2: Landauer Limit (E_min >= kT ln 2)"""
        return float(k * t * np.log(2))


class APGILiquidNeuralNetwork:
    """Section 9: Liquid Neural Network (Reservoir)"""

    def __init__(self, size: int, rng: Optional[np.random.Generator] = None, seed: Optional[int] = None):
        self.size = size
        self.seed = seed
        self.rng: np.random.Generator = rng if rng is not None else np.random.default_rng(seed)
        self.x = np.zeros(size)
        self.W_res = self.rng.standard_normal((size, size)) * (1.0 / np.sqrt(size))
        self.W_in = self.rng.standard_normal((size, 1))

    def reservoir_dynamics(self, x: np.ndarray, u: float, tau_t: float, dt: float) -> np.ndarray:
        """Eq 9.1: Reservoir Dynamics"""
        # x_dot = -x/tau(t) + f(W_res x + W_in u)
        # Using tanh as f
        dx_dt = (-x / tau_t) + np.tanh(self.W_res @ x + self.W_in.flatten() * u)
        res: np.ndarray = x + dx_dt * dt
        return res

    def signal_readout(self, x: np.ndarray) -> float:
        """Eq 9.2: Signal Readout (S = x^T x)"""
        return float(x.T @ x)

    def suprathreshold_amplification(
        self,
        x: np.ndarray,
        s_t: float,
        theta_t: float,
        alpha: float,
        a_amp: float,
        dt: float,
    ) -> np.ndarray:
        """Eq 9.3: Suprathreshold Amplification"""
        # dx/dt = -alpha x + ... + A * x * [S - theta_t]+
        amp = a_amp * x * max(0, s_t - theta_t)
        dx_dt = -alpha * x + amp
        return x + dx_dt * dt


class APGIHierarchy:
    """Section 10 & 11: Hierarchical and Oscillatory Layer"""

    @staticmethod
    def level_count(tau_max: float, tau_min: float, overlap: float) -> int:
        """Eq 10.1: Level Count"""
        if overlap <= 1:
            return 1
        # Handle edge case where tau_max == tau_min (log(1) = 0)
        if tau_max <= tau_min:
            return 1
        return int(np.ceil(np.log(tau_max / tau_min) / np.log(overlap)))

    @staticmethod
    def cross_level_modulation(theta_0: float, pi_next: float, phi_next: float, kappa_down: float) -> float:
        """Eq 10.2: Cross-Level Threshold Modulation"""
        return float(theta_0 * (1 + kappa_down * pi_next * np.cos(phi_next)))

    @staticmethod
    def bottom_up_cascade(theta_l: float, s_prev: float, theta_prev: float, kappa_up: float) -> float:
        """Eq 10.3: Bottom-Up Cascade"""
        # H is Heaviside step function
        h_val = 1.0 if (s_prev - theta_prev) > 0 else 0.0
        return theta_l * (1 - kappa_up * h_val)

    @staticmethod
    def phase_signal(omega: float, t: float, phi_0: float) -> float:
        """Eq 11.1: Phase Signal"""
        return omega * t + phi_0


class APGIRecovery:
    """Section 12: Post-Ignition Reset"""

    @staticmethod
    def reset_rule(s_t: float, theta_t: float, rho: float, delta: float) -> Tuple[float, float]:
        """Eq 12.1: Reset Rule"""
        return s_t * rho, theta_t + delta


class APGIValidationMetrics:
    """Section 13: Statistical Validation"""

    @staticmethod
    def power_spectrum(f: np.ndarray, sigma_l: np.ndarray, tau_l: np.ndarray) -> np.ndarray:
        """Eq 13.1: Power Spectrum (1/f)"""
        # S(f) = sum_l (sigma_l^2 * tau_l^2) / (1 + (2pi f tau_l)^2)
        spectrum = np.zeros_like(f)
        for s, t in zip(sigma_l, tau_l):
            spectrum += (s**2 * t**2) / (1 + (2 * np.pi * f * t) ** 2)
        return spectrum

    @staticmethod
    def hurst_exponent(beta_spec: float) -> float:
        """Eq 13.2: Hurst Exponent (H = (beta + 1) / 2)"""
        return (beta_spec + 1) / 2.0

    @staticmethod
    def bayesian_evidence(data: np.ndarray, model_pred: np.ndarray) -> float:
        """Compute Bayesian evidence approximation."""
        if len(data) == 0 or len(model_pred) == 0:
            raise ValueError("Data and model predictions cannot be empty")

        # Simple evidence approximation based on likelihood
        # E = exp(-0.5 * sum((data - model_pred)^2 / sigma^2))
        # Assuming unit variance for simplicity
        mse = np.mean((data - model_pred) ** 2)
        evidence = np.exp(-0.5 * len(data) * mse)
        return float(evidence)

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Compute Kullback-Leibler divergence D_KL(P||Q)."""
        if len(p) == 0 or len(q) == 0:
            raise ValueError("Probability distributions cannot be empty")

        if np.any(p < 0) or np.any(q < 0):
            raise ValueError("Probabilities must be non-negative")

        # Normalize to ensure they sum to 1
        p_norm = p / np.sum(p)
        q_norm = q / np.sum(q)

        # Avoid division by zero
        q_norm = np.where(q_norm == 0, 1e-10, q_norm)

        # Compute KL divergence: D_KL(P||Q) = sum(P * log(P/Q))
        kl_div = np.sum(p_norm * np.log(p_norm / q_norm))
        return float(kl_div)


class APGISystem:
    """Section 14: Complete Pipeline Orchestrator"""

    def __init__(self, cfg: Optional[APGISettings] = None):
        self.cfg = cfg or get_apgi_settings()
        self.prep_e = APGIPreProcessor()
        self.prep_i = APGIPreProcessor()
        self.precision = APGIPrecisionSystem(self.cfg)
        self.dynamics = APGISystemDynamics(self.cfg)
        self.recovery = APGIRecovery()

        self.s_t = 0.0
        self.theta_t = self.cfg.theta_init
        self.b_t = 0.0

    def step(self, x: float, x_hat: float, x_i: float, x_hat_i: float, m_ca: float = 0.0) -> Dict[str, Any]:
        """
        Executes one time-step of the APGI pipeline.
        """
        # 1. Compute prediction errors
        eps_e = self.prep_e.compute_prediction_error(x, x_hat)
        eps_i = self.prep_i.compute_prediction_error(x_i, x_hat_i)

        # 2. Update stats
        mu_e, sig2_e = self.prep_e.update_statistics(eps_e)
        mu_i, sig2_i = self.prep_i.update_statistics(eps_i)

        # 3. Standardize
        ze = self.prep_e.standardize(eps_e, mu_e, np.sqrt(sig2_e))
        zi = self.prep_i.standardize(eps_i, mu_i, np.sqrt(sig2_i))

        # 4. Compute precision
        pi_e = self.precision.compute_precision(sig2_e)
        pi_i_baseline = self.precision.compute_precision(sig2_i)

        # 5. Apply somatic bias
        pi_i_eff = self.precision.effective_interoceptive_precision(pi_i_baseline, self.cfg.beta_somatic, m_ca)

        # 6. Update system dynamics
        ds_dt = (-self.s_t / self.cfg.tau_S) + (pi_e * abs(ze)) + (self.cfg.beta_somatic * pi_i_eff * abs(zi))
        self.s_t = self.dynamics.signal_dynamics(
            self.s_t,
            pi_e,
            ze,
            pi_i_eff,
            zi,
            self.cfg.beta_somatic,
            self.cfg.tau_S,
            self.cfg.dt_ms / 1000.0,
        )

        # 7. Update threshold
        new_theta = self.dynamics.threshold_dynamics(
            self.theta_t,
            self.cfg.theta0,
            self.b_t,
            ds_dt,
            self.cfg.gamma,
            self.cfg.delta,
            self.cfg.lambda_urg,
            self.cfg.dt_ms / 1000.0,
        )
        # Clamp to prevent negative threshold (biological constraint)
        self.theta_t = max(self.cfg.theta_min, new_theta)

        # 8. Compute ignition BEFORE reset
        self.b_t = APGIIgnitionMechanism.logistic_ignition(self.s_t, self.theta_t, self.cfg.alpha_ignition)
        ignited = APGIIgnitionMechanism.hard_ignition(self.s_t, self.theta_t)
        raw_s = self.s_t

        # 9. Apply reset if ignition
        if ignited:
            # rho=0.1 for signal reset, cfg.delta=0.5 for threshold increment
            self.s_t, self.theta_t = self.recovery.reset_rule(self.s_t, self.theta_t, 0.1, self.cfg.delta)

        return {
            "s_t": self.s_t,
            "raw_s": raw_s,
            "theta_t": self.theta_t,
            "b_t": self.b_t,
            "ignited": ignited,
        }


if __name__ == "__main__":
    # Smoke test for APGISystem
    print("Initializing APGI System...")
    system = APGISystem()

    # Pre-warm statistics to prevent Z-score explosion in small samples
    print("Pre-warming statistics...")
    for _ in range(50):
        system.prep_e.update_statistics(np.random.normal(0, 0.1))
        system.prep_i.update_statistics(np.random.normal(0, 0.1))

    print("\nStarting Simulation Loop (5 steps):")
    print("-" * 50)
    for i in range(5):
        res = system.step(
            x=np.random.normal(0.2, 0.1),
            x_hat=0.0,
            x_i=np.random.normal(0.1, 0.05),
            x_hat_i=0.0,
            m_ca=0.1,
        )
        print(
            f"Step {i + 1:02d} | Signal (raw): {res['raw_s']:.4f} | "
            f"Threshold: {res['theta_t']:.4f} | Prob: {res['b_t']:.4f} | "
            f"Ignited: {res['ignited']}"
        )
    print("-" * 50)
    print("Smoke test completed successfully.")
