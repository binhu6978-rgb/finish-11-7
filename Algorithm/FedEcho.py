"""FedEcho, manuscript equations (1)--(29). No reset or data augmentation.

All tensor geometry is streamed over trainable parameters. BatchNorm buffers
retain the baseline dispatch/aggregation behavior and are not control variables.
"""

import numpy as np
import torch


def solve_dispatch(beta, weights, component_norms, radius):
    """Solve the JOINT trust-ball/weighted-consensus projection in <=2D.

    z_i,a = alpha_i,a * ||d_a|| / radius has a unit Euclidean ball.
    Its target is c_i,a = -beta_i,a * ||d_a|| / (2 radius).
    The dual gives z_i = projection_ball(c_i - lambda).
    """
    beta = np.asarray(beta, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    norms = np.asarray(component_norms, dtype=np.float64)
    if beta.ndim != 2 or beta.shape != (len(weights), len(norms)):
        raise ValueError("Invalid dispatch shapes")
    if (not np.isfinite(beta).all() or not np.isfinite(weights).all()
            or not np.isfinite(norms).all() or not np.isfinite(radius)
            or (weights <= 0).any()):
        raise ValueError("Nonfinite prediction or nonpositive weight")
    weights = weights / weights.sum()
    if radius <= 0 or (norms <= 0).any():
        raise ValueError("Only valid components may enter the solver")
    c = -0.5 * beta * (norms / radius)

    def project(lam):
        v = c - lam
        return v / np.maximum(1.0, np.linalg.norm(v, axis=1))[:, None]

    unconstrained = c - weights @ c
    if np.max(np.linalg.norm(unconstrained, axis=1)) <= 1:
        z = unconstrained
    elif len(norms) == 1:
        lo, hi = float(c.min() - 1), float(c.max() + 1)
        for _ in range(100):
            mid = (lo + hi) / 2
            if float(weights @ project(np.array([mid]))[:, 0]) > 0:
                lo = mid
            else:
                hi = mid
        z = project(np.array([(lo + hi) / 2]))
    else:
        def dual(lam):
            v = c - lam
            r = np.linalg.norm(v, axis=1)
            value = weights @ np.where(r <= 1, 0.5 * r**2, r - 0.5)
            return float(value)

        lam = weights @ c
        for _ in range(200):
            v = c-lam
            lengths = np.linalg.norm(v, axis=1)
            response = weights @ project(lam)
            if np.linalg.norm(response) < 1e-12:
                break
            hessian = np.zeros((2, 2))
            for i, length in enumerate(lengths):
                unit = v[i]/max(length, 1e-30)
                jac = np.eye(2) if length <= 1 else (np.eye(2)-np.outer(unit, unit))/length
                hessian += weights[i]*jac
            hessian += np.eye(2)*1e-10
            # Explicit 2x2 inverse avoids the host's broken LAPACK DLL.
            a, b, d = hessian[0, 0], hessian[0, 1], hessian[1, 1]
            det = a*d-b*b
            direction = np.array([d*response[0]-b*response[1],
                                  a*response[1]-b*response[0]])/max(det, 1e-30)
            before = dual(lam)
            step = 1.
            for _ in range(80):
                trial = lam + step*direction
                if dual(trial) <= before - 1e-4*step*float(response @ direction):
                    lam = trial
                    break
                step *= .5
            else:
                break
        z = project(lam)
        if np.linalg.norm(weights @ z) > 1e-10:
            # Dykstra projection in the p-weighted norm is an independent
            # numerical route to the SAME unique convex projection.
            z, correction_a, correction_b = c.copy(), np.zeros_like(c), np.zeros_like(c)
            for _ in range(100000):
                v = z + correction_a
                centered = v - weights @ v
                correction_a = v-centered
                v = centered + correction_b
                updated = v/np.maximum(1., np.linalg.norm(v, axis=1))[:, None]
                correction_b = v-updated
                change = np.max(abs(updated-z))
                z = updated
                if change < 1e-12 and np.linalg.norm(weights @ z) < 1e-11:
                    break
    residual = float(np.linalg.norm(weights @ z))
    if residual > 1e-9 or not np.isfinite(z).all():
        raise RuntimeError(f"FedEcho joint solver failed: consensus={residual}")
    # Remove only solver-roundoff residual; common scaling preserves consensus.
    z -= weights @ z
    z /= max(1.0, float(np.max(np.linalg.norm(z, axis=1))))
    return z * (radius / norms)


class Persistence:
    """Exact pooled SSE sufficient statistics indexed by participation gap.

    Deterministic 1025-point bracketing + refinement of every grid local minimum
    and both endpoints. This is a numerical solver, not an accuracy-tuned decay.
    """

    def __init__(self):
        self.by_gap = {}
        self.count = 0

    def add(self, previous, current, gap):
        if gap < 1 or not np.isfinite([previous, current]).all():
            raise ValueError("Invalid consecutive response pair")
        sums = self.by_gap.setdefault(int(gap), [0.0, 0.0, 0.0])
        sums[0] += previous**2
        sums[1] += previous * current
        sums[2] += current**2
        self.count += 1

    def estimate(self):
        if not self.count:
            return 0.0
        gaps = np.array(sorted(self.by_gap), dtype=float)
        stats = np.array([self.by_gap[int(g)] for g in gaps])
        # The constant sum(current**2) is omitted for numerical stability.
        def objective(rho):
            decay = rho ** gaps
            return float(np.sum(stats[:, 0] * decay**2 - 2 * stats[:, 1] * decay))
        grid = np.linspace(0, 1, 1025)
        decay = grid[:, None] ** gaps[None, :]
        values = np.sum(stats[:, 0] * decay**2 - 2 * stats[:, 1] * decay, axis=1)
        candidates = [(float(values[0]), 0.0), (float(values[-1]), 1.0)]
        interior = np.flatnonzero((values[1:-1] <= values[:-2]) &
                                 (values[1:-1] <= values[2:])) + 1
        for j in [0, *interior.tolist(), len(grid)-1]:
            # Flat zero-information pools otherwise have 1023 redundant solves.
            low_index, high_index = max(0, j-1), min(len(grid)-1, j+1)
            if values[low_index] == values[j] == values[high_index]:
                continue
            lo, hi = grid[low_index], grid[high_index]
            golden = (np.sqrt(5.)-1.)/2.
            left, right = hi-golden*(hi-lo), lo+golden*(hi-lo)
            fl, fr = objective(left), objective(right)
            for _ in range(80):
                if fl <= fr:
                    hi, right, fr = right, left, fl
                    left = hi-golden*(hi-lo)
                    fl = objective(left)
                else:
                    lo, left, fl = left, right, fr
                    right = lo+golden*(hi-lo)
                    fr = objective(right)
            value = (lo+hi)/2.
            candidates.append((objective(value), value))
        return min(candidates)[1]


def make_basis(model, current, previous, epsilon=1e-8, rank=2):
    if rank not in (1, 2) or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Positive finite trajectory tolerance and rank 1/2 required")
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    conv = {f"{name}.weight" if name else "weight" for name, mod in model.named_modules()
            if isinstance(mod, torch.nn.Conv2d)}
    basis = [{}, {}] if rank == 2 else [{}]
    squared = np.zeros(len(basis))
    if previous is None:
        return [], np.zeros(0), 0.0, []
    for name in names:
        step = current[name].detach() - previous[name].to(current[name].device)
        if rank == 1:
            pieces = [step]
        elif name in conv:
            mean = step.mean(dim=(-2, -1), keepdim=True)
            spatial = step - mean
            pieces = [step - spatial, spatial]
        else:
            pieces = [step, torch.zeros_like(step)]
        for a, piece in enumerate(pieces):
            basis[a][name] = piece
            squared[a] += float(torch.sum(piece.double().square()).item())
    radius = float(np.sqrt(squared.sum()))
    if not np.isfinite(radius):
        raise RuntimeError("Nonfinite global trajectory")
    valid = [a for a, n2 in enumerate(squared) if np.sqrt(n2) >= epsilon]
    return [basis[a] for a in valid], np.sqrt(squared[valid]), radius, valid


def project_update(returned, start, basis, norms):
    dots = np.zeros(len(basis), dtype=np.float64)
    for name in basis[0] if basis else []:
        update = returned[name].detach().double() - start[name].detach().double()
        for a, direction in enumerate(basis):
            dots[a] += float(torch.sum(update * direction[name].double()).item())
    return dots / np.square(norms)


class FedEchoController:
    def __init__(self, rank=2, epsilon=1e-8):
        self.rank = rank
        self.epsilon = epsilon
        self.histories = {}
        self.persistence = Persistence()

    def dispatch(self, clients, weights, round_number, norms, radius, components):
        rho = self.persistence.estimate() if components else 0.0
        beta = np.zeros((len(clients), len(components)))
        has_history = 0
        for j, client in enumerate(clients):
            history = self.histories.get(int(client), {})
            for a, component in enumerate(components):
                if component in history:
                    delta, observed = history[component]
                    if observed >= round_number:
                        raise RuntimeError("Future history leakage")
                    beta[j, a] = rho ** (round_number - observed) * delta
                    has_history += 1
        alpha = solve_dispatch(beta, weights, norms, radius) if components else beta
        return alpha, beta, rho, has_history

    def observe(self, clients, weights, round_number, projections, components):
        projections = np.asarray(projections, dtype=float)
        weights = np.asarray(weights, dtype=float)
        relative = projections - weights @ projections
        for j, client in enumerate(clients):
            if not components:
                continue
            history = self.histories.setdefault(int(client), {})
            for a, component in enumerate(components):
                delta = float(relative[j, a])
                if not np.isfinite(delta):
                    raise RuntimeError("Nonfinite response")
                if component in history:
                    previous, observed = history[component]
                    self.persistence.add(previous, delta, round_number - observed)
                history[component] = (delta, int(round_number))
        return relative
