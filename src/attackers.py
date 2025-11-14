from __future__ import annotations

import random
import numpy as np
from typing import List, Sequence, Tuple



class Attacker:
    """
    Represents a single attacker that moves linearly from a start position
    to a target position in a given number of steps.
    
    If target_position is None, it will be sampled from target_distribution.
    """

    def __init__(self, start_position:tuple, target_position: tuple = None, steps: int = 10, target_distribution = None, noise_std: float = 0.0, speed : float = 0.0, speed_noise: float = 0.0):
        self.start_position = start_position

        # Sample target from distribution if not provided
        if target_position is None and target_distribution is not None:
            # Sample one point from the target distribution
            sampled = target_distribution.sample(1)  # Returns shape [1, 2]
            self.target_position = tuple(sampled[0])  # Convert to tuple
        elif target_position is not None:
            self.target_position = target_position
        else:
            raise ValueError("Either target_position or target_distribution must be provided")

        self.steps = steps
        # standard deviation of gaussian noise to add to intermediate trajectory points
        self.noise_std = noise_std
        self.speed = speed   
        self.trajectory = self.generate_trajectory()
        self.speed_noise = speed_noise

    def generate_trajectory(self):
        """
        Generate a linear trajectory from start to target.

        If `self.speed` > 0, sample positions at 1-second intervals along the
        straight-line path from start to target: traj[0] is start (t=0),
        traj[1] is position after 1s, etc., and the final element is the target
        (arrival time may be fractional, we always include the exact target as
        the last waypoint).

        If `self.speed` <= 0, fall back to the legacy behavior of producing
        `self.steps` evenly spaced waypoints along the line.
        """
        start = np.asarray(self.start_position, dtype=float)
        target = np.asarray(self.target_position, dtype=float)
        diff = target - start
        dist = np.linalg.norm(diff)

        # If start equals target, just return start
        if dist == 0.0:
            return [tuple(start.tolist())]

        # If speed > 0, produce per-second samples
        if self.speed is not None and self.speed > 0.0:
                # Step simulation: at each 1-second timestep sample an effective speed
                # using additive Gaussian noise (mean=self.speed, std=self.speed_noise),
                # move along the line toward the target by effective_speed*dt (dt=1s).
                # Stop when the remaining distance is reached or a safety cap is hit.
                pos = start.copy()
                remaining = dist
                trajectory = [tuple(pos.tolist())]

                # Safety cap: allow up to 10x the nominal expected steps (with a minimum)
                nominal_steps = max(1, int(np.ceil(dist / max(1e-8, self.speed))))
                max_steps = max(1000, nominal_steps * 10)

                steps_taken = 0
                while remaining > 1e-8 and steps_taken < max_steps:
                    # sample effective speed for this 1-second interval
                    if self.speed_noise > 0.0:
                        eff_speed = float(np.random.normal(loc=self.speed, scale=self.speed_noise))
                    else:
                        eff_speed = float(self.speed)

                    # clamp to non-negative small value to avoid stalling
                    if eff_speed <= 0.0:
                        eff_speed = 0.0

                    move_dist = eff_speed * 1.0  # dt = 1s

                    if move_dist <= 0.0:
                        # cannot advance this second; to avoid infinite loop, break
                        break

                    # distance to move is min(move_dist, remaining)
                    travel = min(move_dist, remaining)
                    direction = diff / (np.linalg.norm(diff) + 1e-12)
                    pos = pos + direction * travel
                    remaining = np.linalg.norm(target - pos)
                    trajectory.append(tuple(pos.tolist()))

                    # update diff for direction recalculation
                    diff = target - pos
                    steps_taken += 1

                # ensure final point is the exact target (precise endpoint)
                if tuple(trajectory[-1]) != tuple(target.tolist()):
                    trajectory.append(tuple(target.tolist()))

                # Optionally add Gaussian noise to intermediate points (not start/end)
                if self.noise_std > 0.0:
                    for idx in range(1, len(trajectory) - 1):
                        pos_arr = np.array(trajectory[idx], dtype=float)
                        noise = np.random.normal(loc=0.0, scale=self.noise_std, size=pos_arr.shape)
                        pos_arr = pos_arr + noise
                        trajectory[idx] = tuple(pos_arr.tolist())

                return trajectory

        # Legacy fallback: evenly spaced samples using self.steps
        if self.steps == 1:
            return [tuple(start.tolist())]

        trajectory = []
        for i in range(self.steps):
            t = i / (self.steps - 1)
            pos = start + t * diff
            if 0 < i < (self.steps - 1) and self.noise_std > 0.0:
                pos_arr = np.array(pos, dtype=float)
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=pos_arr.shape)
                pos_arr = pos_arr + noise
                pos = pos_arr
            trajectory.append(tuple(pos.tolist()))

        return trajectory

    def get_trajectory(self):
        if self.trajectory is None:
            raise ValueError("Trajectory not generated")

        return self.trajectory


class AttackerSwarm:
    """
    Generates a swarm of `Attacker` objects.

    - All attackers start around a common `start_position`.
    - Their individual start positions are sampled from a simple distribution
      around that point, controlled by `spread`.
    - One or more `target_positions` can be provided; if there are multiple,
      they are distributed evenly between the attackers (round‑robin).
    - Alternatively, provide `target_distribution` to sample targets from a distribution.
    """

    def __init__(
        self,
        start_position,
        target_positions = None,
        number_of_attackers: int = 1,
        spread: float = 0.0,
        target_distribution = None,
        noise_std: float = 0.0,
    ):
        if number_of_attackers < 1:
            raise ValueError("number_of_attackers must be at least 1")

        if target_positions is None and target_distribution is None:
            raise ValueError("Either target_positions or target_distribution must be provided")

        self.start_position = start_position
        self.target_positions = list(target_positions) if target_positions is not None else None
        self.number_of_attackers: int = number_of_attackers
        self.spread: float = spread
        self.target_distribution = target_distribution
        self.noise_std = float(noise_std)

    def _sample_start_around_center(self):
        """
        Sample a new start position around `self.start_position`.

        The spread parameter controls how far the individual attacker
        start positions can deviate from the central start position.
        Here we use a simple uniform distribution in each dimension
        in the range [coord - spread, coord + spread].
        """
        if self.spread <= 0:
            # No spread requested: all attackers start exactly at start_position.
            return self.start_position

        return tuple(
            coord + random.uniform(-self.spread, self.spread)
            for coord in self.start_position
        )

    def _target_for_index(self, idx: int):
        """
        Choose a target for the attacker with index `idx`.

        Multiple targets are distributed evenly using a round‑robin scheme.
        If using target_distribution, returns None (will be sampled in Attacker.__init__).
        """
        if self.target_positions is None:
            return None
        return self.target_positions[idx % len(self.target_positions)]

    def generate_swarm(self, steps: int, speed: float = 0.0, speed_noise: float = 0.0) -> List[Attacker]:
        """
        Generate and return a list of `Attacker` instances.

        Each attacker:
        - Starts at a position sampled around `start_position` (controlled by `spread`).
        - Is assigned a target chosen from `target_positions` (round-robin), or 
          sampled from `target_distribution` if provided.
        """
        swarm: List[Attacker] = []

        for i in range(self.number_of_attackers):
            attacker_start = self._sample_start_around_center()
            attacker_target = self._target_for_index(i)
            swarm.append(Attacker(
                attacker_start, 
                attacker_target, 
                steps=steps,
                target_distribution=self.target_distribution,
                noise_std=self.noise_std,
                speed=speed,
                speed_noise=speed_noise,
            ))

        return swarm


if __name__ == "__main__":
    attacker_swarm = AttackerSwarm(start_position=(0, 0), target_positions=[(1, 1), (2, 2)], number_of_attackers=2, spread=0.1)
    swarm = attacker_swarm.generate_swarm(steps=10)
    for attacker in swarm:
        print(attacker.start_position)
        print(attacker.target_position)
        print(attacker.trajectory)
