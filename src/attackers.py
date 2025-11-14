from __future__ import annotations

import random
from typing import List, Sequence, Tuple



class Attacker:
    """
    Represents a single attacker that moves linearly from a start position
    to a target position in a given number of steps.
    """

    def __init__(self, start_position:tuple, target_position: tuple, steps: int):
        self.start_position = start_position
        self.target_position = target_position
        self.steps = steps
        self.trajectory = self.generate_trajectory()

    def generate_trajectory(self):
        """
        Generate a linear trajectory from start to target (inclusive),
        with `self.steps` positions.
        """
        # Degenerate case: a single step just returns the start position.
        if self.steps == 1:
            return [self.start_position]

        dimension = len(self.start_position)
        trajectory = []

        for i in range(self.steps):
            # Parameter t goes from 0.0 (start) to 1.0 (target) inclusive.
            t = i / (self.steps - 1)
            position = tuple(
                self.start_position[d]
                + t * (self.target_position[d] - self.start_position[d])
                for d in range(dimension)
            )
            trajectory.append(position)

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
    """

    def __init__(
        self,
        start_position,
        target_positions,
        number_of_attackers: int = 1,
        spread: float = 0.0,
    ):
        if number_of_attackers < 1:
            raise ValueError("number_of_attackers must be at least 1")

        if not target_positions:
            raise ValueError("At least one target position must be provided")

        self.start_position = start_position
        self.target_positions = list(target_positions)
        self.number_of_attackers: int = number_of_attackers
        self.spread: float = spread

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
        """
        return self.target_positions[idx % len(self.target_positions)]

    def generate_swarm(self, steps: int) -> List[Attacker]:
        """
        Generate and return a list of `Attacker` instances.

        Each attacker:
        - Starts at a position sampled around `start_position` (controlled by `spread`).
        - Is assigned a target chosen from `target_positions`, distributed evenly.
        """
        swarm: List[Attacker] = []

        for i in range(self.number_of_attackers):
            attacker_start = self._sample_start_around_center()
            attacker_target = self._target_for_index(i)
            swarm.append(Attacker(attacker_start, attacker_target, steps=steps))

        return swarm


if __name__ == "__main__":
    attacker_swarm = AttackerSwarm(start_position=(0, 0), target_positions=[(1, 1), (2, 2)], number_of_attackers=2, spread=0.1)
    swarm = attacker_swarm.generate_swarm(steps=10)
    for attacker in swarm:
        print(attacker.start_position)
        print(attacker.target_position)
        print(attacker.trajectory)
