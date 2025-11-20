from __future__ import annotations

import random
import numpy as np
from typing import List, Sequence, Tuple
from scipy.integrate import solve_ivp



class Attacker:
    """
    Represents a single attacker that moves linearly from a start position
    to a target position in a given number of steps.
    
    If target_position is None, it will be sampled from target_distribution.
    Optionally, waypoints can be specified to create a multi-segment path:
    - Single waypoint: start -> waypoint -> target
    - Multiple waypoints: start -> waypoint1 -> waypoint2 -> ... -> target
    """

    def __init__(self, start_position:tuple, target_position: tuple = None, steps: int = 10, target_distribution = None, noise_std: float = 0.0, speed : float = 0.0, speed_noise: float = 0.0, waypoint: tuple = None, waypoints: list = None, use_dynamic_trajectory: bool = False, trajectory_aggressiveness: float = 2.0):
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
        self.speed_noise = speed_noise
        
        # Support both single waypoint and list of waypoints
        if waypoints is not None:
            # Use list of waypoints
            self.waypoints = waypoints
        elif waypoint is not None:
            # Convert single waypoint to list for consistency
            self.waypoints = [waypoint]
        else:
            self.waypoints = None
        
        # Dynamic trajectory settings (ODE/SDE-based)
        self.use_dynamic_trajectory = use_dynamic_trajectory
        self.trajectory_aggressiveness = trajectory_aggressiveness

        self.trajectory = self.generate_trajectory()

    @property
    def position(self):
        """
        Return the start position of the attacker.
        This property is needed for compatibility with environment visualization
        which expects drones to have a .position attribute.
        """
        return self.start_position
    
    @position.setter
    def position(self, value):
        """
        Set the start position of the attacker.
        This setter is needed for compatibility with code that updates attacker positions.
        """
        self.start_position = value

    def generate_trajectory(self):
        """
        Generate a trajectory from start to target.
        
        If use_dynamic_trajectory=True, uses an ODE/SDE-based system with attraction
        to target and smooth dynamics. Otherwise uses linear interpolation.
        
        If waypoints are specified, generates a multi-segment path:
        - Single waypoint: start -> waypoint -> target
        - Multiple waypoints: start -> wp1 -> wp2 -> ... -> target
        
        If `self.speed` > 0, sample positions at 1-second intervals along the
        straight-line path from start to target: traj[0] is start (t=0),
        traj[1] is position after 1s, etc., and the final element is the target
        (arrival time may be fractional, we always include the exact target as
        the last waypoint).

        If `self.speed` <= 0, fall back to the legacy behavior of producing
        `self.steps` evenly spaced waypoints along the line.
        """
        # If using dynamic trajectory, use ODE/SDE-based generation
        if self.use_dynamic_trajectory:
            return self._generate_dynamic_trajectory()
        
        # If waypoints are specified, generate trajectory with waypoints
        if self.waypoints is not None and len(self.waypoints) > 0:
            return self._generate_trajectory_with_waypoints()
        
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

    def _generate_dynamic_trajectory(self):
        """
        Generate a trajectory using an ODE/SDE-based system.
        
        Supports waypoints: if waypoints are specified, generates dynamic trajectories
        through each segment (start -> wp1 -> wp2 -> ... -> target).
        
        The trajectory is governed by a second-order system with:
        - Attraction force towards the target (proportional to distance)
        - Velocity damping for smooth motion
        - Optional stochastic perturbations (Brownian motion)
        
        This creates more realistic, smooth attack patterns with acceleration/deceleration
        rather than constant-velocity linear motion.
        
        The system is described by:
            d²x/dt² = -γ(dx/dt) + α(x_target - x) + σξ(t)
        where:
            γ = damping coefficient (prevents oscillation)
            α = attraction strength (aggressiveness parameter)
            σ = noise intensity (from self.noise_std)
            ξ(t) = white noise process
        """
        # If waypoints specified, generate multi-segment dynamic trajectory
        if self.waypoints is not None and len(self.waypoints) > 0:
            return self._generate_dynamic_trajectory_with_waypoints()
        
        # Single segment: start -> target
        start = np.asarray(self.start_position, dtype=float)
        target = np.asarray(self.target_position, dtype=float)
        
        # Calculate distance for time normalization
        dist = np.linalg.norm(target - start)
        
        if dist == 0.0:
            return [tuple(start.tolist())]
        
        # Parameters for the dynamical system
        # Damping coefficient (prevents oscillation, ensures smooth approach)
        damping = 1.5
        
        # Attraction strength (aggressiveness)
        alpha = self.trajectory_aggressiveness
        
        # Estimate time to reach target (used for integration duration)
        if self.speed is not None and self.speed > 0.0:
            # Time based on average speed
            estimated_time = dist / self.speed * 1.5  # Add 50% buffer for curved path
        else:
            # Fallback: estimate based on steps
            estimated_time = self.steps
        
        # Define the ODE system: state = [x, y, vx, vy]
        def dynamics(t, state):
            x, y, vx, vy = state
            pos = np.array([x, y])
            vel = np.array([vx, vy])
            
            # Force towards target
            direction = target - pos
            force = alpha * direction
            
            # Damping force
            damping_force = -damping * vel
            
            # Total acceleration
            acc = force + damping_force
            
            # Return derivatives: [dx/dt, dy/dt, dvx/dt, dvy/dt]
            return [vx, vy, acc[0], acc[1]]
        
        # Initial conditions: position at start, zero velocity
        y0 = [start[0], start[1], 0.0, 0.0]
        
        # Time span for integration
        t_span = (0, estimated_time)
        
        # Solve ODE
        # Use dense_output for smooth interpolation at any time
        sol = solve_ivp(
            dynamics, 
            t_span, 
            y0, 
            method='RK45',  # Runge-Kutta 4/5 method
            dense_output=True,
            max_step=1.0,  # Maximum step size of 1 second
        )
        
        # Sample trajectory at 1-second intervals
        if self.speed is not None and self.speed > 0.0:
            # Sample at 1-second intervals
            t_samples = np.arange(0, estimated_time, 1.0)
            # Always include final time
            if t_samples[-1] < estimated_time:
                t_samples = np.append(t_samples, estimated_time)
        else:
            # Use specified number of steps
            t_samples = np.linspace(0, estimated_time, self.steps)
        
        # Evaluate solution at sample times
        trajectory_states = sol.sol(t_samples)
        
        # Extract positions (first 2 components of state)
        trajectory = []
        for i in range(len(t_samples)):
            pos = trajectory_states[:2, i]
            
            # Add stochastic noise if requested (SDE component)
            if self.noise_std > 0.0 and i > 0 and i < len(t_samples) - 1:
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=2)
                pos = pos + noise
            
            trajectory.append(tuple(pos.tolist()))
        
        # Ensure trajectory starts at exact start position
        trajectory[0] = tuple(start.tolist())
        
        # Ensure trajectory ends at exact target position
        trajectory[-1] = tuple(target.tolist())
        
        return trajectory

    def _generate_dynamic_trajectory_with_waypoints(self):
        """
        Generate a multi-segment dynamic trajectory through all waypoints using ODE/SDE.
        
        Path: start -> waypoint1 -> waypoint2 -> ... -> target
        
        Each segment uses the ODE/SDE system with:
        - Current segment endpoint as temporary target
        - Initial velocity from previous segment for continuity
        - Same aggressiveness and noise parameters
        """
        # Build the complete path: start -> waypoints -> target
        path_points = [np.asarray(self.start_position, dtype=float)]
        
        # Add all waypoints
        for wp in self.waypoints:
            path_points.append(np.asarray(wp, dtype=float))
        
        # Add final target
        path_points.append(np.asarray(self.target_position, dtype=float))
        
        # Generate dynamic trajectory through all segments
        combined_trajectory = []
        current_velocity = np.array([0.0, 0.0])  # Start with zero velocity
        
        for segment_idx in range(len(path_points) - 1):
            segment_start = path_points[segment_idx]
            segment_end = path_points[segment_idx + 1]
            
            # Generate dynamic segment with velocity continuity
            segment_traj, final_velocity = self._generate_dynamic_segment(
                segment_start, 
                segment_end, 
                initial_velocity=current_velocity,
                include_endpoint=True
            )
            
            # Add segment to combined trajectory
            if segment_idx == 0:
                # First segment: include all points
                combined_trajectory.extend(segment_traj)
            else:
                # Later segments: skip first point (already included from previous segment)
                combined_trajectory.extend(segment_traj[1:])
            
            # Update velocity for next segment
            current_velocity = final_velocity
        
        return combined_trajectory

    def _generate_dynamic_segment(self, start_pos, end_pos, initial_velocity=None, include_endpoint=True):
        """
        Generate a single dynamic trajectory segment using ODE/SDE system.
        
        Args:
            start_pos: Starting position as numpy array
            end_pos: Ending position as numpy array
            initial_velocity: Initial velocity for continuity (None = zero velocity)
            include_endpoint: Whether to include the exact endpoint
            
        Returns:
            tuple: (trajectory_points, final_velocity)
                - trajectory_points: List of (x, y) tuples
                - final_velocity: Final velocity as numpy array for next segment
        """
        start = np.asarray(start_pos, dtype=float)
        target = np.asarray(end_pos, dtype=float)
        
        # Calculate distance
        dist = np.linalg.norm(target - start)
        
        if dist == 0.0:
            return [tuple(start.tolist())], np.array([0.0, 0.0])
        
        # Initial velocity (for continuity between segments)
        if initial_velocity is None:
            initial_velocity = np.array([0.0, 0.0])
        
        # Parameters for the dynamical system
        damping = 1.5
        alpha = self.trajectory_aggressiveness
        
        # Estimate time to reach target
        if self.speed is not None and self.speed > 0.0:
            estimated_time = dist / self.speed * 1.5
        else:
            # For multi-segment, divide steps proportionally
            estimated_time = self.steps * 0.5  # Conservative estimate per segment
        
        # Define the ODE system: state = [x, y, vx, vy]
        def dynamics(t, state):
            x, y, vx, vy = state
            pos = np.array([x, y])
            vel = np.array([vx, vy])
            
            # Force towards segment target
            direction = target - pos
            force = alpha * direction
            
            # Damping force
            damping_force = -damping * vel
            
            # Total acceleration
            acc = force + damping_force
            
            return [vx, vy, acc[0], acc[1]]
        
        # Initial conditions: position at start, velocity from previous segment
        y0 = [start[0], start[1], initial_velocity[0], initial_velocity[1]]
        
        # Time span for integration
        t_span = (0, estimated_time)
        
        # Solve ODE
        sol = solve_ivp(
            dynamics, 
            t_span, 
            y0, 
            method='RK45',
            dense_output=True,
            max_step=1.0,
        )
        
        # Sample trajectory
        if self.speed is not None and self.speed > 0.0:
            t_samples = np.arange(0, estimated_time, 1.0)
            if t_samples[-1] < estimated_time:
                t_samples = np.append(t_samples, estimated_time)
        else:
            t_samples = np.linspace(0, estimated_time, max(10, self.steps // 3))
        
        # Evaluate solution
        trajectory_states = sol.sol(t_samples)
        
        # Extract positions and velocities
        trajectory = []
        final_vel = np.array([0.0, 0.0])
        
        for i in range(len(t_samples)):
            pos = trajectory_states[:2, i]
            vel = trajectory_states[2:4, i]
            
            # Add stochastic noise if requested
            if self.noise_std > 0.0 and i > 0 and i < len(t_samples) - 1:
                noise = np.random.normal(loc=0.0, scale=self.noise_std, size=2)
                pos = pos + noise
            
            trajectory.append(tuple(pos.tolist()))
            
            # Store final velocity
            if i == len(t_samples) - 1:
                final_vel = vel.copy()
        
        # Ensure segment starts at exact start position
        trajectory[0] = tuple(start.tolist())
        
        # Ensure segment ends at exact endpoint if requested
        if include_endpoint:
            trajectory[-1] = tuple(target.tolist())
        
        return trajectory, final_vel

    def _generate_trajectory_with_waypoints(self):
        """
        Generate a multi-segment trajectory through all waypoints.
        
        Path: start -> waypoint1 -> waypoint2 -> ... -> target
        
        Uses the same logic as generate_trajectory but applies it to each segment.
        Each waypoint is included exactly once at the junction between segments.
        """
        # Build the complete path: start -> waypoints -> target
        path_points = [np.asarray(self.start_position, dtype=float)]
        
        # Add all waypoints
        for wp in self.waypoints:
            path_points.append(np.asarray(wp, dtype=float))
        
        # Add final target
        path_points.append(np.asarray(self.target_position, dtype=float))
        
        # Generate trajectory through all segments
        combined_trajectory = []
        
        for i in range(len(path_points) - 1):
            segment_start = path_points[i]
            segment_end = path_points[i + 1]
            
            # Generate segment
            segment = self._generate_segment(segment_start, segment_end, include_endpoint=True)
            
            # Add segment to combined trajectory
            if i == 0:
                # First segment: include all points
                combined_trajectory.extend(segment)
            else:
                # Later segments: skip first point (already included from previous segment)
                combined_trajectory.extend(segment[1:])
        
        return combined_trajectory
    
    def _generate_segment(self, start_pos, end_pos, include_endpoint=True):
        """
        Generate trajectory segment between two points using current speed/noise settings.
        
        Args:
            start_pos: Starting position as numpy array
            end_pos: Ending position as numpy array
            include_endpoint: Whether to include the exact endpoint
            
        Returns:
            List of trajectory points (tuples)
        """
        start = np.asarray(start_pos, dtype=float)
        end = np.asarray(end_pos, dtype=float)
        diff = end - start
        dist = np.linalg.norm(diff)
        
        # If start equals end, return just the point
        if dist == 0.0:
            return [tuple(start.tolist())]
        
        # Speed-based trajectory generation
        if self.speed is not None and self.speed > 0.0:
            pos = start.copy()
            remaining = dist
            trajectory = [tuple(pos.tolist())]
            
            # Safety cap
            nominal_steps = max(1, int(np.ceil(dist / max(1e-8, self.speed))))
            max_steps = max(1000, nominal_steps * 10)
            
            steps_taken = 0
            while remaining > 1e-8 and steps_taken < max_steps:
                # Sample effective speed for this 1-second interval
                if self.speed_noise > 0.0:
                    eff_speed = float(np.random.normal(loc=self.speed, scale=self.speed_noise))
                else:
                    eff_speed = float(self.speed)
                
                # Clamp to non-negative
                if eff_speed <= 0.0:
                    eff_speed = 0.0
                
                move_dist = eff_speed * 1.0  # dt = 1s
                
                if move_dist <= 0.0:
                    break
                
                # Move towards end
                travel = min(move_dist, remaining)
                direction = diff / (np.linalg.norm(diff) + 1e-12)
                pos = pos + direction * travel
                remaining = np.linalg.norm(end - pos)
                trajectory.append(tuple(pos.tolist()))
                
                # Update diff for direction recalculation
                diff = end - pos
                steps_taken += 1
            
            # Ensure final point is the exact endpoint if requested
            if include_endpoint and tuple(trajectory[-1]) != tuple(end.tolist()):
                trajectory.append(tuple(end.tolist()))
            
            # Add Gaussian noise to intermediate points (not start/end)
            if self.noise_std > 0.0:
                for idx in range(1, len(trajectory) - 1):
                    pos_arr = np.array(trajectory[idx], dtype=float)
                    noise = np.random.normal(loc=0.0, scale=self.noise_std, size=pos_arr.shape)
                    pos_arr = pos_arr + noise
                    trajectory[idx] = tuple(pos_arr.tolist())
            
            return trajectory
        
        # Legacy fallback: evenly spaced samples using self.steps
        # Divide steps proportionally to segment length (for multi-segment paths)
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
    - When only a single `target_position` is supplied and `spread > 0`, each attacker
      samples an individual target from the same uniform distribution used for start
      positions. Set `spread=0` to disable both start and target jitter.
    - Supports waypoints: single waypoint or ordered list of waypoints for all attackers
    """

    def __init__(
        self,
        start_position,
        target_positions = None,
        number_of_attackers: int = 1,
        spread: float = 0.0,
        target_distribution = None,
        noise_std: float = 0.0,
        waypoint = None,
        waypoints = None,
        use_dynamic_trajectory: bool = False,
        trajectory_aggressiveness: float = 2.0,
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
        
        # Support both single waypoint and list of waypoints
        if waypoints is not None:
            self.waypoints = waypoints
        elif waypoint is not None:
            self.waypoints = [waypoint]
        else:
            self.waypoints = None
        
        # Dynamic trajectory settings
        self.use_dynamic_trajectory = use_dynamic_trajectory
        self.trajectory_aggressiveness = trajectory_aggressiveness

    def _sample_uniform_offset(self, base_point):
        """
        Sample a point uniformly within [-spread, +spread] of `base_point` in
        each dimension. If spread <= 0, returns the base point unchanged.
        """
        if self.spread <= 0:
            return tuple(base_point)

        return tuple(
            float(coord) + random.uniform(-self.spread, self.spread)
            for coord in base_point
        )

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

        return self._sample_uniform_offset(self.start_position)

    def _target_for_index(self, idx: int):
        """
        Choose a target for the attacker with index `idx`.

        Multiple targets are distributed evenly using a round‑robin scheme.
        If using target_distribution, returns None (will be sampled in Attacker.__init__).
        """
        if self.target_positions is None:
            return None
        base_target = self.target_positions[idx % len(self.target_positions)]
        if len(self.target_positions) == 1 and self.spread > 0:
            return self._sample_uniform_offset(base_target)
        return base_target

    def generate_swarm(self, steps: int, speed: float = 0.0, speed_noise: float = 0.0) -> List[Attacker]:
        """
        Generate and return a list of `Attacker` instances.

        Each attacker:
        - Starts at a position sampled around `start_position` (controlled by `spread`).
        - Is assigned a target chosen from `target_positions` (round-robin), or 
          sampled from `target_distribution` if provided.
        - Optionally flies through waypoint(s) if specified.
        - Uses dynamic trajectory (ODE/SDE-based) if use_dynamic_trajectory=True.
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
                waypoints=self.waypoints,
                use_dynamic_trajectory=self.use_dynamic_trajectory,
                trajectory_aggressiveness=self.trajectory_aggressiveness,
            ))

        return swarm


if __name__ == "__main__":
    attacker_swarm = AttackerSwarm(start_position=(0, 0), target_positions=[(1, 1), (2, 2)], number_of_attackers=2, spread=0.1)
    swarm = attacker_swarm.generate_swarm(steps=10)
    for attacker in swarm:
        print(attacker.start_position)
        print(attacker.target_position)
        print(attacker.trajectory)
