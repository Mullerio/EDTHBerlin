import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.detectors import Detector
#from attackers import Attacker 
from abc import ABC, abstractmethod
from src.targets import *


class Environment:
    def __init__(self, width, height, target: TargetDistribution = None, cell_size=1.0, attackers=None): 
        """
        Initialize environment.
        
        Args:
            width: number of grid cells in x direction (integer)
            height: number of grid cells in y direction (integer)
            target: target probability distribution (operates in physical coordinates) - OPTIONAL
                   If not provided, attackers must specify their own targets
            cell_size: physical size of each grid cell in meters (default 1.0)
                      Physical coordinates range: x ∈ [0, width*cell_size), y ∈ [0, height*cell_size)
            attackers: optional list of Attacker objects to add at initialization
        
        The grid is only used for:
        - Selecting non-observable regions (via mask from UI)
        - Checking if a point is observable (grid-based lookup)
        - Optional visualization (prob_map heatmap)
        
        All detection, trajectory, and probability computations operate on continuous
        physical coordinates (meters) independent of grid resolution.
        """
        self.width = width  # number of cells
        self.height = height  # number of cells
        self.cell_size = cell_size  # meters per cell
        
        self.atk_drones = []
        self.def_drones = []
        self.detectors = []
        
        self.target = target  # Optional - can be None if attackers have individual targets
        
        # Grid for visualization only - create sampling points in physical coordinates
        x = np.linspace(0, self.width * self.cell_size, self.width)
        y = np.linspace(0, self.height * self.cell_size, self.height)
        xx, yy = np.meshgrid(x, y)
        
        self.grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Probability map for visualization only (not used in detection logic)
        self.prob_map = self._generate_prob_map() if target is not None else np.ones((self.height, self.width))
        
        # Observable mask: grid-based boolean array for UI region selection
        # mask[j, i] corresponds to physical region [i*cell_size, (i+1)*cell_size) x [j*cell_size, (j+1)*cell_size)
        self.observable_mask = np.ones((self.height, self.width), dtype=bool)
        
        # Add attackers if provided at initialization
        if attackers is not None:
            for attacker in attackers:
                self.add_atk_drone(attacker)
        
    def set_subgrid_unobservable(self, x0, y0, w, h):
        """
        Mark a rectangular subgrid as unobservable.
        
        Args:
            x0, y0: bottom-left corner in grid cell indices (integer)
            w, h: width and height in grid cells (integer)
        
        This sets observable_mask[y0:y0+h, x0:x0+w] = False (clamped to grid).
        Note: x0, y0, w, h are in grid cell units, not physical meters.
        """
        x0i = max(0, int(round(x0)))
        y0i = max(0, int(round(y0)))
        x1i = min(self.width, int(round(x0 + w)))
        y1i = min(self.height, int(round(y0 + h)))
        self.observable_mask[y0i:y1i, x0i:x1i] = False
    
    def _generate_prob_map(self) -> np.ndarray:
        """
        Generate probability map over the grid based on the target distribution.
        Returns: shape [width, height]
        
        If no target distribution is provided, returns uniform probability map.
        """
        if self.target is None:
            # No target distribution - return uniform map
            return np.ones((self.height, self.width)) / (self.height * self.width)
        
        # Evaluate log density at all grid points
        log_probs = self.target.log_prob(self.grid)  #[width*height, 1]
        
        # Convert log densities to probabilities
        densities = np.exp(log_probs.squeeze())  #[width*height,]
        
        # Reshape back to grid
        prob_map = densities.reshape(self.height, self.width)
        
        prob_map = prob_map / np.sum(prob_map)
        
        return prob_map
    
    def add_atk_drone(self, drone : Detector):
        self.atk_drones.append(drone)
    
    def add_def_drone(self, drone):
        self.def_drones.append(drone)
        
    def add_detector(self, detector : Detector):
        self.detectors.append(detector) 
        
    def visualize(self, figsize=(6, 6), ax=None, show=True):
        """
        Visualize the target probability heatmap over the environment grid and
        plot drones as colored dots.

        - attack drones: red
        - defensive drones: green
        - detectors: blue

        All positions are in physical coordinates (meters).
        The heatmap extent matches the physical dimensions: [0, width*cell_size) x [0, height*cell_size).
        """
        # Ensure prob_map has shape (height, width)
        prob = self.prob_map

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Show heatmap; extent in physical coordinates (meters)
        physical_width = self.width * self.cell_size
        physical_height = self.height * self.cell_size
        im = ax.imshow(prob, origin='lower', extent=(0, physical_width, 0, physical_height), cmap='hot', aspect='auto')

        def _plot_list(drone_list, color, label):
            if len(drone_list) == 0:
                return
            pts = np.array([d.position for d in drone_list])
            if pts.size == 0:
                return
            pts = pts.reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=label, edgecolors='k')

        _plot_list(self.atk_drones, 'red', 'attack')
        _plot_list(self.def_drones, 'green', 'defend')
        _plot_list(self.detectors, 'gold', 'detect')

        # Overlay detector probability distributions using the main environment grid
        # This avoids rectangular local-grid artifacts by computing an RGBA image
        # on the same grid as the heatmap and setting alpha=0 outside the detector radius.
        for det in self.detectors:
            try:
                cx, cy = det.position
            except Exception:
                continue

            # Detector radius if available (may be None for non-radial detectors)
            r = getattr(det, 'radius', None)

            # Flattened grid coordinates
            gx = self.grid[:, 0]
            gy = self.grid[:, 1]
            pts = np.stack([gx, gy], axis=1)  # shape (H*W, 2)
            dists_flat = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)

            # Support detectors that accept either (N,2) point arrays or distance arrays.
            probs_flat = None
            try:
                # Try position-aware probability first
                probs_flat = det.probability(pts)
                probs_flat = np.asarray(probs_flat, dtype=float).reshape(-1)
                if probs_flat.shape[0] != pts.shape[0]:
                    raise ValueError('probability returned wrong shape')
            except Exception:
                try:
                    probs_flat = det.probability(dists_flat)
                    probs_flat = np.asarray(probs_flat, dtype=float).reshape(-1)
                except Exception:
                    # Cannot evaluate this detector; skip
                    continue

            # Alpha mask: if radius provided, zero outside radius; otherwise keep alpha everywhere
            alpha_flat = probs_flat.copy()
            if r is not None:
                alpha_flat[dists_flat > r] = 0.0

            # Map probabilities to RGBA using a colormap and apply alpha mask
            cmap = plt.cm.Blues
            colors = cmap(np.clip(probs_flat, 0.0, 1.0))  # (N,4)
            colors[:, 3] = colors[:, 3] * alpha_flat

            # Reshape to image and overlay on the same extent as the heatmap
            rgba = colors.reshape(self.height, self.width, 4)
            ax.imshow(rgba, origin='lower', extent=(0, physical_width, 0, physical_height), zorder=5)

            # Draw detector radius circle with yellow fill (in physical coordinates)
            if r is not None:
                circ = Circle((cx, cy), r, edgecolor='orange', facecolor='yellow', alpha=0.3, linewidth=2)
                ax.add_patch(circ)

        ax.set_xlim(0, physical_width)
        ax.set_ylim(0, physical_height)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')

        if show and created_fig:
            plt.show()

        return ax

    def visualize_trajectories(self, figsize=(6, 6), ax=None, show=True, show_heatmap=True):
        """
        Visualize attacker trajectories as lines from start to target positions.
        
        - Shows the full path each attacker will take
        - Optionally displays the target probability heatmap in the background
        - Marks start positions with 'o' and target positions with 'x'
        
        All coordinates are in physical units (meters).
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        physical_width = self.width * self.cell_size
        physical_height = self.height * self.cell_size

        # Optionally show heatmap in background
        if show_heatmap:
            prob = self.prob_map
            im = ax.imshow(prob, origin='lower', extent=(0, physical_width, 0, physical_height), 
                          cmap='hot', aspect='auto', alpha=0.6)

        # Plot trajectories for attackers that have trajectory attribute
        for attacker in self.atk_drones:
            if hasattr(attacker, 'trajectory') and attacker.trajectory:
                traj = np.array(attacker.trajectory)
                ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.5, linewidth=1)
                # Mark start and end
                ax.plot(traj[0, 0], traj[0, 1], 'ro', markersize=6, label='Start' if attacker == self.atk_drones[0] else '')
                ax.plot(traj[-1, 0], traj[-1, 1], 'rx', markersize=8, label='Target' if attacker == self.atk_drones[0] else '')

        # Plot defensive drones
        def _plot_list(drone_list, color, label):
            if len(drone_list) == 0:
                return
            pts = np.array([d.position for d in drone_list])
            if pts.size == 0:
                return
            pts = pts.reshape(-1, 2)
            ax.scatter(pts[:, 0], pts[:, 1], c=color, label=label, edgecolors='k', s=100)

        _plot_list(self.def_drones, 'green', 'defend')
        _plot_list(self.detectors, 'gold', 'detect')

        for det in self.detectors:
            try:
                cx, cy = det.position
            except Exception:
                continue

            r = getattr(det, 'radius', None)

            gx = self.grid[:, 0]
            gy = self.grid[:, 1]
            pts = np.stack([gx, gy], axis=1)
            dists_flat = np.sqrt((gx - cx) ** 2 + (gy - cy) ** 2)

            probs_flat = None
            try:
                probs_flat = det.probability(pts)
                probs_flat = np.asarray(probs_flat, dtype=float).reshape(-1)
                if probs_flat.shape[0] != pts.shape[0]:
                    raise ValueError('probability returned wrong shape')
            except Exception:
                try:
                    probs_flat = det.probability(dists_flat)
                    probs_flat = np.asarray(probs_flat, dtype=float).reshape(-1)
                except Exception:
                    continue

            alpha_flat = probs_flat.copy()
            if r is not None:
                alpha_flat[dists_flat > r] = 0.0

            cmap = plt.cm.Blues
            colors = cmap(np.clip(probs_flat, 0.0, 1.0))
            colors[:, 3] = colors[:, 3] * alpha_flat
            rgba = colors.reshape(self.height, self.width, 4)
            ax.imshow(rgba, origin='lower', extent=(0, physical_width, 0, physical_height), zorder=5)
            if r is not None:
                circ = Circle((cx, cy), r, edgecolor='orange', facecolor='yellow', alpha=0.3, linewidth=2)
                ax.add_patch(circ)

        ax.set_xlim(0, physical_width)
        ax.set_ylim(0, physical_height)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Attacker Trajectories')
        
        # Remove duplicate labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')

        if show and created_fig:
            plt.show()

        return ax

    def step(self, dt):
        raise NotImplementedError



class SectorEnv(Environment):
    """
    Environment for sector-based detection analysis.
    
    The grid is divided into observable and non-observable regions.
    - Observable region: fully detectable (no detector drones needed, detection probability = 1.0)
    - Non-observable region: requires detector drones with their respective detection distributions
    
    The grid mask is ONLY used for:
    - UI region selection (click to mark cells observable/non-observable)
    - Checking if a continuous point (x,y) in meters falls in observable region
    
    All detection probabilities, trajectories, and statistics are computed continuously
    in physical coordinates (meters) and do NOT depend on grid resolution.
    
    Use cases:
    1. Calculate how long an attacker can fly through non-observable region before detection
    2. Compute cumulative detection probability along a trajectory
    3. Place detectors strategically in non-observable zones
    """
    
    def __init__(self, width, height, target: TargetDistribution = None, 
                 default_observable=True, cell_size=1.0, attackers=None):
        """
        Initialize sector environment.
        
        Args:
            width: number of grid cells in x direction (integer)
            height: number of grid cells in y direction (integer)
            target: target probability distribution (operates in physical meters) - OPTIONAL
                   If not provided, attackers must specify their own targets
            default_observable: if True, entire grid starts as observable (then mark subgrids as non-observable)
                               if False, entire grid starts as non-observable (then mark subgrids as observable)
            cell_size: physical size of each grid cell in meters (default 1.0)
            attackers: optional list of Attacker objects to add at initialization
        """
        super().__init__(width, height, target, cell_size=cell_size, attackers=attackers)
        
        # Override observable mask based on default
        if default_observable:
            self.observable_mask = np.ones((self.height, self.width), dtype=bool)
        else:
            self.observable_mask = np.zeros((self.height, self.width), dtype=bool)
    
    def set_rectangular_sector(self, x0, y0, w, h, observable):
        """
        Set a rectangular sector's observability.
        
        Args:
            x0, y0: bottom-left corner in grid cell indices (integer)
            w, h: width and height in grid cells (integer)
            observable: True to make region observable, False for non-observable
        
        Note: Coordinates are in grid cell units, not physical meters.
        To convert from meters: grid_x = int(physical_x / cell_size)
        """
        x0i = max(0, int(round(x0)))
        y0i = max(0, int(round(y0)))
        x1i = min(self.width, int(round(x0 + w)))
        y1i = min(self.height, int(round(y0 + h)))
        self.observable_mask[y0i:y1i, x0i:x1i] = observable
    
    def set_circular_sector(self, cx, cy, radius, observable):
        """
        Set a circular sector's observability.
        
        Args:
            cx, cy: center in grid cell indices (integer)
            radius: circle radius in grid cells (integer/float)
            observable: True to make region observable, False for non-observable
        
        Note: Coordinates are in grid cell units, not physical meters.
        To convert from meters: grid_x = int(physical_x / cell_size)
        """
        for i in range(self.height):
            for j in range(self.width):
                dist = np.sqrt((j - cx)**2 + (i - cy)**2)
                if dist <= radius:
                    self.observable_mask[i, j] = observable
    
    def is_observable_at(self, x, y):
        """
        Check if a point (x, y) in physical coordinates (meters) is in an observable region.
        
        Args:
            x, y: physical coordinates in meters
        
        Returns:
            bool: True if observable (fully detectable without drones), False otherwise
        
        Maps continuous physical coordinates to grid indices and checks the observable_mask.
        This is the ONLY place where grid resolution affects detection logic (observability boundary).
        All probability computations are continuous and independent of grid.
        """
        # Convert physical coordinates to grid indices
        xi = int(np.floor(x / self.cell_size))
        yi = int(np.floor(y / self.cell_size))
        
        # Clamp to valid grid indices
        xi = np.clip(xi, 0, self.width - 1)
        yi = np.clip(yi, 0, self.height - 1)
        
        return self.observable_mask[yi, xi]
    
    def get_detection_probability_at_point(self, x, y):
        """
        Calculate detection probability at a single point (x, y).
        
        Returns:
            float: detection probability in [0, 1]
                  - 1.0 if point is in observable region
                  - Combined detector probability if in non-observable region
        """
        # If in observable region, detection is certain
        if self.is_observable_at(x, y):
            return 1.0
        
        # In non-observable region, compute detection from detector drones
        if len(self.detectors) == 0:
            return 0.0
        
        # Combine detector probabilities (assuming independent detectors)
        # P(detected) = 1 - P(not detected by any) = 1 - ∏(1 - P_i)
        prob_not_detected = 1.0
        
        for det in self.detectors:
            try:
                cx, cy = det.position
                dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                
                # Try to get detection probability
                try:
                    p_det = det.probability(np.array([[x, y]]))
                    if hasattr(p_det, '__len__'):
                        p_det = float(p_det[0])
                    else:
                        p_det = float(p_det)
                except Exception:
                    try:
                        p_det = det.probability(dist)
                        p_det = float(p_det)
                    except Exception:
                        continue
                
                # Combine probabilities
                prob_not_detected *= (1.0 - p_det)
            except Exception:
                continue
        
        return 1.0 - prob_not_detected
    
    def compute_sliding_window_detection(self, trajectory, time_per_step=1.0, window_sizes=[5, 10, 30, 60], only_nonobservable=True, include_observable_in_stats=False):
        """
        Compute TRUE detection probability over sliding time windows.

        Args:
            trajectory: list of (x, y) waypoints
            time_per_step: seconds per waypoint
            window_sizes: list of window lengths (seconds) to evaluate
            only_nonobservable: if True, only calculate detection for non-observable region
            include_observable_in_stats: if True, treat observable regions as 100% detection

        Returns:
            dict:
                'windows': {
                    'Xs': list of {
                        'start_time': ...,
                        'end_time': ...,
                        'prob_detected': ...,
                        'window_index': ...
                    }
                }
                'summary': {
                    'Xs': {
                        'max': ...,
                        'min': ...,
                        'mean': ...
                    }
                }
        """
        if not trajectory:
            return {'windows': {}, 'summary': {}}

        # Compute instantaneous detection probabilities directly (avoid recursion)
        instant_probs = []
        times = []
        
        for i, (x, y) in enumerate(trajectory):
            t = i * time_per_step
            times.append(t)
            
            # Check if in observable region
            is_observable = self.is_observable_at(x, y)
            
            if only_nonobservable and is_observable:
                # In observable region
                if include_observable_in_stats:
                    instant_probs.append(1.0)  # Observable = 100% detection
                else:
                    instant_probs.append(0.0)  # Exclude observable from stats
            else:
                # In non-observable region OR not filtering by observability
                # Calculate detection probability from all detectors at this point
                p_det = 0.0
                if len(self.detectors) > 0:
                    prob_not_detected = 1.0
                    for det in self.detectors:
                        try:
                            cx, cy = det.position
                            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                            
                            try:
                                p = det.probability(np.array([[x, y]]))
                                if hasattr(p, '__len__'):
                                    p = float(p[0])
                                else:
                                    p = float(p)
                            except Exception:
                                try:
                                    p = det.probability(dist)
                                    p = float(p)
                                except Exception:
                                    continue
                            
                            prob_not_detected *= (1.0 - p)
                        except Exception:
                            continue
                    
                    p_det = 1.0 - prob_not_detected
                
                instant_probs.append(p_det)
        
        N = len(instant_probs)

        results = {'windows': {}, 'summary': {}}

        for W in window_sizes:
            steps_per_window = int(W / time_per_step)
            if steps_per_window < 1:
                continue

            window_results = []
            probs = []

            for start in range(0, N - steps_per_window + 1):
                end = start + steps_per_window

                window_instants = instant_probs[start:end]
                
                # When excluding observable regions, skip windows that contain observable points (0.0)
                # unless we're including them as 1.0
                if only_nonobservable and not include_observable_in_stats:
                    # Check if window contains any observable points (marked as 0.0)
                    # Skip this window if it spans observable regions
                    if any(p == 0.0 and self.is_observable_at(*trajectory[start + idx]) 
                           for idx, p in enumerate(window_instants)):
                        continue

                # REAL probability: 1 − Π (1 − p_i)
                prob_not = 1.0
                for p in window_instants:
                    prob_not *= (1 - p)

                prob_detected = 1 - prob_not

                window_results.append({
                    'window_index': start,
                    'start_time': times[start],
                    'end_time': times[end-1],
                    'prob_detected': prob_detected,
                })

                probs.append(prob_detected)

            results['windows'][f"{W}s"] = window_results

            if probs:
                results['summary'][f"{W}s"] = {
                    'max': max(probs),
                    'min': min(probs),
                    'mean': sum(probs) / len(probs)
                }
            else:
                results['summary'][f"{W}s"] = {'max': 0, 'min': 0, 'mean': 0}

        return results
    
    def analyze_trajectory(self, trajectory, time_per_step=1.0, sliding_window_sizes=[5, 10, 15, 20, 30], only_nonobservable=True, include_observable_in_stats=False):
        """
        Analyze detection probability along an attacker trajectory.
        
        Args:
            trajectory: list of (x, y) waypoints
            time_per_step: time elapsed between consecutive waypoints (seconds)
            sliding_window_sizes: list of window sizes (in seconds) for sliding window analysis
            only_nonobservable: if True, only calculate detection for non-observable region
                              (ignores observable regions where detection is automatic)
            include_observable_in_stats: if True, include observable regions in probability calculations
                                         (treats observable as 100% detection, includes in averages/intervals)
                                         Only effective when only_nonobservable=True
        
        Returns:
            dict with keys:
                - 'cumulative_detection_prob': probability of being detected at least once in non-observable region
                - 'point_probs': detection probability at each waypoint (from detectors only)
                - 'time_to_detection': time segments and detection probabilities in non-observable region
                - 'time_in_observable': total time spent in observable region
                - 'time_in_nonobservable': total time spent in non-observable region
                - 'first_detection_time': time when cumulative prob exceeds threshold (if any)
                - 'avg_detection_per_second': average detection probability per second in non-observable region
                - 'detection_rate_per_interval': detection probability per time interval (e.g., per 10s)
        """
        if not trajectory or len(trajectory) == 0:
            return {
                'cumulative_detection_prob': 0.0,
                'point_probs': [],
                'time_to_detection': [],
                'time_in_observable': 0.0,
                'time_in_nonobservable': 0.0,
                'first_detection_time': None,
                'avg_detection_per_second': 0.0,
                'detection_rate_per_interval': {}
            }
        
        point_probs = []
        time_in_obs = 0.0
        time_in_nonobs = 0.0
        
        # Calculate detection probability at each waypoint
        # Only from detectors (not automatic detection in observable regions)
        for waypoint in trajectory:
            x, y = waypoint
            
            # Track time in each region type
            if self.is_observable_at(x, y):
                time_in_obs += time_per_step
                if only_nonobservable:
                    # In observable region, don't count towards detection probability
                    # (we know they're detected anyway, but we care about detector performance)
                    point_probs.append(0.0)
                else:
                    point_probs.append(1.0)
            else:
                time_in_nonobs += time_per_step
                # In non-observable region, get detection probability from detectors
                p_det = 0.0
                if len(self.detectors) > 0:
                    prob_not_detected = 1.0
                    for det in self.detectors:
                        try:
                            cx, cy = det.position
                            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                            
                            try:
                                p = det.probability(np.array([[x, y]]))
                                if hasattr(p, '__len__'):
                                    p = float(p[0])
                                else:
                                    p = float(p)
                            except Exception:
                                try:
                                    p = det.probability(dist)
                                    p = float(p)
                                except Exception:
                                    continue
                            
                            prob_not_detected *= (1.0 - p)
                        except Exception:
                            continue
                    
                    p_det = 1.0 - prob_not_detected
                
                point_probs.append(p_det)
        
        # Calculate cumulative detection probability (only in non-observable region)
        # P(detected by step i) = 1 - ∏(1 - p_j) for j in non-observable region
        cumulative_prob_not_detected = 1.0
        cumulative_probs = []
        time_to_detection = []
        first_detection_time = None
        nonobs_steps = 0
        
        for i, p in enumerate(point_probs):
            x, y = trajectory[i]
            in_nonobs = not self.is_observable_at(x, y)
            
            if in_nonobs or not only_nonobservable:
                cumulative_prob_not_detected *= (1.0 - p)
                if in_nonobs:
                    nonobs_steps += 1
            
            cumulative_detection = 1.0 - cumulative_prob_not_detected
            cumulative_probs.append(cumulative_detection)
            
            time_to_detection.append({
                'time': i * time_per_step,
                'point': trajectory[i],
                'instant_prob': p,
                'cumulative_prob': cumulative_detection,
                'in_nonobservable': in_nonobs
            })
            
            # Check if detection becomes likely (e.g., > 50%)
            if first_detection_time is None and cumulative_detection > 0.5:
                first_detection_time = i * time_per_step
        
        # Calculate average detection probability per second
        # This should be the AVERAGE instantaneous probability, not cumulative sum / time
        if include_observable_in_stats and only_nonobservable:
            # Include observable regions (as 100% detection) in the calculation
            # For non-observable points: use actual detector probabilities
            # For observable points: treat as 1.0 detection probability per second
            nonobs_probs = [p for i, p in enumerate(point_probs) 
                           if not self.is_observable_at(*trajectory[i])]
            # Average of detector probs in non-obs + 1.0 in obs, weighted by time
            if time_in_nonobs > 0:
                avg_nonobs = sum(nonobs_probs) / len(nonobs_probs) if nonobs_probs else 0.0
            else:
                avg_nonobs = 0.0
            
            total_time = time_in_obs + time_in_nonobs
            # Weighted average: (non_obs_time * avg_nonobs + obs_time * 1.0) / total_time
            avg_detection_per_second = ((time_in_nonobs * avg_nonobs + time_in_obs * 1.0) / total_time 
                                       if total_time > 0 else 0.0)
            time_for_intervals = total_time
        else:
            # Only consider non-observable region
            nonobs_probs = [p for i, p in enumerate(point_probs) 
                           if not self.is_observable_at(*trajectory[i])]
            # Average instantaneous probability per second
            avg_detection_per_second = (sum(nonobs_probs) / len(nonobs_probs) 
                                       if nonobs_probs else 0.0)
            time_for_intervals = time_in_nonobs
        
        # Calculate detection rate per various time intervals
        # Using average detection probability raised to the power of the interval
        detection_rate_per_interval = {}
        for interval in [1, 5, 10, 30, 60]:  # seconds
            if time_for_intervals == 0:
                detection_rate_per_interval[f'{interval}s'] = 0.0
                continue
            
            # Use average detection per second
            # P(not detected in interval) = (1 - avg_prob)^interval
            # P(detected in interval) = 1 - (1 - avg_prob)^interval
            prob_not_detected_interval = (1.0 - avg_detection_per_second) ** interval
            detection_rate_per_interval[f'{interval}s'] = 1.0 - prob_not_detected_interval
        
        # Calculate maximum undetected gap statistics
        # This answers: "What's the probability we have NO gap longer than X seconds?"
        # We use a sliding window approach based on the average detection rate
        max_undetected_gap = {}
        
        # Calculate the longest consecutive time with low detection probability
        # Using the average detection per second, we can estimate gap probabilities
        if time_in_nonobs > 0 and avg_detection_per_second > 0:
            # For each gap threshold, calculate probability using exponential model
            for gap_threshold in [5, 10, 30, 60]:  # seconds
                # Probability of NOT being detected in a gap of length gap_threshold
                prob_gap_undetected = (1.0 - avg_detection_per_second) ** gap_threshold
                
                # Count how many such gaps could theoretically exist
                num_possible_gaps = max(1, int(time_in_nonobs / gap_threshold))
                
                # Probability that at least one gap of this length exists undetected
                # P(at least one gap undetected) = 1 - P(all gaps have detection)
                # P(all gaps have detection) = (1 - prob_gap_undetected)^num_gaps
                prob_all_gaps_covered = (1.0 - prob_gap_undetected) ** num_possible_gaps
                
                # We want: P(NO gap longer than threshold without detection)
                # This is the probability that all potential gaps have at least one detection
                max_undetected_gap[f'{gap_threshold}s'] = prob_all_gaps_covered
        else:
            # No non-observable time or zero detection rate
            for gap_threshold in [5, 10, 30, 60]:
                max_undetected_gap[f'{gap_threshold}s'] = 0.0 if time_in_nonobs > 0 else 1.0
        
        # Calculate sliding window detection statistics (TRUE probabilities)
        sliding_window_stats = self.compute_sliding_window_detection(
            trajectory, 
            time_per_step=time_per_step, 
            window_sizes=sliding_window_sizes,
            only_nonobservable=only_nonobservable,
            include_observable_in_stats=include_observable_in_stats
        )
        
        return {
            'cumulative_detection_prob': cumulative_probs[-1] if cumulative_probs else 0.0,
            'point_probs': point_probs,
            'time_to_detection': time_to_detection,
            'time_in_observable': time_in_obs,
            'time_in_nonobservable': time_in_nonobs,
            'first_detection_time': first_detection_time,
            'cumulative_probs': cumulative_probs,
            'avg_detection_per_second': avg_detection_per_second,
            'detection_rate_per_interval': detection_rate_per_interval,
            'max_undetected_gap': max_undetected_gap,
            'sliding_window_stats': sliding_window_stats
        }
    
    def time_until_detection(self, trajectory, threshold=0.9, time_per_step=1.0):
        """
        Calculate time until cumulative detection probability exceeds threshold.
        
        Args:
            trajectory: list of (x, y) waypoints
            threshold: detection probability threshold (default 0.9 = 90%)
            time_per_step: time per step in seconds
        
        Returns:
            float: time in seconds until threshold exceeded, or None if never exceeded
        """
        analysis = self.analyze_trajectory(trajectory, time_per_step)
        
        for entry in analysis['time_to_detection']:
            if entry['cumulative_prob'] >= threshold:
                return entry['time']
        
        return None
    
    def visualize(self, figsize=(6, 6), ax=None, show=True, show_sectors=True):
        """
        Visualize environment with observable/non-observable sectors highlighted.
        
        Args:
            show_sectors: if True, overlay observable/non-observable regions
        
        All coordinates shown are in physical units (meters).
        """
        # Call parent visualization
        ax = super().visualize(figsize=figsize, ax=ax, show=False)
        
        physical_width = self.width * self.cell_size
        physical_height = self.height * self.cell_size
        
        if show_sectors:
            # Create a mask overlay: yellow tint for observable, light blue tint for non-observable
            sector_overlay = np.zeros((self.height, self.width, 4))
            
            # Observable regions: yellow tint (much more visible)
            sector_overlay[self.observable_mask, :] = [1.0, 1.0, 0.0, 0.5]
            
            # Non-observable regions: light blue tint (much more visible)
            sector_overlay[~self.observable_mask, :] = [0.5, 0.7, 1.0, 0.5]
            
            ax.imshow(sector_overlay, origin='lower', extent=(0, physical_width, 0, physical_height), zorder=1)
            
            # Add legend entries for sectors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='yellow', alpha=0.3, label='Observable'),
                Patch(facecolor='lightblue', alpha=0.3, label='Non-observable')
            ]
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + legend_elements, loc='upper right')
        
        if show:
            plt.show()
        
        return ax
    
    def visualize_trajectories(self, figsize=(6, 6), ax=None, show=True, show_heatmap=True, show_sectors=True):
        """
        Visualize attacker trajectories.
        EXACT same approach as visualize_trajectory_analysis left plot.
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # EXACT same as visualize_trajectory_analysis: call visualize first
        self.visualize(ax=ax, show=False, show_sectors=show_sectors)
        
        # EXACT same as visualize_trajectory_analysis: plot trajectories
        first_attacker = True
        for attacker in self.atk_drones:
            if hasattr(attacker, 'trajectory') and attacker.trajectory:
                traj_arr = np.array(attacker.trajectory)
                # Simple blue line - EXACT same as visualize_trajectory_analysis
                # Only label the first attacker to avoid duplicate legend entries
                if first_attacker:
                    ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, label='Trajectory', zorder=10)
                    ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=10, label='Start', zorder=11)
                    ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=12, label='Target', zorder=11)
                    first_attacker = False
                else:
                    ax.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, zorder=10)
                    ax.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=10, zorder=11)
                    ax.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=12, zorder=11)

        ax.set_title('Attacker Trajectories')
        
        # Filter out 'attack' from legend (redundant with 'Start' markers)
        handles, labels = ax.get_legend_handles_labels()
        filtered = [(h, l) for h, l in zip(handles, labels) if l != 'attack']
        if filtered:
            handles, labels = zip(*filtered)
            ax.legend(handles, labels, loc='upper right')

        if show:
            plt.show()
        
        return ax
    
    def visualize_trajectory_analysis(self, attacker, figsize=(12, 5), sliding_window_sizes=[5, 10, 15, 20, 30], include_observable_in_stats=False):
        """
        Visualize trajectory with detection probability analysis.
        
        Args:
            attacker: Attacker object with .trajectory attribute
            figsize: tuple of (width, height) for figure size
            sliding_window_sizes: list of window sizes (in seconds) for sliding window analysis
            include_observable_in_stats: if True, include observable regions in probability calculations
                                         (treats observable as 100% detection, includes in averages/intervals)
        
        Returns:
            fig, (ax1, ax2): matplotlib figure and axes
        """
        if not hasattr(attacker, 'trajectory') or not attacker.trajectory:
            raise ValueError("Attacker must have a trajectory")
        
        trajectory = attacker.trajectory
        
        # Determine time_per_step from attacker speed if available
        if hasattr(attacker, 'speed') and attacker.speed > 0:
            # Trajectory was generated with per-second sampling
            time_per_step = 1.0
        else:
            # Legacy: estimate from total steps
            start = np.array(trajectory[0])
            target = np.array(trajectory[-1])
            total_dist = np.linalg.norm(target - start)
            time_per_step = 1.0  # default fallback
        
        analysis = self.analyze_trajectory(trajectory, time_per_step, sliding_window_sizes=sliding_window_sizes, only_nonobservable=True, include_observable_in_stats=include_observable_in_stats)
        
        # Get sliding window stats for visualization
        sliding_stats = analysis['sliding_window_stats']['summary']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Trajectory on grid with sectors
        self.visualize(ax=ax1, show=False, show_sectors=True)
        traj_arr = np.array(trajectory)
        ax1.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, label='Trajectory', zorder=10)
        ax1.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=10, label='Start', zorder=11)
        ax1.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=12, label='Target', zorder=11)
        ax1.set_title('Trajectory & Sectors')
        ax1.legend(loc='upper right')
        
        # Right plot: Detection probability over time (only from detectors in non-observable)
        times = np.array([entry['time'] for entry in analysis['time_to_detection']])
        instant_probs = np.array([entry['instant_prob'] for entry in analysis['time_to_detection']])
        in_nonobservable = np.array([entry['in_nonobservable'] for entry in analysis['time_to_detection']])
        
        # Create detection probability array: show detector prob only in non-observable, 1.0 in observable
        detection_probs = np.where(in_nonobservable, instant_probs, 1.0)
        
        # Smooth transitions at boundaries to reduce "wiggle" effect
        # Apply a 3-point moving average at region boundaries
        smoothed_probs = detection_probs.copy()
        for i in range(1, len(detection_probs) - 1):
            # Check if this is a boundary point (region change)
            if in_nonobservable[i] != in_nonobservable[i-1] or in_nonobservable[i] != in_nonobservable[i+1]:
                smoothed_probs[i] = (detection_probs[i-1] + detection_probs[i] + detection_probs[i+1]) / 3.0
        
        # Create continuous shading for observable regions
        # Find contiguous observable segments
        observable_mask = ~in_nonobservable
        if np.any(observable_mask):
            # Find transitions
            diff = np.diff(np.concatenate(([False], observable_mask, [False])).astype(int))
            starts = np.where(diff == 1)[0]
            ends = np.where(diff == -1)[0]
            
            # Shade each observable segment
            for start, end in zip(starts, ends):
                ax2.axvspan(times[start], times[end-1], alpha=0.15, color='green', label='Observable region' if start == starts[0] else '')
        
        # Plot smoothed detection probability
        ax2.plot(times, smoothed_probs, 'b-', linewidth=2, label='Detection probability')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Detection Probability')
        ax2.set_title('Detection Probability Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim(0, 1.05)
        
        # Add text summary with sliding window statistics
        summary_lines = [
            f"Total time: {times[-1]:.1f}s",
            f"Time in observable: {analysis['time_in_observable']:.1f}s",
            f"Time in non-obs: {analysis['time_in_nonobservable']:.1f}s",
            f"\nDetection per second: {analysis['avg_detection_per_second']:.4%}",
            f"\nSliding Window (Mean):"
        ]
        
        if '5s' in sliding_stats:
            summary_lines.append(f"  5s: {sliding_stats['5s']['mean']:.2%}")
        if '15s' in sliding_stats:
            summary_lines.append(f"  15s: {sliding_stats['15s']['mean']:.2%}")
        if '30s' in sliding_stats:
            summary_lines.append(f"  30s: {sliding_stats['30s']['mean']:.2%}")
        
        summary = '\n'.join(summary_lines)
        ax2.text(0.02, 0.98, summary, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)