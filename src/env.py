import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from src.detectors import Detector
#from attackers import Attacker 
from abc import ABC, abstractmethod
from src.targets import *


class Environment:
    def __init__(self, width, height, target : TargetDistribution, cell_size=1.0): 
        """
        Initialize environment.
        
        Args:
            width: number of grid cells in x direction (integer)
            height: number of grid cells in y direction (integer)
            target: target probability distribution (operates in physical coordinates)
            cell_size: physical size of each grid cell in meters (default 1.0)
                      Physical coordinates range: x ∈ [0, width*cell_size), y ∈ [0, height*cell_size)
        
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
        
        self.target = target
        
        # Grid for visualization only - create sampling points in physical coordinates
        x = np.linspace(0, self.width * self.cell_size, self.width)
        y = np.linspace(0, self.height * self.cell_size, self.height)
        xx, yy = np.meshgrid(x, y)
        
        self.grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        # Probability map for visualization only (not used in detection logic)
        self.prob_map = self._generate_prob_map()
        
        # Observable mask: grid-based boolean array for UI region selection
        # mask[j, i] corresponds to physical region [i*cell_size, (i+1)*cell_size) x [j*cell_size, (j+1)*cell_size)
        self.observable_mask = np.ones((self.height, self.width), dtype=bool)
        
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
        """
        
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
        _plot_list(self.detectors, 'blue', 'detect')

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

            # Draw detector radius outline if available (in physical coordinates)
            if r is not None:
                circ = Circle((cx, cy), r, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
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
        _plot_list(self.detectors, 'blue', 'detect')

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
                circ = Circle((cx, cy), r, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
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
    
    def __init__(self, width, height, target: TargetDistribution, 
                 default_observable=True, cell_size=1.0):
        """
        Initialize sector environment.
        
        Args:
            width: number of grid cells in x direction (integer)
            height: number of grid cells in y direction (integer)
            target: target probability distribution (operates in physical meters)
            default_observable: if True, entire grid starts as observable (then mark subgrids as non-observable)
                               if False, entire grid starts as non-observable (then mark subgrids as observable)
            cell_size: physical size of each grid cell in meters (default 1.0)
        """
        super().__init__(width, height, target, cell_size=cell_size)
        
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
    
    def analyze_trajectory(self, trajectory, time_per_step=1.0, only_nonobservable=True):
        """
        Analyze detection probability along an attacker trajectory.
        
        Args:
            trajectory: list of (x, y) waypoints
            time_per_step: time elapsed between consecutive waypoints (seconds)
            only_nonobservable: if True, only calculate detection for non-observable region
                              (ignores observable regions where detection is automatic)
        
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
        
        # Calculate average detection probability per second in non-observable region
        total_detection_prob_nonobs = sum(p for i, p in enumerate(point_probs) 
                                         if not self.is_observable_at(*trajectory[i]))
        avg_detection_per_second = (total_detection_prob_nonobs / time_in_nonobs 
                                   if time_in_nonobs > 0 else 0.0)
        
        # Calculate detection rate per various time intervals
        detection_rate_per_interval = {}
        for interval in [1, 5, 10, 30, 60]:  # seconds
            steps_per_interval = max(1, int(interval / time_per_step))
            # Probability of being detected at least once in 'interval' seconds
            prob_not_detected_interval = (1.0 - avg_detection_per_second) ** interval
            prob_detected_interval = 1.0 - prob_not_detected_interval
            detection_rate_per_interval[f'{interval}s'] = prob_detected_interval
        
        return {
            'cumulative_detection_prob': cumulative_probs[-1] if cumulative_probs else 0.0,
            'point_probs': point_probs,
            'time_to_detection': time_to_detection,
            'time_in_observable': time_in_obs,
            'time_in_nonobservable': time_in_nonobs,
            'first_detection_time': first_detection_time,
            'cumulative_probs': cumulative_probs,
            'avg_detection_per_second': avg_detection_per_second,
            'detection_rate_per_interval': detection_rate_per_interval
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
            # Create a mask overlay: green tint for observable, red tint for non-observable
            sector_overlay = np.zeros((self.height, self.width, 4))
            
            # Observable regions: slight green tint
            sector_overlay[self.observable_mask, :] = [0.0, 1.0, 0.0, 0.15]
            
            # Non-observable regions: slight red tint
            sector_overlay[~self.observable_mask, :] = [1.0, 0.0, 0.0, 0.15]
            
            ax.imshow(sector_overlay, origin='lower', extent=(0, physical_width, 0, physical_height), zorder=1)
            
            # Add legend entries for sectors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.3, label='Observable'),
                Patch(facecolor='red', alpha=0.3, label='Non-observable')
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
    
    def visualize_trajectory_analysis(self, attacker, figsize=(12, 5)):
        """
        Visualize trajectory with detection probability analysis.
        
        Args:
            attacker: Attacker object with .trajectory attribute
        
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
        
        analysis = self.analyze_trajectory(trajectory, time_per_step)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Left plot: Trajectory on grid with sectors
        self.visualize(ax=ax1, show=False, show_sectors=True)
        traj_arr = np.array(trajectory)
        ax1.plot(traj_arr[:, 0], traj_arr[:, 1], 'b-', linewidth=2, label='Trajectory', zorder=10)
        ax1.plot(traj_arr[0, 0], traj_arr[0, 1], 'go', markersize=10, label='Start', zorder=11)
        ax1.plot(traj_arr[-1, 0], traj_arr[-1, 1], 'rx', markersize=12, label='Target', zorder=11)
        ax1.set_title('Trajectory & Sectors')
        ax1.legend(loc='upper right')
        
        # Right plot: Detection probability over time
        times = [entry['time'] for entry in analysis['time_to_detection']]
        instant_probs = [entry['instant_prob'] for entry in analysis['time_to_detection']]
        cumulative_probs = [entry['cumulative_prob'] for entry in analysis['time_to_detection']]
        
        ax2.plot(times, instant_probs, 'b-', label='Instant detection prob', alpha=0.6)
        ax2.plot(times, cumulative_probs, 'r-', linewidth=2, label='Cumulative detection prob')
        ax2.axhline(y=0.5, color='orange', linestyle='--', label='50% threshold')
        ax2.axhline(y=0.9, color='red', linestyle='--', label='90% threshold')
        
        if analysis['first_detection_time'] is not None:
            ax2.axvline(x=analysis['first_detection_time'], color='purple', 
                       linestyle=':', label=f'50% at t={analysis["first_detection_time"]:.1f}s')
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Detection Probability')
        ax2.set_title('Detection Probability Analysis')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')
        ax2.set_ylim(0, 1.05)
        
        # Add text summary
        summary = (f"Total time: {times[-1]:.1f}s\n"
                  f"Time in observable: {analysis['time_in_observable']:.1f}s\n"
                  f"Time in non-obs: {analysis['time_in_nonobservable']:.1f}s\n"
                  f"Final detection prob: {analysis['cumulative_detection_prob']:.2%}\n"
                  f"\nDetection per second: {analysis['avg_detection_per_second']:.3%}\n"
                  f"Prob per 10s: {analysis['detection_rate_per_interval']['10s']:.2%}\n"
                  f"Prob per 30s: {analysis['detection_rate_per_interval']['30s']:.2%}")
        ax2.text(0.02, 0.98, summary, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=8)
        
        plt.tight_layout()
        plt.show()
        
        return fig, (ax1, ax2)