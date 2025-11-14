import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from detectors import Detector
#from attackers import Attacker 
from abc import ABC, abstractmethod
from targets import *


class Environment:
    def __init__(self, width, height, target : TargetDistribution): 
        self.width = width
        self.height = height
        
        self.atk_drones = []
        self.def_drones = []
        self.detectors = []
        
        self.target = target
        x = np.linspace(0, self.width, self.width)
        y = np.linspace(0, self.height, self.height)
        xx, yy = np.meshgrid(x, y)
        
        self.grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
        
        self.prob_map = self._generate_prob_map()
    
    def _generate_prob_map(self) -> np.ndarray:
        """
        Generate probability map over the grid based on the target distribution.
        Returns: shape [width, height]
        """
        
        # Evaluate log density at all grid points
        log_probs = self.target.log_prob(self.grid)  # [width*height, 1]
        
        # Convert log densities to probabilities
        densities = np.exp(log_probs.squeeze())  # [width*height,]
        
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

        Assumes each drone has a `position` attribute with (x, y) coordinates
        in the same coordinate system used by `self.grid` (i.e., x in [0,width], y in [0,height]).
        """
        # Ensure prob_map has shape (height, width)
        prob = self.prob_map

        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Show heatmap; use origin='lower' so coords match (0,0) at bottom-left
        im = ax.imshow(prob, origin='lower', extent=(0, self.width, 0, self.height), cmap='hot', aspect='auto')

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
            ax.imshow(rgba, origin='lower', extent=(0, self.width, 0, self.height), zorder=5)

            # Draw detector radius outline if available
            if r is not None:
                circ = Circle((cx, cy), r, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
                ax.add_patch(circ)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
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
        """
        created_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True

        # Optionally show heatmap in background
        if show_heatmap:
            prob = self.prob_map
            im = ax.imshow(prob, origin='lower', extent=(0, self.width, 0, self.height), 
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

        # Overlay detector probability distributions using the main environment grid
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
            ax.imshow(rgba, origin='lower', extent=(0, self.width, 0, self.height), zorder=5)
            if r is not None:
                circ = Circle((cx, cy), r, edgecolor='blue', facecolor='none', linestyle='--', linewidth=1)
                ax.add_patch(circ)

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
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


