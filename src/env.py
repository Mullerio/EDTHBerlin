import numpy as np
import matplotlib.pyplot as plt
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

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right')

        if show and created_fig:
            plt.show()

        return ax

    def step(self, dt):
        raise NotImplementedError


