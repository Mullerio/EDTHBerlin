"""
Statistical analysis functions for detector sweep results and trajectory data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class SweepAnalyzer:
    """Analyzer for detector sweep results (aggregated across detectors)."""
    
    def __init__(self, csv_path: str):
        """
        Initialize analyzer with sweep results CSV.
        
        Args:
            csv_path: Path to detector_sweep_results.csv file
        """
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        
    def summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for all numeric columns.
        
        Returns:
            DataFrame with mean, std, min, max, median for each metric
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats = self.df[numeric_cols].describe().T
        stats['median'] = self.df[numeric_cols].median()
        return stats
    
    def detection_stats_by_detector_count(self) -> pd.DataFrame:
        """
        Get detection statistics organized by number of detectors.
        
        Returns:
            DataFrame indexed by n_detectors with key detection metrics
        """
        cols = ['n_detectors', 'mean_cumulative_detection_prob', 
                'mean_avg_detection_per_second', 'mean_time_in_nonobservable']
        return self.df[cols].set_index('n_detectors')
    
    def sliding_window_stats(self, window_size: str = '5s') -> pd.DataFrame:
        """
        Extract statistics for a specific sliding window size.
        
        Args:
            window_size: One of '5s', '10s', '15s', '30s', '60s'
            
        Returns:
            DataFrame with min, max, mean for the specified window
        """
        cols = ['n_detectors',
                f'mean_sliding_window_{window_size}_min',
                f'mean_sliding_window_{window_size}_max',
                f'mean_sliding_window_{window_size}_mean']
        return self.df[cols].set_index('n_detectors')
    
    def all_window_means(self) -> pd.DataFrame:
        """
        Get mean detection probability for all time windows.
        
        Returns:
            DataFrame with n_detectors as index and window sizes as columns
        """
        window_cols = {
            '5s': 'mean_sliding_window_5s_mean',
            '10s': 'mean_sliding_window_10s_mean',
            '15s': 'mean_sliding_window_15s_mean',
            '30s': 'mean_sliding_window_30s_mean',
            '60s': 'mean_sliding_window_60s_mean'
        }
        df = self.df[['n_detectors'] + list(window_cols.values())].copy()
        df.columns = ['n_detectors'] + list(window_cols.keys())
        return df.set_index('n_detectors')
    
    def plot_detection_vs_detectors(self, figsize=(14, 5)):
        """Plot detection rate and sliding windows vs number of detectors."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Detection per second
        ax1.plot(self.df['n_detectors'], 
                self.df['mean_avg_detection_per_second'], 
                'o-', linewidth=2, markersize=8, color='coral')
        ax1.set_xlabel('Number of Detectors', fontsize=12)
        ax1.set_ylabel('Mean Detection Probability per Second', fontsize=12)
        ax1.set_title('Detection Rate vs Detector Count', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Sliding window means comparison
        window_data = self.all_window_means()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, col in enumerate(window_data.columns):
            ax2.plot(window_data.index, window_data[col], 
                   marker=markers[i], linewidth=2, markersize=7,
                   label=col, color=colors[i])
        
        ax2.set_xlabel('Number of Detectors', fontsize=12)
        ax2.set_ylabel('Mean Detection Probability', fontsize=12)
        ax2.set_title('Sliding Window Detection vs Detector Count', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_sliding_windows(self, figsize=(14, 6)):
        """Plot all sliding window means across detector counts."""
        window_data = self.all_window_means()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'v']
        
        for i, col in enumerate(window_data.columns):
            ax.plot(window_data.index, window_data[col], 
                   marker=markers[i], linewidth=2, markersize=8,
                   label=f'{col} window', color=colors[i])
        
        ax.set_xlabel('Number of Detectors', fontsize=12)
        ax.set_ylabel('Mean Detection Probability', fontsize=12)
        ax.set_title('Sliding Window Detection Probability vs Detector Count', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_time_analysis(self, figsize=(10, 6)):
        """Plot time spent in observable vs non-observable regions."""
        fig, ax = plt.subplots(figsize=figsize)
        
        x = self.df['n_detectors']
        width = 2
        
        ax.bar(x - width/2, self.df['mean_time_in_observable'], 
              width, label='Observable', color='green', alpha=0.7)
        ax.bar(x + width/2, self.df['mean_time_in_nonobservable'], 
              width, label='Non-observable', color='red', alpha=0.7)
        
        ax.set_xlabel('Number of Detectors', fontsize=12)
        ax.set_ylabel('Mean Time (seconds)', fontsize=12)
        ax.set_title('Time in Observable vs Non-Observable Regions', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax


class TrajectoryAnalyzer:
    """Analyzer for individual trajectory results (per-detector analysis)."""
    
    def __init__(self, csv_path: str, n_detectors: Optional[int] = None):
        """
        Initialize analyzer with trajectory-level CSV.
        
        Args:
            csv_path: Path to with_detectors_exclude_observable.csv file
            n_detectors: Number of detectors (extracted from path if None)
        """
        self.df = pd.read_csv(csv_path)
        self.csv_path = csv_path
        
        # Ensure trajectory_id is numeric
        if 'trajectory_id' in self.df.columns:
            self.df['trajectory_id'] = pd.to_numeric(self.df['trajectory_id'], errors='coerce')
        
        # Try to extract n_detectors from path
        if n_detectors is None:
            path_parts = Path(csv_path).parts
            for part in path_parts:
                if part.startswith('n') and '_detectors' in part:
                    n_detectors = int(part.replace('n', '').replace('_detectors', ''))
                    break
        self.n_detectors = n_detectors
    
    def summary_statistics(self) -> pd.DataFrame:
        """
        Compute summary statistics for all numeric columns.
        
        Returns:
            DataFrame with mean, std, min, max, median for each metric
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'trajectory_id']
        stats = self.df[numeric_cols].describe().T
        stats['median'] = self.df[numeric_cols].median()
        return stats
    
    def trajectory_detection_summary(self) -> pd.DataFrame:
        """
        Get detection summary for each trajectory.
        
        Returns:
            DataFrame with key metrics per trajectory
        """
        cols = ['trajectory_id', 'cumulative_detection_prob', 
                'avg_detection_per_second', 'time_in_nonobservable']
        return self.df[cols]
    
    def sliding_window_stats(self, window_size: str = '5s') -> Dict[str, float]:
        """
        Get statistics for a specific sliding window across all trajectories.
        
        Args:
            window_size: One of '5s', '10s', '15s', '30s', '60s'
            
        Returns:
            Dictionary with mean, std, min, max of the window metric
        """
        col_mean = f'sliding_window_{window_size}_mean'
        col_min = f'sliding_window_{window_size}_min'
        col_max = f'sliding_window_{window_size}_max'
        
        return {
            'mean_of_means': self.df[col_mean].mean(),
            'std_of_means': self.df[col_mean].std(),
            'min_across_all': self.df[col_min].min(),
            'max_across_all': self.df[col_max].max(),
            'mean_of_mins': self.df[col_min].mean(),
            'mean_of_maxs': self.df[col_max].mean()
        }
    
    def all_window_means(self) -> pd.DataFrame:
        """
        Get mean detection for all windows across all trajectories.
        
        Returns:
            DataFrame with trajectory_id as index and window sizes as columns
        """
        window_cols = {
            '5s': 'sliding_window_5s_mean',
            '10s': 'sliding_window_10s_mean',
            '15s': 'sliding_window_15s_mean',
            '30s': 'sliding_window_30s_mean',
            '60s': 'sliding_window_60s_mean'
        }
        df = self.df[['trajectory_id'] + list(window_cols.values())].copy()
        df.columns = ['trajectory_id'] + list(window_cols.keys())
        return df.set_index('trajectory_id')
    
    def plot_trajectory_detection(self, figsize=(14, 5)):
        """Plot detection rate and sliding windows for each trajectory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Detection rate per trajectory
        ax1.bar(self.df['trajectory_id'], 
               self.df['avg_detection_per_second'],
               color='coral', alpha=0.7)
        ax1.set_xlabel('Trajectory ID', fontsize=12)
        ax1.set_ylabel('Detection Probability per Second', fontsize=12)
        ax1.set_title(f'Detection Rate per Trajectory ({self.n_detectors} detectors)', 
                     fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Sliding window means for each trajectory
        window_data = self.all_window_means()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, col in enumerate(window_data.columns):
            ax2.plot(window_data.index, window_data[col], 
                    linewidth=1.5, alpha=0.7, color=colors[i], label=col)
        
        ax2.set_xlabel('Trajectory ID', fontsize=12)
        ax2.set_ylabel('Mean Detection Probability', fontsize=12)
        ax2.set_title(f'Sliding Window Detection per Trajectory ({self.n_detectors} detectors)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def plot_sliding_windows_distribution(self, figsize=(14, 8)):
        """Plot distribution of sliding window means across trajectories."""
        window_data = self.all_window_means()
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, col in enumerate(window_data.columns):
            ax = axes[i]
            ax.hist(window_data[col], bins=20, color='steelblue', 
                   alpha=0.7, edgecolor='black')
            ax.set_xlabel('Mean Detection Probability', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{col} Window', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add statistics text
            mean_val = window_data[col].mean()
            std_val = window_data[col].std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.3f}')
            ax.legend(fontsize=9)
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.suptitle(f'Sliding Window Detection Distributions ({self.n_detectors} detectors)', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        return fig, axes
    
    def plot_time_comparison(self, figsize=(10, 6)):
        """Plot time in observable vs non-observable for each trajectory."""
        fig, ax = plt.subplots(figsize=figsize)
        
        x = self.df['trajectory_id']
        width = 0.35
        
        ax.bar(x - width/2, self.df['time_in_observable'], 
              width, label='Observable', color='green', alpha=0.7)
        ax.bar(x + width/2, self.df['time_in_nonobservable'], 
              width, label='Non-observable', color='red', alpha=0.7)
        
        ax.set_xlabel('Trajectory ID', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title(f'Time per Region by Trajectory ({self.n_detectors} detectors)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def correlation_matrix(self, figsize=(12, 10)):
        """Plot correlation matrix of all numeric features."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_cols = [c for c in numeric_cols if c != 'trajectory_id']
        
        corr = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_title(f'Feature Correlation Matrix ({self.n_detectors} detectors)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig, ax


def compare_detector_counts(trajectory_csv_paths: List[str], figsize=(14, 6)):
    """
    Compare detection statistics across different detector counts.
    
    Args:
        trajectory_csv_paths: List of paths to trajectory CSV files
        figsize: Figure size
        
    Returns:
        Figure and axes objects
    """
    analyzers = [TrajectoryAnalyzer(path) for path in trajectory_csv_paths]
    labels = [f"{a.n_detectors}" for a in analyzers]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Box plot of detection per second
    data_rate = [a.df['avg_detection_per_second'] for a in analyzers]
    
    bp1 = ax1.boxplot(data_rate, labels=labels, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('coral')
        patch.set_alpha(0.7)
    ax1.set_xlabel('Number of Detectors', fontsize=12)
    ax1.set_ylabel('Detection Probability per Second', fontsize=12)
    ax1.set_title('Detection Rate Distribution by Detector Count', 
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Box plot comparison for 30s sliding window
    data_30s = [a.df['sliding_window_30s_mean'] for a in analyzers]
    
    bp2 = ax2.boxplot(data_30s, labels=labels, patch_artist=True)
    for patch in bp2['boxes']:
        patch.set_facecolor('steelblue')
        patch.set_alpha(0.7)
    ax2.set_xlabel('Number of Detectors', fontsize=12)
    ax2.set_ylabel('30s Window Detection Probability', fontsize=12)
    ax2.set_title('30s Sliding Window Distribution by Detector Count', 
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return fig, (ax1, ax2)
