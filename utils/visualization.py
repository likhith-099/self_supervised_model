"""
Visualization utilities for satellite imagery and model outputs
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import cv2


def display_image_grid(
    images: List[np.ndarray],
    titles: List[str] = None,
    cols: int = 4,
    figsize: Tuple[int, int] = (20, 5),
    save_path: str = None,
    show: bool = True
):
    """
    Display a grid of images
    
    Args:
        images: List of image arrays
        titles: Optional titles for each image
        cols: Number of columns
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, ax in enumerate(axes.flat):
        if idx < n_images:
            ax.imshow(images[idx])
            if titles:
                ax.set_title(titles[idx])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved grid to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    cmap: str = 'Reds',
    save_path: str = None,
    show: bool = True
):
    """
    Visualize mask overlaid on image
    
    Args:
        image: Base image
        mask: Mask array (binary or probabilistic)
        alpha: Overlay transparency
        cmap: Colormap for mask
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ax.imshow(image)
    ax.imshow(mask, cmap=cmap, alpha=alpha)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_reconstruction(
    original: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray = None,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize MAE reconstruction quality
    
    Args:
        original: Original image
        reconstruction: Reconstructed image
        mask: Optional mask showing masked regions
        save_path: Path to save figure
        show: Whether to display plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(original)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Reconstruction
    axes[1].imshow(reconstruction)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')
    
    # Difference
    diff = np.abs(original.astype(float) - reconstruction.astype(float))
    diff = (diff - diff.min()) / (diff.max() - diff.min())
    axes[2].imshow(diff)
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_feature_maps(
    feature_maps: np.ndarray,
    titles: List[str] = None,
    cols: int = 8,
    figsize: Tuple[int, int] = (20, 3),
    save_path: str = None,
    show: bool = True
):
    """
    Visualize feature maps from intermediate layers
    
    Args:
        feature_maps: Array of shape (N, H, W) or (N, H, W, C)
        titles: Optional titles for each map
        cols: Number of columns
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display plot
    """
    if len(feature_maps.shape) == 4:
        # Average across channels if needed
        feature_maps = feature_maps.mean(axis=-1)
    
    n_maps = feature_maps.shape[0]
    rows = (n_maps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < n_maps:
            fm = feature_maps[idx]
            fm = (fm - fm.min()) / (fm.max() - fm.min())
            ax.imshow(fm, cmap='viridis')
            
            if titles and idx < len(titles):
                ax.set_title(titles[idx], fontsize=8)
            
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    metrics: List[str] = None,
    save_path: str = None,
    show: bool = True
):
    """
    Plot training history curves
    
    Args:
        history: Dictionary of metric lists
        metrics: List of metrics to plot
        save_path: Path to save figure
        show: Whether to display plot
    """
    if metrics is None:
        metrics = list(history.keys())
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        values = history[metric]
        
        ax.plot(values, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
        
        # Add min/max annotations
        if len(values) > 0:
            min_val = min(values)
            max_val = max(values)
            ax.axhline(y=min_val, color='green', linestyle='--', alpha=0.5, label=f'Min: {min_val:.4f}')
            ax.axhline(y=max_val, color='red', linestyle='--', alpha=0.5, label=f'Max: {max_val:.4f}')
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def create_comparison_plot(
    images_dict: Dict[str, np.ndarray],
    title: str = "Comparison",
    save_path: str = None,
    show: bool = True
):
    """
    Create side-by-side comparison of different images
    
    Args:
        images_dict: Dictionary of {label: image}
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    n_images = len(images_dict)
    fig, axes = plt.subplots(1, n_images, figsize=(5 * n_images, 5))
    
    if n_images == 1:
        axes = [axes]
    
    for idx, (label, image) in enumerate(images_dict.items()):
        ax = axes[idx]
        ax.imshow(image)
        ax.set_title(label)
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_attention_rollout(
    attention_weights: List[np.ndarray],
    input_image: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    Visualize attention rollout from transformer layers
    
    Args:
        attention_weights: List of attention weight matrices
        input_image: Original input image
        save_path: Path to save figure
        show: Whether to display plot
    """
    # TODO: Implement attention rollout visualization
    # This requires aggregating attention across layers
    print("Attention rollout visualization not yet implemented")


def main():
    """Test visualization functions"""
    # Create dummy data
    original = np.random.rand(256, 256, 3)
    reconstruction = original + np.random.randn(256, 256, 3) * 0.1
    mask = np.random.rand(256, 256) > 0.5
    
    # Test reconstruction visualization
    visualize_reconstruction(
        original,
        reconstruction,
        mask,
        save_path='test_reconstruction.png',
        show=False
    )
    
    # Test feature map visualization
    feature_maps = np.random.rand(16, 32, 32)
    visualize_feature_maps(
        feature_maps,
        save_path='test_features.png',
        show=False
    )
    
    print("Visualizations saved successfully!")


if __name__ == '__main__':
    main()
