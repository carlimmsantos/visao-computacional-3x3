
from typing import  List, Optional, Tuple
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class StatsVisualizer:

    @staticmethod
    def plot_heatmap(
        heatmap: np.ndarray,
        court_image_path: Optional[str] = None,
        title: str = "Player Heatmap",
        alpha: float = 0.6,
        save_path: Optional[str] = None
    ):
        '''Heatmap por jogador'''
        
        fig,ax = plt.subplots(figsize=(10,8))
        if court_image_path and Path(court_image_path).exists():

            court_img = cv2.imread(court_image_path)
            court_img = cv2.cvtColor(court_img, cv2.COLOR_BGR2RGB)
            ax.imshow(court_img, extent=[0,heatmap.shape[1], 0 , heatmap.shape[0]])

        if np.max(heatmap) > 0:
            hm_norm = heatmap / np.max(heatmap)
        
        else:
            hm_norm = heatmap
        
        im = ax.imshow(hm_norm, cmap='hot', alpha=alpha, 
                       extent=[0, heatmap.shape[1], 0, heatmap.shape[0]],
                       interpolation='bilinear')
        
        plt.colorbar(im, ax=ax, label='Presence intensity')
        ax.set_title(title)
        ax.set_xlabel("Court X (grid cells)")
        ax.set_ylabel("Court Y (grid cells)")
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_distance_bar(stats_df: pd.DataFrame, save_path: Optional[str] = None):

        '''Grafico de barras com distancias por jogador'''
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(stats_df['track_id'].astype(str), stats_df['total_distance (m)'],
                      color='skyblue', edgecolor='navy')
        
        ax.set_xlabel('Player ID')
        ax.set_ylabel('Total distance (meters)')
        ax.set_title('Total distance traveled')
        for bar, d in zip(bars, stats_df['total_distance (m)']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{d:.1f}m', ha='center', va='bottom')
            
        plt.xticks(rotation=45)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()
    
    @staticmethod
    def plot_speed_profile(track_id: int,
                           speeds: List[Tuple[int, float]],
                           fps: float,
                           save_path: Optional[str] = None):
        
        """Velocidade em relacao ao tempo por jogador."""
        if not speeds:
            print(f"No speed data for player {track_id}")
            return
        frames, speed_vals = zip(*speeds)
        time_sec = np.array(frames) / fps
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(time_sec, speed_vals, 'b-', alpha=0.6, linewidth=1)
        # moving average
        window = min(30, len(speed_vals))
        if window > 1:
            ma = np.convolve(speed_vals, np.ones(window)/window, mode='valid')
            time_ma = time_sec[window-1:]
            ax.plot(time_ma, ma, 'r-', linewidth=2, label='Moving average')
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Speed (m/s)')
        ax.set_title(f'Player {track_id} – Speed profile')
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()

    @staticmethod
    def plot_trajectory(positions: List[Tuple[int, float, float]],
                        track_id: int,
                        color_by_speed: bool = False,
                        speeds: Optional[List[Tuple[int, float]]] = None,
                        save_path: Optional[str] = None):
        """Plota a trajetoria de um jogador na quadra baseado em seu ID do tracking"""
        frames, xs, ys = zip(*positions)
        xs = np.array(xs)
        ys = np.array(ys)
        fig, ax = plt.subplots(figsize=(8, 8))
        if color_by_speed and speeds:
            # interpolate speeds to each position
            speed_dict = dict(speeds)
            colors = [speed_dict.get(f, 0) for f in frames]
            sc = ax.scatter(xs, ys, c=colors, cmap='plasma', s=20, alpha=0.7)
            plt.colorbar(sc, ax=ax, label='Speed (m/s)')
        else:
            ax.plot(xs, ys, 'b-', alpha=0.5)
            ax.scatter(xs[0], ys[0], c='green', s=100, marker='o', label='Start')
            ax.scatter(xs[-1], ys[-1], c='red', s=100, marker='s', label='End')
        ax.set_aspect('equal')
        ax.set_xlabel('X (world units)')
        ax.set_ylabel('Y (world units)')
        ax.set_title(f'Player {track_id} – Movement trajectory')
        ax.legend()
        ax.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()