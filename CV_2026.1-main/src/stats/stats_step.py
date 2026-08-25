import numpy as np
import cv2
import pandas as pd
import pathlib

from ..core.frame_data import FrameData
from ..core.pipeline_step import PipelineStep
from .player_heatmap_stats import PlayerHeatmapTracker

class StatsStep(PipelineStep):
    def __init__(
        self,
        tracker: PlayerHeatmapTracker,
        court_length_meters: float = 28.0,
        output_dir: str = "outputs/stats/"
    ):
        """
        Inicializa o step de estatísticas do jogador.
        
        Args:
            tracker (PlayerHeatmapTracker): O acumulador de estatísticas e heatmaps.
            court_length_meters (float): O comprimento real da quadra em metros (padrão FIBA: 28.0).
            output_dir (str): O diretório onde serão salvos todos os outputs (CSV e heatmaps).
        """
        self.tracker = tracker
        self.court_length_meters = court_length_meters
        self.output_dir = output_dir
        self.scale_calibrated = False

    def process(self, data: FrameData) -> FrameData:
        # Atualiza dinamicamente o FPS do tracker a partir do frame_data
        if hasattr(data, "fps") and data.fps is not None and data.fps > 0:
            self.tracker.fps = data.fps

        # Auto-detecta as dimensões da quadra a partir dos pontos do template se ainda não configurado
        if self.tracker.court_width is None and getattr(data, "template_points", None) is not None:
            self.tracker.set_court_dimensions_from_template(data.template_points)
            
        # Calibra a escala física se ainda não foi feito e temos template_points
        if not self.scale_calibrated and getattr(data, "template_points", None) is not None:
            if self.tracker.court_width is None:
                self.tracker.set_court_dimensions_from_template(data.template_points)
            max_template_dim = max(self.tracker.court_width, self.tracker.court_length)
            pixels_per_meter = max_template_dim / self.court_length_meters
            self.tracker.set_world_scale(pixels_per_meter)
            print(f"[StatsStep] Calibrated scale: {pixels_per_meter:.2f} pixels/meter (using length={self.court_length_meters}m)")
            self.scale_calibrated = True

        # Atualiza as estatísticas com as detecções do frame, homografia, times e arremessos
        self.tracker.update_from_frame(
            frame_id=data.frame_id,
            detections=data.detections,
            homography_matrix=data.homography_matrix,
            team_ids=getattr(data, "team_ids", None),
            shooting_flags=getattr(data, "shooting_flags", None)
        )

        # Propaga a lista de arremessos acumulados para o FrameData (usado pelo DrawStep para desenhar no minimapa)
        data.recorded_shots = list(self.tracker.recorded_shots)

        return data

    def _draw_court_template(self) -> np.ndarray:
        """Desenha uma quadra de basquete procedural em tamanho 300x161."""
        minimap = np.zeros((161, 300, 3), dtype=np.uint8)
        
        bg_color = (30, 30, 30)
        court_color = (200, 150, 100)
        paint_color = (150, 90, 40)
        line_color = (255, 255, 255)
        
        minimap[:] = bg_color
        
        x_min, y_min = 0, 0
        x_max, y_max = 300, 161
        x_center, y_center = 150, 80
        
        cv2.rectangle(minimap, (x_min, y_min), (x_max, y_max), court_color, -1)
        cv2.rectangle(minimap, (x_min, y_min), (x_max, y_max), line_color, 2)
        
        cv2.line(minimap, (x_center, y_min), (x_center, y_max), line_color, 2)
        cv2.circle(minimap, (x_center, y_center), 19, line_color, 2)
        
        cv2.rectangle(minimap, (x_min, 54), (62, 106), paint_color, -1)
        cv2.rectangle(minimap, (x_min, 54), (62, 106), line_color, 2)
        cv2.ellipse(minimap, (62, y_center), (19, 19), 0, -90, 90, line_color, 2)
        
        cv2.rectangle(minimap, (238, 54), (x_max, 106), paint_color, -1)
        cv2.rectangle(minimap, (238, 54), (x_max, 106), line_color, 2)
        cv2.ellipse(minimap, (238, y_center), (19, 19), 0, 90, 270, line_color, 2)
        
        cv2.ellipse(minimap, (x_min, y_center), (90, 80), 0, -70, 70, line_color, 2)
        cv2.ellipse(minimap, (x_max, y_center), (90, 80), 0, 110, 250, line_color, 2)
        
        return minimap

    def finish(self):
        """
        Método chamado ao final do pipeline. Cospe o relatório e exporta os arquivos.
        """
        print("\n" + "=" * 60)
        print("  RELATÓRIO DE ESTATÍSTICAS DOS JOGADORES")
        print("=" * 60)
        
        df = self.tracker.get_stats_dataframe()
        if df.empty:
            print("  [AVISO] Nenhuma estatística foi registrada.")
            print("=" * 60)
            return

        with pd.option_context(
            "display.float_format", "{:.2f}".format,
            "display.max_columns", None,
            "display.width", 120
        ):
            print(df.to_string(index=False))
        
        # Cria as pastas de output
        base_path = pathlib.Path(self.output_dir)
        players_path = base_path / "players"
        teams_path = base_path / "teams"
        
        try:
            base_path.mkdir(parents=True, exist_ok=True)
            players_path.mkdir(parents=True, exist_ok=True)
            teams_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"  [ERRO] Não foi possível criar os diretórios de output: {e}")
            print("=" * 60)
            return

        # Salva o arquivo CSV de sumário
        csv_path = base_path / "stats_summary.csv"
        try:
            df.to_csv(csv_path, index=False)
            print(f"\n  CSV de sumário salvo com sucesso → {csv_path}")
        except Exception as e:
            print(f"  [ERRO] Não foi possível salvar o CSV: {e}")

        # Salva o heatmap geral
        try:
            hm_whole = self.tracker.get_player_heatmap(blur_sigma=2.0)
            if hm_whole.max() > 0:
                hm_norm = cv2.normalize(hm_whole, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                path_whole = base_path / "heatmap_whole.png"
                cv2.imwrite(str(path_whole), hm_color)
                print(f"  Heatmap geral salvo → {path_whole}")
        except Exception as e:
            print(f"  [AVISO] Não foi possível salvar o heatmap geral: {e}")

        # Salva os heatmaps por time
        saved_teams = 0
        unique_teams = set(self.tracker.player_teams.values())
        for team_id in unique_teams:
            if team_id < 0:
                continue
            try:
                hm_team = self.tracker.get_team_heatmap(team_id=team_id, blur_sigma=2.0)
                if hm_team.max() > 0:
                    hm_norm = cv2.normalize(hm_team, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                    path_team = teams_path / f"heatmap_team_{team_id}.png"
                    cv2.imwrite(str(path_team), hm_color)
                    saved_teams += 1
            except Exception as e:
                pass
        if saved_teams > 0:
            print(f"  Heatmaps de times salvos ({saved_teams} times) → {teams_path}/")

        # Salva os heatmaps por jogador
        saved_players = 0
        for track_id, positions in self.tracker.player_positions.items():
            if not positions:
                continue
            try:
                hm_player = self.tracker.get_player_heatmap(track_id=track_id, blur_sigma=2.0)
                if hm_player.max() > 0:
                    hm_norm = cv2.normalize(hm_player, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
                    path_player = players_path / f"heatmap_player_{track_id}.png"
                    cv2.imwrite(str(path_player), hm_color)
                    saved_players += 1
            except Exception as e:
                pass
        if saved_players > 0:
            print(f"  Heatmaps individuais salvos ({saved_players} jogadores) → {players_path}/")

        # Salva o mapa de arremessos (shotmap)
        try:
            shotmap = self._draw_court_template()
            for _, _, x_world, y_world in self.tracker.recorded_shots:
                xi, yi = int(x_world), int(y_world)
                if 0 <= xi < 300 and 0 <= yi < 161:
                    length = 4
                    # Desenha borda preta para contraste
                    cv2.line(shotmap, (xi - length, yi - length), (xi + length, yi + length), (0, 0, 0), 3)
                    cv2.line(shotmap, (xi - length, yi + length), (xi + length, yi - length), (0, 0, 0), 3)
                    # Desenha cruz vermelha (shot marker)
                    cv2.line(shotmap, (xi - length, yi - length), (xi + length, yi + length), (0, 0, 255), 1)
                    cv2.line(shotmap, (xi - length, yi + length), (xi + length, yi - length), (0, 0, 255), 1)
            
            # Redimensiona para maior nitidez (900x483)
            shotmap_resized = cv2.resize(shotmap, (900, 483), interpolation=cv2.INTER_CUBIC)
            shotmap_path = base_path / "shotmap.png"
            cv2.imwrite(str(shotmap_path), shotmap_resized)
            print(f"  Shotmap salvo com sucesso → {shotmap_path}")
        except Exception as e:
            print(f"  [AVISO] Não foi possível gerar/salvar o shotmap: {e}")
        
        print("=" * 60)
