import json
import math
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np


# =========================================================
# Canvas com zoom, scroll e suporte a desenho de pontos
# =========================================================
class ZoomCanvas(ttk.Frame):
    def __init__(self, master, titulo, point_color="red", on_click_callback=None, on_drag_callback=None):
        super().__init__(master)
    
        self.titulo = ttk.Label(self, text=titulo, font=("Arial", 11, "bold"))
        self.titulo.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(5, 2))

        self.canvas = tk.Canvas(self, bg="#222222", highlightthickness=1, highlightbackground="#666666")
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.hbar = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)

        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)

        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.vbar.grid(row=1, column=1, sticky="ns")
        self.hbar.grid(row=2, column=0, sticky="ew")

        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_bgr = None
        self.image_rgb = None
        self.pil_image = None
        self.tk_image = None
        self.image_item = None

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 6.0

        self.point_color = point_color
        self.points = []  # pontos em coordenadas da imagem original [(x, y), ...]
        self.point_items = []
        self.text_items = []

        self.on_click_callback = on_click_callback
        self.on_drag_callback = on_drag_callback

        self.dragging_index = None
        self.drag_threshold = 12  # em pixels de tela

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<ButtonPress-3>", self._on_right_press)
        self.canvas.bind("<B3-Motion>", self._on_right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_right_release)

        # Zoom mouse wheel
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)  # Linux down

    def set_title(self, text):
        self.titulo.config(text=text)

    def clear_all(self):
        self.points = []
        self.dragging_index = None
        self.redraw()

    def set_points(self, points):
        self.points = list(points)
        self.redraw()

    def get_points(self):
        return list(self.points)

    def set_image_bgr(self, bgr):
        self.image_bgr = bgr.copy()
        self.image_rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        self.pil_image = Image.fromarray(self.image_rgb)
        self.scale = 1.0
        self.redraw()

    def has_image(self):
        return self.pil_image is not None

    def redraw(self):
        self.canvas.delete("all")
        self.point_items = []
        self.text_items = []

        if self.pil_image is None:
            self.canvas.config(scrollregion=(0, 0, 1000, 800))
            return

        w, h = self.pil_image.size
        disp_w = max(1, int(w * self.scale))
        disp_h = max(1, int(h * self.scale))

        resized = self.pil_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized)
        self.image_item = self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)

        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        self._draw_points()

    def _draw_points(self):
        r = 5
        font_size = max(10, int(11 * self.scale**0.2))

        for i, (x, y) in enumerate(self.points, start=1):
            sx, sy = self.image_to_screen(x, y)

            oval = self.canvas.create_oval(
                sx - r, sy - r, sx + r, sy + r,
                fill=self.point_color, outline="white", width=1.5
            )
            text = self.canvas.create_text(
                sx + 10, sy - 10,
                text=str(i),
                fill=self.point_color,
                font=("Arial", font_size, "bold"),
                anchor="sw"
            )
            self.point_items.append(oval)
            self.text_items.append(text)

    def image_to_screen(self, x, y):
        return x * self.scale, y * self.scale

    def screen_to_image(self, sx, sy):
        return sx / self.scale, sy / self.scale

    def _canvas_event_to_image_coords(self, event):
        sx = self.canvas.canvasx(event.x)
        sy = self.canvas.canvasy(event.y)
        return self.screen_to_image(sx, sy)

    def _inside_image(self, x, y):
        if self.pil_image is None:
            return False
        w, h = self.pil_image.size
        return 0 <= x < w and 0 <= y < h

    def _on_left_click(self, event):
        if not self.has_image():
            return
        x, y = self._canvas_event_to_image_coords(event)
        if not self._inside_image(x, y):
            return
        if self.on_click_callback:
            self.on_click_callback(self, float(x), float(y))

    def _find_nearest_point(self, sx, sy):
        if not self.points:
            return None

        best_idx = None
        best_dist = float("inf")

        for idx, (x, y) in enumerate(self.points):
            px, py = self.image_to_screen(x, y)
            dist = math.hypot(px - sx, py - sy)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx

        if best_dist <= self.drag_threshold:
            return best_idx
        return None

    def _on_right_press(self, event):
        if not self.has_image():
            return
        sx = self.canvas.canvasx(event.x)
        sy = self.canvas.canvasy(event.y)
        self.dragging_index = self._find_nearest_point(sx, sy)

    def _on_right_drag(self, event):
        if self.dragging_index is None:
            return

        x, y = self._canvas_event_to_image_coords(event)
        if self.pil_image is None:
            return

        w, h = self.pil_image.size
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)

        self.points[self.dragging_index] = (float(x), float(y))
        self.redraw()

        if self.on_drag_callback:
            self.on_drag_callback(self)

    def _on_right_release(self, event):
        self.dragging_index = None

    def _zoom_at(self, factor, mouse_x, mouse_y):
        if self.pil_image is None:
            return

        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, self.scale * factor))
        if abs(new_scale - old_scale) < 1e-9:
            return

        canvas_x = self.canvas.canvasx(mouse_x)
        canvas_y = self.canvas.canvasy(mouse_y)

        img_x = canvas_x / old_scale
        img_y = canvas_y / old_scale

        self.scale = new_scale
        self.redraw()

        new_canvas_x = img_x * new_scale
        new_canvas_y = img_y * new_scale

        bbox = self.canvas.bbox("all")
        if bbox is None:
            return

        total_w = max(1, bbox[2] - bbox[0])
        total_h = max(1, bbox[3] - bbox[1])

        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())

        xfrac = (new_canvas_x - mouse_x) / max(1, total_w - view_w)
        yfrac = (new_canvas_y - mouse_y) / max(1, total_h - view_h)

        xfrac = min(max(xfrac, 0.0), 1.0)
        yfrac = min(max(yfrac, 0.0), 1.0)

        self.canvas.xview_moveto(xfrac)
        self.canvas.yview_moveto(yfrac)

    def _on_mousewheel(self, event):
        if event.delta > 0:
            self._zoom_at(1.1, event.x, event.y)
        else:
            self._zoom_at(1 / 1.1, event.x, event.y)

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self._zoom_at(1.1, event.x, event.y)
        elif event.num == 5:
            self._zoom_at(1 / 1.1, event.x, event.y)


# =========================================================
# Aplicação principal
# =========================================================
class HomographyApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Editor Interativo de Homografia - Tkinter")
        self.geometry("1650x900")
        self.minsize(1300, 750)

        self.original_path = None
        self.target_path = None

        self.original_img = None  # BGR
        self.target_img = None    # BGR
        self.result_img = None    # BGR

        self.phase = "original"  # original -> target
        self.max_points_var = tk.IntVar(value=8)

        self._build_ui()
        self._update_status()

    def _build_ui(self):
        # Topbar
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=8, pady=8)

        ttk.Button(top, text="Carregar Imagem Original", command=self.load_original).pack(side="left", padx=4)
        ttk.Button(top, text="Carregar Imagem Target", command=self.load_target).pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(top, text="Máximo de pontos (N):").pack(side="left", padx=(4, 2))
        self.spin_max = ttk.Spinbox(top, from_=4, to=200, width=6, textvariable=self.max_points_var)
        self.spin_max.pack(side="left", padx=4)

        ttk.Button(top, text="Ir para Target", command=self.go_to_target_phase).pack(side="left", padx=6)
        ttk.Button(top, text="Calcular Homografia", command=self.calculate_homography).pack(side="left", padx=6)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Button(top, text="Desfazer Original", command=lambda: self.undo_last("original")).pack(side="left", padx=4)
        ttk.Button(top, text="Desfazer Target", command=lambda: self.undo_last("target")).pack(side="left", padx=4)
        ttk.Button(top, text="Limpar Original", command=lambda: self.clear_points("original")).pack(side="left", padx=4)
        ttk.Button(top, text="Limpar Target", command=lambda: self.clear_points("target")).pack(side="left", padx=4)
        ttk.Button(top, text="Limpar Tudo", command=self.clear_all_points).pack(side="left", padx=4)

        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Button(top, text="Salvar Pontos JSON", command=self.save_points_json).pack(side="left", padx=4)

        # Corpo principal
        body = ttk.Frame(self)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        # Painel esquerdo: imagens
        left_panel = ttk.Frame(body)
        left_panel.pack(side="left", fill="both", expand=True)

        # Painel lateral direito: status / pares
        right_panel = ttk.Frame(body, width=320)
        right_panel.pack(side="right", fill="y", padx=(8, 0))
        right_panel.pack_propagate(False)

        # Grid de canvases
        canvases_frame = ttk.Frame(left_panel)
        canvases_frame.pack(fill="both", expand=True)

        self.original_canvas = ZoomCanvas(
            canvases_frame,
            "Imagem Original",
            point_color="#ff4d4d",
            on_click_callback=self.on_canvas_click,
            on_drag_callback=self.on_canvas_drag
        )
        self.target_canvas = ZoomCanvas(
            canvases_frame,
            "Imagem Target",
            point_color="#4da6ff",
            on_click_callback=self.on_canvas_click,
            on_drag_callback=self.on_canvas_drag
        )
        self.result_canvas = ZoomCanvas(
            canvases_frame,
            "Resultado (Original projetada sobre Target)",
            point_color="#00cc66",
            on_click_callback=None,
            on_drag_callback=None
        )

        self.original_canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.target_canvas.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.result_canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

        canvases_frame.grid_rowconfigure(0, weight=1)
        canvases_frame.grid_rowconfigure(1, weight=1)
        canvases_frame.grid_columnconfigure(0, weight=1)
        canvases_frame.grid_columnconfigure(1, weight=1)

        # Painel lateral
        status_frame = ttk.LabelFrame(right_panel, text="Status")
        status_frame.pack(fill="x", padx=4, pady=4)

        self.status_label = ttk.Label(status_frame, text="", justify="left")
        self.status_label.pack(anchor="w", padx=8, pady=8)

        help_frame = ttk.LabelFrame(right_panel, text="Controles")
        help_frame.pack(fill="x", padx=4, pady=4)

        help_text = (
            "• Clique esquerdo: adicionar ponto\n"
            "• Botão direito + arrastar: mover ponto\n"
            "• Mouse wheel: zoom\n"
            "• Primeiro marque a imagem original\n"
            "• Depois clique em 'Ir para Target'\n"
            "• A target deve ter o mesmo número de pontos"
        )
        ttk.Label(help_frame, text=help_text, justify="left").pack(anchor="w", padx=8, pady=8)

        pairs_frame = ttk.LabelFrame(right_panel, text="Pares de Pontos")
        pairs_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.pairs_text = tk.Text(pairs_frame, wrap="none", width=40)
        self.pairs_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)

        pairs_scroll = ttk.Scrollbar(pairs_frame, orient="vertical", command=self.pairs_text.yview)
        pairs_scroll.pack(side="right", fill="y", pady=6, padx=(0, 6))
        self.pairs_text.configure(yscrollcommand=pairs_scroll.set)

    # -----------------------------------------------------
    # Carregamento de imagens
    # -----------------------------------------------------
    def load_original(self):
        path = filedialog.askopenfilename(
            title="Selecione a imagem original",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos os arquivos", "*.*")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem original.")
            return

        self.original_path = path
        self.original_img = img
        self.original_canvas.set_image_bgr(img)
        self.result_img = None
        self.result_canvas.clear_all()
        self.result_canvas.set_title("Resultado (Original projetada sobre Target)")
        self._update_status()

    def load_target(self):
        path = filedialog.askopenfilename(
            title="Selecione a imagem target",
            filetypes=[("Imagens", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("Todos os arquivos", "*.*")]
        )
        if not path:
            return

        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", "Não foi possível carregar a imagem target.")
            return

        self.target_path = path
        self.target_img = img
        self.target_canvas.set_image_bgr(img)
        self.result_img = None
        self.result_canvas.clear_all()
        self._update_status()

    # -----------------------------------------------------
    # Eventos de clique / arraste
    # -----------------------------------------------------
    def on_canvas_click(self, canvas_obj, x, y):
        max_n = self._get_max_points()

        if canvas_obj == self.original_canvas:
            if self.phase != "original":
                messagebox.showwarning("Fase atual", "Você já está na fase de marcação da imagem target.")
                return

            pts = self.original_canvas.get_points()
            if len(pts) >= max_n:
                messagebox.showwarning("Limite atingido", f"Você atingiu o máximo de pontos: {max_n}.")
                return

            pts.append((x, y))
            self.original_canvas.set_points(pts)

        elif canvas_obj == self.target_canvas:
            if self.phase != "target":
                messagebox.showwarning("Fase atual", "Primeiro conclua a marcação da imagem original.")
                return

            src_pts = self.original_canvas.get_points()
            dst_pts = self.target_canvas.get_points()

            if len(src_pts) < 4:
                messagebox.showwarning("Pontos insuficientes", "A imagem original precisa de pelo menos 4 pontos.")
                return

            if len(dst_pts) >= len(src_pts):
                messagebox.showwarning("Quantidade completa", "A imagem target já possui o mesmo número de pontos da original.")
                return

            dst_pts.append((x, y))
            self.target_canvas.set_points(dst_pts)

        self._refresh_pairs_panel()
        self._update_status()

    def on_canvas_drag(self, canvas_obj):
        self._refresh_pairs_panel()
        self._update_status()

    # -----------------------------------------------------
    # Lógica de fases
    # -----------------------------------------------------
    def go_to_target_phase(self):
        src_pts = self.original_canvas.get_points()
        if len(src_pts) < 4:
            messagebox.showwarning("Pontos insuficientes", "Marque pelo menos 4 pontos na imagem original antes de ir para a target.")
            return

        self.phase = "target"
        self._update_status()

    # -----------------------------------------------------
    # Gestão de pontos
    # -----------------------------------------------------
    def undo_last(self, which):
        if which == "original":
            pts = self.original_canvas.get_points()
            if pts:
                pts.pop()
                self.original_canvas.set_points(pts)

                # Se target tiver mais pontos que original, corta o excedente
                dst_pts = self.target_canvas.get_points()
                if len(dst_pts) > len(pts):
                    dst_pts = dst_pts[:len(pts)]
                    self.target_canvas.set_points(dst_pts)

                if len(pts) < 4:
                    self.phase = "original"

        elif which == "target":
            pts = self.target_canvas.get_points()
            if pts:
                pts.pop()
                self.target_canvas.set_points(pts)

        self._refresh_pairs_panel()
        self._update_status()

    def clear_points(self, which):
        if which == "original":
            self.original_canvas.clear_all()
            self.target_canvas.clear_all()
            self.phase = "original"
        elif which == "target":
            self.target_canvas.clear_all()

        self.result_img = None
        self.result_canvas.clear_all()
        self._refresh_pairs_panel()
        self._update_status()

    def clear_all_points(self):
        self.original_canvas.clear_all()
        self.target_canvas.clear_all()
        self.result_canvas.clear_all()
        self.result_img = None
        self.phase = "original"
        self._refresh_pairs_panel()
        self._update_status()

    # -----------------------------------------------------
    # Homografia
    # -----------------------------------------------------
    def calculate_homography(self):
        if self.original_img is None or self.target_img is None:
            messagebox.showwarning("Imagens ausentes", "Carregue a imagem original e a target.")
            return

        src_pts = self.original_canvas.get_points()
        dst_pts = self.target_canvas.get_points()

        if len(src_pts) < 4:
            messagebox.showwarning("Pontos insuficientes", "A imagem original precisa ter pelo menos 4 pontos.")
            return

        if len(dst_pts) < 4:
            messagebox.showwarning("Pontos insuficientes", "A imagem target precisa ter pelo menos 4 pontos.")
            return

        if len(src_pts) != len(dst_pts):
            messagebox.showwarning("Quantidade incompatível", "A imagem original e a target devem ter o mesmo número de pontos.")
            return

        src = np.array(src_pts, dtype=np.float32)
        dst = np.array(dst_pts, dtype=np.float32)

        # Usa RANSAC para maior robustez
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            messagebox.showerror("Falha", "Não foi possível calcular a homografia.")
            return

        h_t, w_t = self.target_img.shape[:2]

        warped = cv2.warpPerspective(self.original_img, H, (w_t, h_t))

        # Máscara para sobreposição
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask_valid = gray > 0

        overlay = self.target_img.copy()
        alpha = 0.9

        blended = overlay.copy()
        blended[mask_valid] = cv2.addWeighted(
            overlay[mask_valid], 1 - alpha,
            warped[mask_valid], alpha,
            0
        )

        # Desenha pontos da target por cima no resultado
        for i, (x, y) in enumerate(dst_pts, start=1):
            cv2.circle(blended, (int(round(x)), int(round(y))), 6, (255, 180, 0), -1)
            cv2.putText(
                blended, str(i), (int(round(x)) + 8, int(round(y)) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 180, 0), 2, cv2.LINE_AA
            )

        self.result_img = blended
        self.result_canvas.set_image_bgr(self.result_img)
        self.result_canvas.set_points([])

        # Opcional: informa quantos inliers o RANSAC manteve
        inliers = int(mask.sum()) if mask is not None else len(src_pts)
        total = len(src_pts)

        self.result_canvas.set_title(
            f"Resultado (Original projetada sobre Target) | Inliers RANSAC: {inliers}/{total}"
        )

        self._update_status(extra_msg=f"Homografia calculada com sucesso. Inliers: {inliers}/{total}")

    # -----------------------------------------------------
    # Salvamento JSON
    # -----------------------------------------------------
    def save_points_json(self):
        src_pts = self.original_canvas.get_points()
        dst_pts = self.target_canvas.get_points()

        if not src_pts and not dst_pts:
            messagebox.showwarning("Sem dados", "Não há pontos para salvar.")
            return

        data = {
            "imagem_original": self.original_path,
            "imagem_target": self.target_path,
            "max_pontos_configurado": self._get_max_points(),
            "fase_atual": self.phase,
            "pontos_original": [
                {"id": i + 1, "x": float(x), "y": float(y)}
                for i, (x, y) in enumerate(src_pts)
            ],
            "pontos_target": [
                {"id": i + 1, "x": float(x), "y": float(y)}
                for i, (x, y) in enumerate(dst_pts)
            ]
        }

        path = filedialog.asksaveasfilename(
            title="Salvar pontos em JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            messagebox.showinfo("Sucesso", "Pontos salvos com sucesso em JSON.")
        except Exception as e:
            messagebox.showerror("Erro ao salvar", str(e))

    # -----------------------------------------------------
    # Utilidades
    # -----------------------------------------------------
    def _get_max_points(self):
        try:
            n = int(self.max_points_var.get())
            return max(4, n)
        except Exception:
            return 4

    def _refresh_pairs_panel(self):
        self.pairs_text.delete("1.0", tk.END)

        src_pts = self.original_canvas.get_points()
        dst_pts = self.target_canvas.get_points()

        total_pairs = max(len(src_pts), len(dst_pts))

        if total_pairs == 0:
            self.pairs_text.insert(tk.END, "Nenhum ponto marcado ainda.\n")
            return

        for i in range(total_pairs):
            self.pairs_text.insert(tk.END, f"Ponto {i+1}\n")
            if i < len(src_pts):
                x, y = src_pts[i]
                self.pairs_text.insert(tk.END, f"  Original: ({x:.2f}, {y:.2f})\n")
            else:
                self.pairs_text.insert(tk.END, "  Original: —\n")

            if i < len(dst_pts):
                x, y = dst_pts[i]
                self.pairs_text.insert(tk.END, f"  Target  : ({x:.2f}, {y:.2f})\n")
            else:
                self.pairs_text.insert(tk.END, "  Target  : —\n")

            self.pairs_text.insert(tk.END, "\n")

    def _update_status(self, extra_msg=None):
        src_n = len(self.original_canvas.get_points())
        dst_n = len(self.target_canvas.get_points())
        max_n = self._get_max_points()

        if self.phase == "original":
            phase_text = "Marcando pontos na IMAGEM ORIGINAL"
        else:
            phase_text = "Marcando pontos na IMAGEM TARGET"

        original_loaded = "Sim" if self.original_img is not None else "Não"
        target_loaded = "Sim" if self.target_img is not None else "Não"

        status = (
            f"Fase atual: {phase_text}\n"
            f"Imagem original carregada: {original_loaded}\n"
            f"Imagem target carregada: {target_loaded}\n"
            f"Pontos na original: {src_n}\n"
            f"Pontos na target: {dst_n}\n"
            f"Máximo configurado (N): {max_n}\n"
        )

        if src_n >= 4 and self.phase == "original":
            status += "\nVocê já pode clicar em 'Ir para Target'.\n"

        if self.phase == "target":
            faltam = max(0, src_n - dst_n)
            status += f"\nFaltam {faltam} ponto(s) na target.\n"

        if extra_msg:
            status += f"\n{extra_msg}\n"

        self.status_label.config(text=status)
        self._refresh_pairs_panel()


if __name__ == "__main__":
    app = HomographyApp()
    app.mainloop()