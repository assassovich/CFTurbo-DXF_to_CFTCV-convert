#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np
import os
import math
from collections import defaultdict, deque

# Импорт логики конвертации
EPS = 1e-6
AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}

def best_axes(points3d):
    n = len(points3d)
    xs, ys, zs = zip(*points3d)
    cols = [xs, ys, zs]

    def zero_ratio(col):
        return sum(1 for v in col if abs(v) <= EPS) / n

    def spread(col):
        return (max(col) - min(col)) if n > 1 else 0.0

    zr = [zero_ratio(c) for c in cols]
    sp = [spread(c) for c in cols]

    pairs = [(0, 1), (0, 2), (1, 2)]
    scored = []
    for a, b in pairs:
        score = (zr[a] + zr[b], -(sp[a] + sp[b]))
        scored.append((score, (a, b)))
    scored.sort()
    return scored[0][1]

def parse_dxf_polylines_smart(file_path):
    try:
        import ezdxf
    except ImportError:
        raise ImportError("Требуется библиотека ezdxf. Установите: pip install ezdxf")
    
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    segments = []

    for e in msp:
        points3d = []
        closed = False

        if e.dxftype() == "POLYLINE":
            points3d = [(v.dxf.location.x, v.dxf.location.y, v.dxf.location.z) for v in e.vertices]
            closed = bool(e.is_closed)

        elif e.dxftype() == "LWPOLYLINE":
            elev = float(getattr(e.dxf, "elevation", 0.0) or 0.0)
            pts2d = [(x, y) for x, y in e.get_points("xy")]
            points3d = [(x, y, elev) for x, y in pts2d]
            closed = bool(e.is_closed)
        else:
            continue

        if len(points3d) < 2:
            continue

        a, b = best_axes(points3d)
        plane = f"{AXIS_NAMES[a]}{AXIS_NAMES[b]}"
        pts2d = [(p[a], p[b]) for p in points3d]

        for i in range(len(pts2d) - 1):
            segments.append({"start": pts2d[i], "end": pts2d[i + 1], "plane": plane})
        if closed:
            segments.append({"start": pts2d[-1], "end": pts2d[0], "plane": plane})

    return segments

def quantize_point(p, tol=EPS):
    return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

def dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx*dx + dy*dy

def build_chains(segments):
    point_index = defaultdict(list)
    for i, seg in enumerate(segments):
        point_index[quantize_point(seg["start"])].append((i, 0))
        point_index[quantize_point(seg["end"])].append((i, 1))

    visited = [False] * len(segments)
    chains = []

    for i in range(len(segments)):
        if visited[i]:
            continue

        seg = segments[i]
        visited[i] = True
        chain = deque([{"start": seg["start"], "end": seg["end"], "plane": seg["plane"]}])

        curr_end = chain[-1]["end"]
        while True:
            key = quantize_point(curr_end)
            candidates = [c for c in point_index.get(key, []) if not visited[c[0]]]
            if not candidates:
                break
            best_j = None
            best_d2 = float("inf")
            best_flip = False
            for j, endpoint in candidates:
                s = segments[j]
                if endpoint == 0:
                    other = s["end"]
                    flip = False
                else:
                    other = s["start"]
                    flip = True
                d2 = dist2(curr_end, other)
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
                    best_flip = flip
            visited[best_j] = True
            s = segments[best_j]
            if best_flip:
                s = {"start": s["end"], "end": s["start"], "plane": s["plane"]}
            chain.append(s)
            curr_end = s["end"]

        curr_start = chain[0]["start"]
        while True:
            key = quantize_point(curr_start)
            candidates = [c for c in point_index.get(key, []) if not visited[c[0]]]
            if not candidates:
                break
            best_j = None
            best_d2 = float("inf")
            best_flip = False
            for j, endpoint in candidates:
                s = segments[j]
                if endpoint == 1:
                    other = s["start"]
                    flip = False
                else:
                    other = s["end"]
                    flip = True
                d2 = dist2(curr_start, other)
                if d2 < best_d2:
                    best_d2 = d2
                    best_j = j
                    best_flip = flip
            visited[best_j] = True
            s = segments[best_j]
            if best_flip:
                s = {"start": s["end"], "end": s["start"], "plane": s["plane"]}
            chain.appendleft(s)
            curr_start = s["start"]

        chains.append(list(chain))

    return chains

def order_chains_min_hops(chains):
    if not chains:
        return []

    def chain_start(c): return c[0]["start"]
    def chain_end(c): return c[-1]["end"]

    remaining = set(range(len(chains)))
    start_idx = min(remaining, key=lambda k: (chain_start(chains[k])[0], chain_start(chains[k])[1]))
    order = []
    curr_end = chain_end(chains[start_idx])
    order.append((start_idx, False))
    remaining.remove(start_idx)

    while remaining:
        best_k = None
        best_flip = False
        best_d2 = float("inf")
        for k in remaining:
            c = chains[k]
            d_start = dist2(curr_end, chain_start(c))
            d_end   = dist2(curr_end, chain_end(c))
            if d_start < best_d2:
                best_d2 = d_start
                best_k = k
                best_flip = False
            if d_end < best_d2:
                best_d2 = d_end
                best_k = k
                best_flip = True
        order.append((best_k, best_flip))
        if best_flip:
            curr_end = chain_start(chains[best_k])
        else:
            curr_end = chain_end(chains[best_k])
        remaining.remove(best_k)

    return order

class DenseGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("DXF → CFTCV")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Данные для работы
        self.segments = []
        self.chains = []
        self.chain_order = []
        self.current_points = []
        self.rotation_angle = 0
        self.diagonal_flip = False
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.manual_zoom = False
        self.auto_scaled = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Основной контейнер - горизонтальное разделение
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Левая панель с кнопками (компактная)
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(5, 2), pady=5)
        
        # Файловые операции
        ttk.Label(control_frame, text="ФАЙЛ", font=("Arial", 8, "bold")).pack(pady=(0, 2))
        ttk.Button(control_frame, text="Открыть DXF", 
                  command=self.load_dxf_file, width=12).pack(pady=1)
        ttk.Button(control_frame, text="Конвертировать", 
                  command=self.convert_dxf, width=12).pack(pady=1)
        
        # Разделитель
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Трансформации
        ttk.Label(control_frame, text="ТРАНСФОРМАЦИИ", font=("Arial", 8, "bold")).pack(pady=(0, 2))
        ttk.Button(control_frame, text="Rotate 90°", 
                  command=self.rotate_points, width=12).pack(pady=1)
        ttk.Button(control_frame, text="Turn Over", 
                  command=self.flip_points, width=12).pack(pady=1)
        ttk.Button(control_frame, text="Сброс", 
                  command=self.reset_transform, width=12).pack(pady=1)
        
        # Разделитель
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Масштабирование
        ttk.Label(control_frame, text="МАСШТАБ", font=("Arial", 8, "bold")).pack(pady=(0, 2))
        
        zoom_frame = ttk.Frame(control_frame)
        zoom_frame.pack(pady=1)
        ttk.Button(zoom_frame, text="+", command=self.zoom_in, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_frame, text="-", command=self.zoom_out, width=3).pack(side=tk.LEFT, padx=1)
        
        ttk.Button(control_frame, text="Авто", 
                  command=self.auto_scale, width=12).pack(pady=1)
        
        # Разделитель
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Сохранение
        ttk.Label(control_frame, text="СОХРАНЕНИЕ", font=("Arial", 8, "bold")).pack(pady=(0, 2))
        ttk.Button(control_frame, text="Сохранить", 
                  command=self.save_cftcv, width=12).pack(pady=1)
        
        # Разделитель
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Информация
        ttk.Label(control_frame, text="СТАТУС", font=("Arial", 8, "bold")).pack(pady=(0, 2))
        
        self.info_frame = ttk.Frame(control_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True)
        
        self.info_text = tk.Text(self.info_frame, width=15, height=10, font=("Courier", 8))
        self.info_text.pack(fill=tk.BOTH, expand=True)
        
        # Правая панель с графиком (квадратная)
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(2, 5), pady=5)
        
        # Создание matplotlib фигуры (квадратная)
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # Canvas для matplotlib
        self.canvas = FigureCanvasTkAgg(self.fig, plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Привязка событий мыши
        self.canvas.mpl_connect('scroll_event', self.on_mouse_scroll)
        
        # Инициализация графика
        self.setup_plot()
        self.update_info()
        
    def setup_plot(self):
        """Настройка графика с квадрантами"""
        self.ax.clear()
        self.ax.set_xlim(-100, 100)
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_title('DXF → CFTCV', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Оси координат
        self.ax.axhline(y=0, color='black', linewidth=1, alpha=0.8)
        self.ax.axvline(x=0, color='black', linewidth=1, alpha=0.8)
        
        # Диагональная ось
        self.ax.plot([-100, 100], [-100, 100], 'r--', linewidth=1.5, alpha=0.7)
        
        # Квадранты (тонкие)
        quad_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        
        rect1 = patches.Rectangle((0, 0), 100, 100, linewidth=0, facecolor=quad_colors[0], alpha=0.1)
        self.ax.add_patch(rect1)
        rect2 = patches.Rectangle((-100, 0), 100, 100, linewidth=0, facecolor=quad_colors[1], alpha=0.1)
        self.ax.add_patch(rect2)
        rect3 = patches.Rectangle((-100, -100), 100, 100, linewidth=0, facecolor=quad_colors[2], alpha=0.1)
        self.ax.add_patch(rect3)
        rect4 = patches.Rectangle((0, -100), 100, 100, linewidth=0, facecolor=quad_colors[3], alpha=0.1)
        self.ax.add_patch(rect4)
        
        # Подписи квадрантов (компактные)
        self.ax.text(80, 80, 'I', ha='center', va='center', fontsize=16, alpha=0.5)
        self.ax.text(-80, 80, 'II', ha='center', va='center', fontsize=16, alpha=0.5)
        self.ax.text(-80, -80, 'III', ha='center', va='center', fontsize=16, alpha=0.5)
        self.ax.text(80, -80, 'IV', ha='center', va='center', fontsize=16, alpha=0.5)
        
        self.canvas.draw()
        
    def load_dxf_file(self):
        """Загрузка DXF файла"""
        file_path = filedialog.askopenfilename(
            title="Выберите DXF файл",
            filetypes=[("DXF files", "*.dxf"), ("All files", "*.*")]
        )
        if file_path:
            self.file_path = file_path
            self.update_info(f"Файл: {os.path.basename(file_path)}")
            
    def convert_dxf(self):
        """Конвертация DXF в CFTCV формат"""
        if not hasattr(self, 'file_path'):
            messagebox.showerror("Ошибка", "Сначала выберите DXF файл")
            return
            
        try:
            self.segments = parse_dxf_polylines_smart(self.file_path)
            if not self.segments:
                messagebox.showwarning("Предупреждение", "В файле не найдено полилиний")
                return
                
            self.chains = build_chains(self.segments)
            self.chain_order = order_chains_min_hops(self.chains)
            self.current_points = self.get_points_from_chains()
            
            # Сброс трансформаций
            self.rotation_angle = 0
            self.diagonal_flip = False
            self.zoom_factor = 1.0
            self.zoom_center_x = 0.0
            self.zoom_center_y = 0.0
            self.manual_zoom = False
            self.auto_scaled = False
            
            self.plot_points()
            self.update_info()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при конвертации: {str(e)}")
            
    def get_points_from_chains(self):
        """Получение точек из цепочек"""
        points = []
        for idx, flip in self.chain_order:
            chain = self.chains[idx]
            if flip:
                chain = [{"start": s["end"], "end": s["start"], "plane": s["plane"]} 
                        for s in reversed(chain)]
            for s in chain:
                points.append(s["start"])
                points.append(s["end"])
        return points
        
    def rotate_points(self):
        """Поворот точек на 90 градусов"""
        if not self.current_points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и конвертируйте DXF файл")
            return
            
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.plot_points()
        self.update_info()
        
    def flip_points(self):
        """Диагональное зеркальное отражение"""
        if not self.current_points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и конвертируйте DXF файл")
            return
            
        self.diagonal_flip = not self.diagonal_flip
        self.plot_points()
        self.update_info()
        
    def reset_transform(self):
        """Сброс всех трансформаций"""
        if not self.current_points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и конвертируйте DXF файл")
            return
            
        self.rotation_angle = 0
        self.diagonal_flip = False
        self.zoom_factor = 1.0
        self.zoom_center_x = 0.0
        self.zoom_center_y = 0.0
        self.manual_zoom = False
        self.auto_scaled = False
        self.plot_points()
        self.update_info()
        
    def zoom_in(self):
        """Увеличение масштаба"""
        if not self.current_points:
            return
        self.zoom_factor *= 1.2
        self.manual_zoom = True
        self.auto_scaled = False
        self.plot_points()
        self.update_info()
        
    def zoom_out(self):
        """Уменьшение масштаба"""
        if not self.current_points:
            return
        self.zoom_factor /= 1.2
        self.manual_zoom = True
        self.auto_scaled = False
        self.plot_points()
        self.update_info()
        
    def auto_scale(self):
        """Автоматическое масштабирование"""
        if not self.current_points:
            return
        self.auto_scaled = not self.auto_scaled
        self.manual_zoom = False
        self.plot_points()
        self.update_info()
        
    def on_mouse_scroll(self, event):
        """Обработка прокрутки колесика мыши"""
        if not self.current_points or event.inaxes != self.ax:
            return
            
        if event.button == 'up':
            self.zoom_factor *= 1.1
        elif event.button == 'down':
            self.zoom_factor /= 1.1
        
        self.zoom_center_x = event.xdata if event.xdata is not None else 0
        self.zoom_center_y = event.ydata if event.ydata is not None else 0
        
        self.manual_zoom = True
        self.auto_scaled = False
        self.plot_points()
        self.update_info()
        
    def transform_points(self, points):
        """Применение трансформаций к точкам"""
        transformed = []
        for x, y in points:
            # Поворот
            angle_rad = math.radians(self.rotation_angle)
            cos_a = math.cos(angle_rad)
            sin_a = math.sin(angle_rad)
            x_rot = x * cos_a - y * sin_a
            y_rot = x * sin_a + y * cos_a
            
            # Диагональное отражение
            if self.diagonal_flip:
                x_rot, y_rot = y_rot, x_rot
                
            transformed.append((x_rot, y_rot))
        return transformed
        
    def plot_points(self):
        """Отрисовка точек"""
        if not self.current_points:
            self.setup_plot()
            return
            
        self.ax.clear()
        
        transformed_points = self.transform_points(self.current_points)
        
        # Настройка осей
        if self.manual_zoom:
            base_range = 200
            scaled_range = base_range / self.zoom_factor
            x_min = self.zoom_center_x - scaled_range / 2
            x_max = self.zoom_center_x + scaled_range / 2
            y_min = self.zoom_center_y - scaled_range / 2
            y_max = self.zoom_center_y + scaled_range / 2
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        elif self.auto_scaled:
            x_coords = [p[0] for p in transformed_points]
            y_coords = [p[1] for p in transformed_points]
            margin = max(10, (max(x_coords) - min(x_coords)) * 0.1)
            x_min, x_max = min(x_coords) - margin, max(x_coords) + margin
            y_min, y_max = min(y_coords) - margin, max(y_coords) + margin
            x_range = x_max - x_min
            y_range = y_max - y_min
            max_range = max(x_range, y_range)
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            x_min = x_center - max_range / 2
            x_max = x_center + max_range / 2
            y_min = y_center - max_range / 2
            y_max = y_center + max_range / 2
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        else:
            self.ax.set_xlim(-100, 100)
            self.ax.set_ylim(-100, 100)
        
        # Оси координат
        self.ax.axhline(y=0, color='black', linewidth=1, alpha=0.8)
        self.ax.axvline(x=0, color='black', linewidth=1, alpha=0.8)
        
        # Диагональная ось
        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        self.ax.plot([x_min, x_max], [y_min, y_max], 'r--', linewidth=1.5, alpha=0.7)
        
        # Квадранты
        quad_colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
        
        if x_max > 0 and y_max > 0:
            rect1 = patches.Rectangle((0, 0), x_max, y_max, linewidth=0, facecolor=quad_colors[0], alpha=0.1)
            self.ax.add_patch(rect1)
        if x_min < 0 and y_max > 0:
            rect2 = patches.Rectangle((x_min, 0), -x_min, y_max, linewidth=0, facecolor=quad_colors[1], alpha=0.1)
            self.ax.add_patch(rect2)
        if x_min < 0 and y_min < 0:
            rect3 = patches.Rectangle((x_min, y_min), -x_min, -y_min, linewidth=0, facecolor=quad_colors[2], alpha=0.1)
            self.ax.add_patch(rect3)
        if x_max > 0 and y_min < 0:
            rect4 = patches.Rectangle((0, y_min), x_max, -y_min, linewidth=0, facecolor=quad_colors[3], alpha=0.1)
            self.ax.add_patch(rect4)
        
        # Отрисовка данных
        if transformed_points:
            for i in range(0, len(transformed_points), 2):
                if i + 1 < len(transformed_points):
                    self.ax.plot([transformed_points[i][0], transformed_points[i+1][0]], 
                               [transformed_points[i][1], transformed_points[i+1][1]], 
                               'b-', linewidth=2)
            
            # Начальная точка
            self.ax.plot(transformed_points[0][0], transformed_points[0][1], 
                       'ro', markersize=6)
        
        # Настройка осей
        self.ax.set_xlabel('X', fontsize=10)
        self.ax.set_ylabel('Y', fontsize=10)
        self.ax.set_title('DXF → CFTCV', fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        self.canvas.draw()
        
    def update_info(self, message=None):
        """Обновление информационной панели"""
        self.info_text.delete(1.0, tk.END)
        
        if message:
            self.info_text.insert(tk.END, f"{message}\n\n")
        
        if self.segments:
            info = f"Сегментов: {len(self.segments)}\n"
            info += f"Цепочек: {len(self.chains)}\n"
            info += f"Точек: {len(self.current_points)}\n\n"
            info += f"Поворот: {self.rotation_angle}°\n"
            info += f"Отражение: {'Да' if self.diagonal_flip else 'Нет'}\n"
            info += f"Масштаб: {self.zoom_factor:.1f}x\n\n"
            
            mode = "Ручной" if self.manual_zoom else "Авто" if self.auto_scaled else "Фиксир."
            info += f"Режим: {mode}"
            
            self.info_text.insert(tk.END, info)
        else:
            self.info_text.insert(tk.END, "Готов к работе\n\nВыберите DXF файл\nи нажмите\n'Конвертировать'")
        
    def save_cftcv(self):
        """Сохранение результата в CFTCV формат"""
        if not self.current_points:
            messagebox.showwarning("Предупреждение", "Сначала загрузите и конвертируйте DXF файл")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Сохранить CFTCV файл",
            defaultextension=".cftcv",
            filetypes=[("CFTCV files", "*.cftcv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                transformed_points = self.transform_points(self.current_points)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    for x, y in transformed_points:
                        f.write(f"{x} {y}\n")
                        
                messagebox.showinfo("Успех", f"Файл сохранен: {file_path}")
                self.update_info("Файл сохранен")
                
            except Exception as e:
                messagebox.showerror("Ошибка", f"Ошибка при сохранении: {str(e)}")

def main():
    root = tk.Tk()
    app = DenseGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
