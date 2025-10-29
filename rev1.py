import ezdxf
from pathlib import Path
from contextlib import redirect_stdout
import sys
import math
from collections import defaultdict, deque

EPS = 1e-6  # допуск совпадения координат
AXIS_NAMES = {0: "X", 1: "Y", 2: "Z"}

class Tee:
    """Пишет одновременно в несколько потоков (консоль + файл)."""
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# ---------- выбор лучшей плоскости для каждой полилинии ----------
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

# ---------- разбор DXF в список 2D-сегментов ----------
def parse_dxf_polylines_smart(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()
    segments = []  # список словарей: {"start":(x,y), "end":(x,y), "plane":"XY"}

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

# ---------- утилиты для склейки ----------
def quantize_point(p, tol=EPS):
    """Квантование точки для словаря соответствий (устойчиво к шуму)."""
    return (round(p[0] / tol) * tol, round(p[1] / tol) * tol)

def dist2(p, q):
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return dx*dx + dy*dy

# ---------- склейка сегментов в максимальные цепочки ----------
def build_chains(segments):
    """
    Возвращает список цепочек; каждая цепочка — это список сегментов (словари),
    ориентированных так, чтобы конец предыдущего совпадал с началом следующего.
    Совпадение по допуску EPS.
    """
    # Индекс концов: ключ — квантованная точка; значение — список (seg_id, endpoint)
    # endpoint: 0 -> start, 1 -> end
    point_index = defaultdict(list)
    for i, seg in enumerate(segments):
        point_index[quantize_point(seg["start"])].append((i, 0))
        point_index[quantize_point(seg["end"])].append((i, 1))

    visited = [False] * len(segments)
    chains = []

    for i in range(len(segments)):
        if visited[i]:
            continue

        # начнём новую цепочку с сегмента i
        seg = segments[i]
        visited[i] = True
        chain = deque([{"start": seg["start"], "end": seg["end"], "plane": seg["plane"]}])

        # расширяем вперёд (по end)
        curr_end = chain[-1]["end"]
        while True:
            key = quantize_point(curr_end)
            candidates = [c for c in point_index.get(key, []) if not visited[c[0]]]
            if not candidates:
                break
            # Выбираем ближайший по другому концу (меньше зазор угла не считаем — пожелай, добавим)
            best_j = None
            best_d2 = float("inf")
            best_flip = False
            for j, endpoint in candidates:
                s = segments[j]
                if endpoint == 0:   # совпало начало — оставим как есть
                    other = s["end"]
                    flip = False
                else:               # совпал конец — развернём
                    other = s["start"]
                    flip = True
                d2 = dist2(curr_end, other)  # как «насколько продолжает»
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

        # расширяем назад (по start)
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
                if endpoint == 1:   # совпал конец — оставим как есть (его start присоединим)
                    other = s["start"]
                    flip = False
                else:               # совпало начало — развернём
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

# ---------- упорядочивание цепочек с минимальными переходами ----------
def order_chains_min_hops(chains):
    """
    Жадный порядок цепочек: стартуем с самой «левой-низкой»,
    затем каждый раз берём ближайшую по расстоянию от текущего конца.
    При необходимости переворачиваем цепочку.
    Возвращает список (chain, flipped: bool) в порядке обхода.
    """
    if not chains:
        return []

    def chain_start(c): return c[0]["start"]
    def chain_end(c): return c[-1]["end"]

    remaining = set(range(len(chains)))
    # старт: по минимальному (x,y) старта для стабильности
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
                best_flip = True  # перевернём, чтобы ближе был его конец
        order.append((best_k, best_flip))
        # обновляем текущий конец
        if best_flip:
            curr_end = chain_start(chains[best_k])
        else:
            curr_end = chain_end(chains[best_k])
        remaining.remove(best_k)

    return order

# ---------- печать результата ----------
def print_chains(chains, chain_order):
    """
    Печатает сегменты в порядке chain_order.
    Между цепочками — пустая строка как разделитель (можно убрать/заменить).
    """
    for idx, flip in chain_order:
        chain = chains[idx]
        if flip:
            # переворачиваем цепочку (и каждый сегмент)
            chain = [{"start": s["end"], "end": s["start"], "plane": s["plane"]} for s in reversed(chain)]
        for s in chain:
            print(f"{s['start'][0]} {s['start'][1]}")
            print(f"{s['end'][0]} {s['end'][1]}")
        print("")  # разделитель между списками

# ---------- main ----------
if __name__ == "__main__":
    dxf_path = Path(r"D:\python\projects\DXF_to_CFTCV\sketch.dxf")

    segments = parse_dxf_polylines_smart(str(dxf_path))
    chains = build_chains(segments)
    chain_order = order_chains_min_hops(chains)

    out_path = dxf_path.with_suffix(".cftcv")
    with open(out_path, "w", encoding="utf-8") as f, redirect_stdout(Tee(sys.stdout, f)):
        print_chains(chains, chain_order)

    print(f"Сохранено в файл: {out_path}")
