import matplotlib.pyplot as plt
from pathlib import Path

# --- Чтение координат ---
path = Path(r'D:\python\projects\DXF_to_CFTCV\sketch.cftcv')

if not path.exists():
    raise FileNotFoundError(f"Файл не найден: {path}")

points = []
bad_lines = []
with path.open('r', encoding='utf-8', errors='replace') as f:
    for i, line in enumerate(f, start=1):
        s = line.strip()
        if not s:
            bad_lines.append(i)
            continue
        parts = s.replace(',', ' ').split()
        if len(parts) != 2:
            bad_lines.append(i)
            continue
        try:
            x, y = map(float, parts)
            points.append((x, y))
        except ValueError:
            bad_lines.append(i)

if not points:
    raise ValueError("Файл не содержит корректных координат (ни одной валидной строки).")
if len(points) < 2:
    raise ValueError("Недостаточно точек для построения (нужно минимум 2).")

if bad_lines:
    print(f"Пропущены строки (невалидный формат): {bad_lines[:10]}{' ...' if len(bad_lines) > 10 else ''}")

# --- Подготовка графика ---
fig, ax = plt.subplots()
all_x, all_y = zip(*points)
ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
ax.set_aspect('equal')
plt.title("Пошаговая отрисовка координат")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)

# Показываем пустой график
plt.ion()
plt.show()

# --- Пошаговая отрисовка ---
xdata = [points[0][0]]
ydata = [points[0][1]]
(line,) = ax.plot(xdata, ydata, 'b-')

for i in range(1, len(points)):
    start = points[i - 1]
    end = points[i]
    print(f"Отрезок {i}:")
    print(f"  Начало: {start}")
    print(f"  Конец:  {end}")
    input("Нажмите Enter для отрисовки следующего отрезка...")

    xdata.append(end[0])
    ydata.append(end[1])
    line.set_data(xdata, ydata)
    plt.draw()
    plt.pause(0.001)  # помогает обновить окно на некоторых системах

plt.ioff()
plt.show()
