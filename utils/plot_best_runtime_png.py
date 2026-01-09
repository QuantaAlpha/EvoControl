#!/usr/bin/env python3

import argparse
import json
import math
import re
import struct
import zlib
from pathlib import Path


def sanitize_name(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = s.strip("._-")
    return s[:80] if s else "instance"


def load_instances(input_path: Path):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    # Prefer iteration_summary structure
    if isinstance(data, dict) and "instances" in data and "Ks" in data:
        Ks = data["Ks"]
        out = []
        for row in data["instances"]:
            name = row.get("instance") or "unknown"
            # Filter out system prompt-like entries defensively
            sname = name.strip().lower()
            if ("system_prompt" in sname) or ("system prompt" in sname):
                continue
            metrics = row.get("metrics", {})
            series = []
            for K in Ks:
                m = metrics.get(str(K)) or metrics.get(K)
                if not m:
                    continue
                b = m.get("best")
                if b is None:
                    continue
                try:
                    val = float(b)
                except Exception:
                    continue
                if math.isfinite(val):
                    series.append((K, val))
            if series:
                out.append({"name": name, "series": series})
        return out
    # Fallback: all_hist.json structure
    out = []
    for inst, info in data.items():
        sname = str(inst).strip().lower()
        if ("system_prompt" in sname) or ("system prompt" in sname):
            continue
        iter_map = info.get("iteration", {})
        points = []
        for k_str, v in iter_map.items():
            try:
                k = int(k_str)
            except Exception:
                continue
            rt = v.get("runtime")
            if rt is None:
                continue
            try:
                val = float(rt)
            except Exception:
                continue
            if math.isfinite(val):
                points.append((k, val))
        if points:
            points.sort(key=lambda x: x[0])
            # prefix minimum as best runtime per K
            best = None
            series = []
            j = 0
            Ks = sorted({k for k, _ in points})
            for K in Ks:
                while j < len(points) and points[j][0] <= K:
                    v = points[j][1]
                    best = v if best is None else (v if v < best else best)
                    j += 1
                if best is not None:
                    series.append((K, best))
            if series:
                out.append({"name": inst, "series": series})
    return out


class Canvas:
    def __init__(self, w: int, h: int, bg=(255, 255, 255, 255)):
        self.w = w
        self.h = h
        self.buf = bytearray(w * h * 4)
        r, g, b, a = bg
        for i in range(0, len(self.buf), 4):
            self.buf[i] = r
            self.buf[i + 1] = g
            self.buf[i + 2] = b
            self.buf[i + 3] = a

    def set_px(self, x: int, y: int, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 4
            self.buf[i : i + 4] = bytes(color)

    def line(self, x0, y0, x1, y1, color, thickness=1):
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        if dx > dy:
            err = dx // 2
            while x != x1:
                self._dot(x, y, color, thickness)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy // 2
            while y != y1:
                self._dot(x, y, color, thickness)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        self._dot(x, y, color, thickness)

    def rect(self, x0, y0, x1, y1, color):
        for x in range(min(x0, x1), max(x0, x1) + 1):
            self.set_px(x, y0, color)
            self.set_px(x, y1, color)
        for y in range(min(y0, y1), max(y0, y1) + 1):
            self.set_px(x0, y, color)
            self.set_px(x1, y, color)

    def _dot(self, x, y, color, t):
        r = max(0, t // 2)
        for yy in range(y - r, y + r + 1):
            for xx in range(x - r, x + r + 1):
                self.set_px(xx, yy, color)


# Minimal 5x7 font for digits and '.'
FONT_5x7 = {
    "0": [
        " ### ",
        "#   #",
        "#  ##",
        "# # #",
        "##  #",
        "#   #",
        " ### ",
    ],
    "1": [
        "  #  ",
        " ##  ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
        " ### ",
    ],
    "2": [
        " ### ",
        "#   #",
        "    #",
        "   # ",
        "  #  ",
        " #   ",
        "#####",
    ],
    "3": [
        " ### ",
        "    #",
        "    #",
        " ### ",
        "    #",
        "    #",
        " ### ",
    ],
    "4": [
        "   # ",
        "  ## ",
        " # # ",
        "#  # ",
        "#####",
        "   # ",
        "   # ",
    ],
    "5": [
        "#####",
        "#    ",
        "#    ",
        "#### ",
        "    #",
        "    #",
        "#### ",
    ],
    "6": [
        " ### ",
        "#    ",
        "#    ",
        "#### ",
        "#   #",
        "#   #",
        " ### ",
    ],
    "7": [
        "#####",
        "    #",
        "   # ",
        "  #  ",
        "  #  ",
        "  #  ",
        "  #  ",
    ],
    "8": [
        " ### ",
        "#   #",
        "#   #",
        " ### ",
        "#   #",
        "#   #",
        " ### ",
    ],
    "9": [
        " ### ",
        "#   #",
        "#   #",
        " ####",
        "    #",
        "    #",
        " ### ",
    ],
    ".": [
        "     ",
        "     ",
        "     ",
        "     ",
        "     ",
        " ### ",
        " ### ",
    ],
}


def draw_text(canvas: Canvas, x: int, y: int, text: str, color=(60, 60, 60, 255)):
    cx = x
    for ch in text:
        glyph = FONT_5x7.get(ch)
        if glyph is None:
            cx += 6
            continue
        for yy, row in enumerate(glyph):
            for xx, c in enumerate(row):
                if c != " ":
                    canvas.set_px(cx + xx, y + yy, color)
        cx += 6


def encode_png(canvas: Canvas) -> bytes:
    w, h = canvas.w, canvas.h
    # PNG signature
    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR
    ihdr = struct.pack(
        ">IIBBBBB",
        w,
        h,
        8,  # bit depth
        6,  # color type RGBA
        0,  # compression
        0,  # filter
        0,  # interlace
    )

    def pack(typ: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + typ + data + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    # IDAT
    # Each row: filter type 0 + rgba bytes
    rows = bytearray()
    stride = w * 4
    for y in range(h):
        rows.append(0)
        start = y * stride
        rows.extend(canvas.buf[start : start + stride])
    comp = zlib.compress(bytes(rows), level=9)
    # IEND
    return sig + pack(b"IHDR", ihdr) + pack(b"IDAT", comp) + pack(b"IEND", b"")


def draw_chart(inst_name: str, series, out_path: Path, width=800, height=420):
    if not series:
        return False
    xs = [k for k, _ in series]
    ys = [v for _, v in series]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax == xmin:
        xmax = xmin + 1
    if ymax == ymin:
        ymax = ymin + 1e-9
    ml, mr, mt, mb = 60, 20, 30, 40
    pw, ph = width - ml - mr, height - mt - mb

    def x2px(x):
        return ml + int((x - xmin) / (xmax - xmin) * pw + 0.5)

    def y2py(y):
        return height - mb - int((y - ymin) / (ymax - ymin) * ph + 0.5)

    c = Canvas(width, height)
    # Axes
    axis = (50, 50, 50, 255)
    c.line(ml, height - mb, width - mr, height - mb, axis)
    c.line(ml, mt, ml, height - mb, axis)
    # Grid + ticks + labels
    grid = (230, 230, 230, 255)
    for i in range(5):
        tx = xmin + (i / 4.0) * (xmax - xmin)
        px = x2px(tx)
        c.line(px, mt, px, height - mb, grid)
        draw_text(c, px - 12, height - mb + 8, str(int(round(tx))), (80, 80, 80, 255))
    for i in range(5):
        ty = ymin + (i / 4.0) * (ymax - ymin)
        py = y2py(ty)
        c.line(ml, py, width - mr, py, grid)
        draw_text(c, ml - 50, py - 3, f"{ty:.3f}", (80, 80, 80, 255))
    # Title
    draw_text(c, ml, 8, sanitize_name(inst_name), (30, 30, 30, 255))
    # Polyline
    blue = (31, 119, 180, 255)
    prev = None
    for k, v in series:
        x, y = x2px(k), y2py(v)
        if prev is not None:
            c.line(prev[0], prev[1], x, y, blue, thickness=2)
        prev = (x, y)
    # Save
    out_path.write_bytes(encode_png(c))
    return True


def _compute_prefix_min(points):
    pts = sorted(points, key=lambda x: x[0])
    best = None
    out = []
    j = 0
    ks = sorted({k for k, _ in pts})
    for k in ks:
        while j < len(pts) and pts[j][0] <= k:
            v = pts[j][1]
            best = v if best is None else (v if v < best else best)
            j += 1
        if best is not None:
            out.append((k, best))
    return out


def _infer_label(path: Path) -> str:
    stem = path.stem
    stem = stem.replace("_all_hist", "").replace("-all_hist", "")
    return stem


def _to_map(items, use_index: bool):
    m = {}
    for it in items:
        seq = it.get("series") or []
        if use_index:
            seq = [(i, y) for i, (_, y) in enumerate(seq)]
        m[str(it["name"])] = seq
    return m


def _plot_matplotlib(inst_map_list, labels, out_dir: Path, width: int, height: int):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    names = set()
    for m in inst_map_list:
        names.update(m.keys())
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        for idx, m in enumerate(inst_map_list):
            s = m.get(name)
            if not s:
                continue
            xs = [p[0] for p in s]
            ys = [p[1] for p in s]
            ax.plot(xs, ys, label=labels[idx])
        ax.set_title(str(name))
        ax.set_xlabel("iteration")
        ax.set_ylabel("best runtime")
        ax.legend()
        fig.tight_layout()
        outp = out_dir / f"{sanitize_name(str(name))}.png"
        fig.savefig(outp)
        plt.close(fig)
    return True


# Removed HTML/ECharts generation to comply with requirement of PNG-only output


def main():
    ap = argparse.ArgumentParser(description="绘制 best runtime，对比一个或两个方法，仅输出 PNG")
    ap.add_argument("inputs", nargs="+", help="一个或两个 all_hist/iteration_summary JSON 路径")
    ap.add_argument("--out_dir", default="static/best_runtime_png")
    ap.add_argument("--width", type=int, default=800)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--use_k_values", action="store_true")
    ap.add_argument("--xmode", nargs="*", choices=["index", "real"], help="为每个输入指定横轴模式(index/real)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.inputs][:2]
    if not paths:
        print("缺少输入")
        return
    labels = args.labels
    if not labels or len(labels) < len(paths):
        labels = [_infer_label(p) for p in paths]
    # 构造每个输入的横轴模式（强制真实 iteration）
    modes = ["real" for _ in paths]

    inst_map_list = []
    for i, p in enumerate(paths):
        items = load_instances(p)
        inst_map_list.append(_to_map(items, use_index=(modes[i] == "index")))

    ok = _plot_matplotlib(inst_map_list, labels, Path(args.out_dir), args.width, args.height)
    if ok:
        print(f"输出: {args.out_dir}")
    else:
        print("matplotlib 不可用，PNG输出失败")


if __name__ == "__main__":
    main()
