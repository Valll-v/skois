import argparse
import math
from operator import mul

import cv2
import numpy as np


def kernel(k, sx, sy):
    r = k // 2
    kx = [math.exp(-(i * i) / (2 * sx * sx)) for i in range(-r, r + 1)]
    ky = [math.exp(-(j * j) / (2 * sy * sy)) for j in range(-r, r + 1)]
    ker = [[a * b for b in kx] for a in ky]
    total = sum(map(sum, ker))
    return [[v / total for v in row] for row in ker]


def reflect(i, n):
    # отражение без повтора крайнего пикселя, как BORDER_REFLECT_101 в OpenCV
    if i < 0:
        i = -i
    if i >= n:
        i = 2 * (n - 1) - i
    return i


def blur_channel(ch, ker):
    h, w = len(ch), len(ch[0])
    k = len(ker)
    r = k // 2
    cols = [reflect(x, w) for x in range(-r, w + r)]
    padded = [[ch[reflect(y, h)][c] for c in cols] for y in range(-r, h + r)]
    out = []
    for y in range(h):
        rows = padded[y:y + k]
        line = []
        for x in range(w):
            s = 0.0
            for kr, row in zip(ker, rows):
                s += sum(map(mul, kr, row[x:x + k]))
            line.append(min(255, max(0, int(s + 0.5))))
        out.append(line)
    return out


def blur_native(img, k, sx, sy=None):
    sy = sx if sy is None else sy
    ker = kernel(k, sx, sy)
    chans = [img[:, :, c].tolist() for c in range(img.shape[2])]
    res = [blur_channel(ch, ker) for ch in chans]
    return np.stack([np.array(c, dtype=np.uint8) for c in res], axis=2)


def blur_cv(img, k, sx, sy=None):
    sy = sx if sy is None else sy
    return cv2.GaussianBlur(img, (k, k), sigmaX=sx, sigmaY=sy)


def main():
    p = argparse.ArgumentParser(description="Фильтр Гаусса с переменным ядром")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("-k", type=int, default=5, help="размер ядра, нечётное")
    p.add_argument("-s", "--sigma", type=float, default=1.0, help="sigma по x")
    p.add_argument("--sigma-y", type=float, default=None, help="sigma по y, по умолчанию как по x")
    p.add_argument("--impl", choices=["native", "cv"], default="native")
    a = p.parse_args()
    if a.k < 3 or a.k % 2 == 0:
        p.error("k должно быть нечётным и не меньше 3")
    if a.sigma <= 0 or (a.sigma_y is not None and a.sigma_y <= 0):
        p.error("sigma должна быть больше нуля")
    img = cv2.imread(a.src)
    if img is None:
        p.error("не удалось прочитать " + a.src)
    if a.k // 2 >= min(img.shape[:2]):
        p.error("ядро слишком большое для изображения")
    f = blur_native if a.impl == "native" else blur_cv
    cv2.imwrite(a.dst, f(img, a.k, a.sigma, a.sigma_y))


if __name__ == "__main__":
    main()
