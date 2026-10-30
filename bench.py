import math
import os
import time

import cv2
import numpy as np

from gauss import blur_cv, blur_native

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def best(f, repeat):
    t = []
    for _ in range(repeat):
        s = time.perf_counter()
        f()
        t.append(time.perf_counter() - s)
    return min(t)


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (255, 255, 255), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def psnr(a, b):
    mse = np.mean((a.astype(float) - b.astype(float)) ** 2)
    return 10 * math.log10(255 ** 2 / mse)


def plot(path, xs, series, xlabel):
    w, h, m = 640, 400, 60
    img = np.full((h, w, 3), 255, np.uint8)
    vals = [v for s in series.values() for v in s]
    lo, hi = math.log10(min(vals)) - 0.2, math.log10(max(vals)) + 0.2
    px = lambda i: int(m + i * (w - 2 * m) / (len(xs) - 1))
    py = lambda v: int(h - m - (math.log10(v) - lo) / (hi - lo) * (h - 2 * m))
    cv2.rectangle(img, (m, m // 2), (w - m // 2, h - m), (0, 0, 0), 1)
    for i, x in enumerate(xs):
        cv2.putText(img, str(x), (px(i) - 10, h - m + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    d = int(math.floor(lo))
    while d <= hi:
        if lo <= d <= hi:
            y = py(10 ** d)
            cv2.line(img, (m, y), (w - m // 2, y), (220, 220, 220), 1)
            cv2.putText(img, "1e%d s" % d, (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        d += 1
    cv2.putText(img, xlabel, (w // 2 - 80, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    colors = [(200, 80, 0), (0, 140, 0)]
    for n, ((name, ys), c) in enumerate(zip(series.items(), colors)):
        pts = [(px(i), py(v)) for i, v in enumerate(ys)]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, c, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, p, 4, c, -1)
        cv2.putText(img, name, (m + 10, m // 2 + 20 + 20 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)


def main():
    src = cv2.imread(os.path.join(HERE, "data", "input.png"))
    rng = np.random.default_rng(1)
    noisy = np.clip(src + rng.normal(0, 20, src.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(RES, "noisy.png"), noisy)
    log = open(os.path.join(RES, "bench.txt"), "w", encoding="utf-8")

    def out(s):
        print(s)
        log.write(s + "\n")

    out("image %dx%d" % (src.shape[1], src.shape[0]))

    out("\n[сравнение результатов, шум sigma=20]")
    out("k  sigma  max|diff|  mean|diff|  PSNR к исходнику")
    for k, s in [(3, 0.8), (5, 1.0), (5, 3.0), (9, 2.0), (15, 3.0), (15, 0.5)]:
        a, b = blur_native(noisy, k, s), blur_cv(noisy, k, s)
        d = np.abs(a.astype(int) - b.astype(int))
        out("%-2d %-6s %-10d %-11.4f %.2f" % (k, s, d.max(), d.mean(), psnr(a, src)))
    out("без фильтра, PSNR %.2f" % psnr(noisy, src))
    a, b = blur_native(src, 7, 2.0, 0.7), blur_cv(src, 7, 2.0, 0.7)
    out("sigma_x=2.0 sigma_y=0.7 k=7: max|diff| %d" % np.abs(a.astype(int) - b.astype(int)).max())

    out("\n[время от размера ядра, sigma=k/5, %dx%d]" % (src.shape[1], src.shape[0]))
    out("k   native, c   opencv, c   во сколько раз")
    ks = [3, 5, 7, 9, 11, 15]
    tn, tc = [], []
    for k in ks:
        s = k / 5
        tn.append(best(lambda: blur_native(src, k, s), 2))
        tc.append(best(lambda: blur_cv(src, k, s), 200))
        out("%-3d %-11.3f %-11.6f %.0f" % (k, tn[-1], tc[-1], tn[-1] / tc[-1]))
    plot(os.path.join(RES, "time_kernel.png"), ks, {"native": tn, "opencv": tc}, "kernel size k")

    out("\n[время от размера изображения, k=7, sigma=1.4]")
    out("размер     native, c   opencv, c   во сколько раз")
    widths = [80, 160, 320, 640]
    tn, tc = [], []
    for w in widths:
        img = cv2.resize(src, (w, w * src.shape[0] // src.shape[1]), interpolation=cv2.INTER_AREA if w < 320 else cv2.INTER_CUBIC)
        tn.append(best(lambda: blur_native(img, 7, 1.4), 2))
        tc.append(best(lambda: blur_cv(img, 7, 1.4), 200))
        out("%-10s %-11.3f %-11.6f %.0f" % ("%dx%d" % (img.shape[1], img.shape[0]), tn[-1], tc[-1], tn[-1] / tc[-1]))
    plot(os.path.join(RES, "time_size.png"), widths, {"native": tn, "opencv": tc}, "image width, px")

    cases = [(5, 1.0), (5, 3.0), (15, 3.0), (15, 0.5), (15, 8.0)]
    tiles = [label(src, "original"), label(noisy, "noise sigma=20")]
    for k, s in cases:
        r = blur_native(noisy, k, s)
        cv2.imwrite(os.path.join(RES, "k%d_s%s.png" % (k, s)), r)
        tiles.append(label(r, "k=%d sigma=%s" % (k, s)))
    tiles.append(label(blur_native(noisy, 7, 2.0, 0.7), "k=7 sx=2.0 sy=0.7"))
    cv2.imwrite(os.path.join(RES, "montage.png"), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
    log.close()


if __name__ == "__main__":
    main()
