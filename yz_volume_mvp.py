
# -*- coding: utf-8 -*-
# YZ Image Series to 3D Volume (Z, Y, X) MVP
# - Scans base_dir for subfolders each containing YZ JPG slices.
# - Sorts folders and images numerically; stacks into a 3D volume (Z, Y, X).
# - Saves volume (.npy) and exports orthogonal slices as PNG.
# - Optionally downsamples and exports a capped CSV of voxels for inspection.

import os, glob, re, argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def natural_key(s: str):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', os.path.basename(s))]

def list_ordered_images(base_dir: str):
    subdirs = [p for p in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(p)]
    subdirs = sorted(subdirs, key=natural_key)

    paths = []
    for d in subdirs:
        imgs = glob.glob(os.path.join(d, "*.jpg"))
        imgs += glob.glob(os.path.join(d, "*.png"))
        imgs = sorted(imgs, key=natural_key)
        paths.extend(imgs)

    root_imgs = glob.glob(os.path.join(base_dir, "*.jpg")) + glob.glob(os.path.join(base_dir, "*.png"))
    root_imgs = sorted(root_imgs, key=natural_key)
    paths.extend(root_imgs)

    if not paths:
        raise FileNotFoundError("No JPG/PNG images found under: " + base_dir)
    return paths

def load_image_gray(path: str):
    im = Image.open(path).convert("L")
    return np.array(im)  # (Z, Y)

def build_volume(paths, resize_to=None, crop_to=None):
    slices = []
    for p in paths:
        arr = load_image_gray(p)
        if resize_to:
            im = Image.fromarray(arr)
            im = im.resize((resize_to[1], resize_to[0]), resample=Image.NEAREST)
            arr = np.array(im)
        if crop_to:
            zt, yt = crop_to
            z0 = max(0, (arr.shape[0] - zt)//2); z1 = z0 + zt
            y0 = max(0, (arr.shape[1] - yt)//2); y1 = y0 + yt
            arr = arr[z0:z1, y0:y1]
        slices.append(arr)

    shapes = {a.shape for a in slices}
    if len(shapes) != 1:
        raise ValueError(f"Slice sizes are not uniform: {shapes}. Use --resize_to or --crop_to.")
    Z, Y = slices[0].shape
    X = len(slices)
    vol = np.stack(slices, axis=-1)  # (Z, Y, X)
    return vol

def export_orthogonal_slices(volume: np.ndarray, out_dir: str, x_indices=None, y_indices=None, z_indices=None):
    os.makedirs(out_dir, exist_ok=True)
    Z, Y, X = volume.shape

    if x_indices is None: x_indices = [X//2]
    if y_indices is None: y_indices = [Y//2]
    if z_indices is None: z_indices = [Z//2]

    def save_png(arr2d, path, title):
        plt.figure(figsize=(6, 5))
        plt.imshow(arr2d, aspect='auto', origin='upper')
        plt.colorbar(label="intensity")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    saved = []
    for xi in x_indices:
        yz = volume[:, :, int(xi)]
        path = os.path.join(out_dir, f"slice_YZ_x{int(xi)}.png")
        save_png(yz, path, f"YZ @ X={xi}")
        saved.append(path)

    for yi in y_indices:
        zx = volume[:, int(yi), :]
        path = os.path.join(out_dir, f"slice_ZX_y{int(yi)}.png")
        save_png(zx, path, f"ZX @ Y={yi}")
        saved.append(path)

    for zi in z_indices:
        xy = volume[int(zi), :, :]
        path = os.path.join(out_dir, f"slice_XY_z{int(zi)}.png")
        save_png(xy, path, f"XY @ Z={zi}")
        saved.append(path)

    return saved

def export_downsampled_csv(volume: np.ndarray, out_csv: str, step_z=2, step_y=2, step_x=2, max_points=200000):
    vol = volume[::step_z, ::step_y, ::step_x]
    Z, Y, X = vol.shape
    zz, yy, xx = np.meshgrid(np.arange(Z), np.arange(Y), np.arange(X), indexing="ij")
    vals = vol.ravel()
    if vals.size > max_points:
        idx = np.linspace(0, vals.size-1, max_points, dtype=int)
        zz = zz.ravel()[idx]
        yy = yy.ravel()[idx]
        xx = xx.ravel()[idx]
        vals = vals[idx]
    else:
        zz = zz.ravel(); yy = yy.ravel(); xx = xx.ravel()

    import pandas as pd
    df = pd.DataFrame({
        "Z_index": zz.astype(int),
        "Y_index": yy.astype(int),
        "X_index": xx.astype(int),
        "intensity": vals.astype(int)
    })
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv, vol.shape

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build 3D volume (Z,Y,X) from YZ image series and export slices.")
    parser.add_argument("--base_dir", default="yz_images", help="Root directory containing YZ images in subfolders")
    parser.add_argument("--out_dir", default="yz_volume_out", help="Output directory for slices and artifacts")
    parser.add_argument("--out_npy", default="volume_zyx.npy", help="Output .npy filename (saved inside out_dir)")
    parser.add_argument("--resize_to", default="", help="Target size 'ZxY' e.g. '256x512' (optional)")
    parser.add_argument("--crop_to", default="", help="Center-crop to size 'ZxY' e.g. '256x512' (optional)")
    parser.add_argument("--x_indices", default="", help="Comma list of X indices for YZ slices, e.g., '10,20,30'")
    parser.add_argument("--y_indices", default="", help="Comma list of Y indices for ZX slices")
    parser.add_argument("--z_indices", default="", help="Comma list of Z indices for XY slices")
    parser.add_argument("--csv_out", default="", help="Optional path to export downsampled CSV")
    parser.add_argument("--csv_step", default="2x2x2", help="Downsample steps 'ZxYxX' for CSV (default 2x2x2)")
    parser.add_argument("--csv_max_points", type=int, default=200000, help="Max rows in CSV")
    args = parser.parse_args()

    img_paths = list_ordered_images(args.base_dir)
    print(f"[INFO] Found {len(img_paths)} images. Building volume...")

    resize_to = tuple(map(int, args.resize_to.lower().split('x'))) if args.resize_to else None
    crop_to = tuple(map(int, args.crop_to.lower().split('x'))) if args.crop_to else None
    vol = build_volume(img_paths, resize_to=resize_to, crop_to=crop_to)
    Z, Y, X = vol.shape
    print(f"[OK] Volume shape: (Z={Z}, Y={Y}, X={X})")

    os.makedirs(args.out_dir, exist_ok=True)
    npy_path = os.path.join(args.out_dir, args.out_npy)
    np.save(npy_path, vol)
    print("[OK] Saved volume to:", npy_path)

    def parse_indices(s, limit):
        if not s: return None
        out = []
        for tok in s.split(","):
            tok = tok.strip()
            if not tok: continue
            try:
                i = int(tok)
                if 0 <= i < limit:
                    out.append(i)
            except:
                pass
        return out or None

    xi = parse_indices(args.x_indices, X)
    yi = parse_indices(args.y_indices, Y)
    zi = parse_indices(args.z_indices, Z)

    saved_pngs = export_orthogonal_slices(vol, args.out_dir, x_indices=xi, y_indices=yi, z_indices=zi)
    print("[OK] Saved slices:")
    for p in saved_pngs:
        print(" -", p)

    if args.csv_out:
        step = tuple(map(int, args.csv_step.lower().split('x')))
        csv_path, ds_shape = export_downsampled_csv(vol, args.csv_out, step_z=step[0], step_y=step[1], step_x=step[2], max_points=args.csv_max_points)
        print(f"[OK] Exported downsampled CSV with shape={ds_shape} to:", csv_path)
