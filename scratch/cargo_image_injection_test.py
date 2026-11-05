#!/usr/bin/env python3
"""
scratch/cargo_image_injection_test.py

「貨物の像による見かけの流れの低下」の検証（画像注入テスト）。

物理的な狙い:
  実験で見える貨物の縁での |u| の低下が、本当の流体力学的な効果なのか、
  それとも GFP 画像に写り込んだ貨物の像（MT テクスチャのない円盤が、流れと違う動き方をする）が
  RAFT の推定を縁で引きずる「見かけの」低下なのかを切り分ける。
手順:
  1. 実際の貨物の像（半径 Rc の円盤、縁を 3 px でなめらかに落とす）を同じフレームの GFP 画像から切り出す。
  2. 貨物がない背景の場所に、フレーム i と i+1 で貨物の実変位をつけて貼り付ける（流れ場は変えない）。
  3. 元画像と注入画像の両方について、本番と同じ RAFT（finetune epoch 5）で流れを推定する。
  4. 注入位置まわりで |u_inj|(r)/|u_orig|(r) を求める。ここで r は縁からの距離。
     1 より小さければ、その分だけ像による見かけの低下になる。
出力: figure/cargo_image_injection/
"""
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate_virtual_tracer_flow_profile import BEADS_INFO, TARGET_DIRS, get_bead_info_from_dir, SCALE_UM_PER_PX

CUR = Path(__file__).resolve().parents[1]
OUT = CUR / 'figure' / 'cargo_image_injection'
OUT.mkdir(parents=True, exist_ok=True)
MODEL = '/home/sasaki/opticalflow-activenematics/raft_finetune/model/raft_finetune_epoch_5.pth'

CROP = 1024
N_FRAMES = 6        # 動画ごとのフレーム対の数
N_SITES = 3         # フレームごとの注入位置の数
TAPER = 3.0         # px
MAX_R_UM = 35.0
BIN_UM = 0.5
bins_px = np.arange(0, MAX_R_UM + BIN_UM, BIN_UM) / SCALE_UM_PER_PX
NB = len(bins_px) - 1

dev = torch.device('cuda')
weights = Raft_Large_Weights.DEFAULT
model = raft_large(weights=weights, progress=False)
model.load_state_dict(torch.load(MODEL, map_location=dev))
model.to(dev).eval()
tfm = weights.transforms()


@torch.no_grad()
def raft(a, b):
    t1 = torch.from_numpy(np.repeat(a[None], 3, 0)).unsqueeze(0).to(dev)
    t2 = torch.from_numpy(np.repeat(b[None], 3, 0)).unsqueeze(0).to(dev)
    t1, t2 = tfm(t1, t2)
    with torch.autocast(device_type='cuda'):
        f = model(t1, t2)[-1][0].float()
    return f.cpu().numpy()


def load(p):
    return np.array(Image.open(p).convert('L'))


def paste(img, src, sx, sy, tx, ty, R):
    """src の (sx,sy) を中心とする半径 R の円盤を img の (tx,ty) に貼る（サブピクセルは双一次補間）。"""
    from scipy.ndimage import map_coordinates
    h = int(np.ceil(R + TAPER)) + 2
    iy0, ix0 = int(round(ty)), int(round(tx))
    yy, xx = np.mgrid[iy0 - h:iy0 + h + 1, ix0 - h:ix0 + h + 1]
    d = np.hypot(xx - tx, yy - ty)
    alpha = np.clip((R + TAPER - d) / TAPER, 0, 1)
    vals = map_coordinates(src.astype(np.float32), [yy - ty + sy, xx - tx + sx], order=1, mode='nearest')
    out = img.astype(np.float32).copy()
    sl = (slice(iy0 - h, iy0 + h + 1), slice(ix0 - h, ix0 + h + 1))
    out[sl] = alpha * vals + (1 - alpha) * out[sl]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def radial(sp, cx, cy, R):
    yy, xx = np.mgrid[0:sp.shape[0], 0:sp.shape[1]]
    r = np.hypot(xx - cx, yy - cy) - R
    m = (r >= 0) & (r < bins_px[-1])
    b = np.digitize(r[m], bins_px) - 1
    return np.bincount(b, sp[m], NB), np.bincount(b, minlength=NB)


def run_video(d):
    bname, info = get_bead_info_from_dir(d)
    R = info['rc_px']
    d = Path(d)
    tr = pd.read_csv(d / 'beads_tracks.csv')
    imgs = sorted((d / 'GFP').glob('*.tif'))
    rng = np.random.default_rng(abs(hash(d.name)) % 2**32)
    # 次フレームでも追跡されている貨物があるフレーム
    pairs = []
    for f, g in tr.groupby('frame'):
        nxt = tr[tr.frame == f + 1]
        for _, row in g.iterrows():
            m = nxt[nxt.particle == row.particle]
            if len(m) and f + 1 < len(imgs):
                pairs.append((int(f), row.x, row.y, m.x.iloc[0], m.y.iloc[0]))
                break
    if not pairs:
        return None
    sel = [pairs[i] for i in np.linspace(0, len(pairs) - 1, min(N_FRAMES, len(pairs)), dtype=int)]
    acc = dict(s_orig=np.zeros(NB), s_inj=np.zeros(NB), s_err=np.zeros(NB), n=np.zeros(NB))
    for f, x0, y0, x1, y1 in sel:
        a, b = load(imgs[f]), load(imgs[f + 1])
        H, W = a.shape
        if not (h_ok := (H >= CROP and W >= CROP)):
            return None
        cargos = tr[tr.frame == f][['x', 'y']].values
        n_ok, tries = 0, 0
        while n_ok < N_SITES and tries < 500:
            tries += 1
            tx = rng.uniform(CROP / 2 + 4, W - CROP / 2 - 4)
            ty = rng.uniform(CROP / 2 + 4, H - CROP / 2 - 4)
            if len(cargos) and np.min(np.hypot(cargos[:, 0] - tx, cargos[:, 1] - ty)) < 2 * R + bins_px[-1] + 20:
                continue
            ox, oy = int(tx) - CROP // 2, int(ty) - CROP // 2
            ca, cb = a[oy:oy + CROP, ox:ox + CROP], b[oy:oy + CROP, ox:ox + CROP]
            cx, cy = tx - ox, ty - oy
            ia = paste(ca, a, x0, y0, cx, cy, R)
            ib = paste(cb, b, x1, y1, cx + (x1 - x0), cy + (y1 - y0), R)
            fo, fi = raft(ca, cb), raft(ia, ib)
            so, si = np.hypot(*fo), np.hypot(*fi)
            se = np.hypot(*(fi - fo))
            for k, sp in [('s_orig', so), ('s_inj', si), ('s_err', se)]:
                s, n = radial(sp, cx, cy, R)
                acc[k] += s
            acc['n'] += n
            n_ok += 1
    acc.update(bead_name=bname, exp_dir=d.name, radius_um=info['radius_um'])
    print(f'[{bname}] {d.name}: edge ratio = {acc["s_inj"][0] / acc["s_orig"][0]:.3f}', flush=True)
    return acc


def main():
    res = [r for r in (run_video(d) for d in TARGET_DIRS) if r is not None]
    with open(CUR / 'scratch' / 'cargo_image_injection_cache.pkl', 'wb') as f:
        pickle.dump(res, f)


if __name__ == '__main__':
    main()
