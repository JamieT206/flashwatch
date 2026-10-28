"""M3: a convolutional network on the map (DESIGN.md L1), as a drop-in model for eval.cv.

    python -m flashwatch.eval.cv --models B0 M1 M3 --leads week1 week2 --fast --tag m3

M1 and M2 see each grid cell on its own (plus a few hand-made neighbourhood features). M3 sees the whole
map at once: the same features, laid out as a picture per forecast date (one channel per feature, plus a
channel saying where there is a row), through a small U-Net that can learn how a drying patch spreads.
It is trained inside each cross-validation fold on the training years only, like the other models, so
leakage discipline is the fold's: features are the as-of features of eval.cv.fold_table.

Training: masked binary cross-entropy on cells with a row; the latest training year is held back for
early stopping and a Platt map (calibration), then the predictions for the test year are gathered back
to its rows. Needs PyTorch (requirements-dl.txt); runs on a GPU if there is one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from flashwatch.models.gbm import ID_COLUMNS, LABEL_PREFIXES


def _features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLUMNS and not c.startswith(LABEL_PREFIXES)
            and pd.api.types.is_numeric_dtype(df[c])]


class Grid:
    """Rows (issue, iy, ix) <-> arrays (issue, H, W) on the thinned grid."""

    def __init__(self, df: pd.DataFrame):
        self.thin = int(np.gcd.reduce(np.r_[df.iy.to_numpy(), df.ix.to_numpy()])) or 1
        self.h = int(df.iy.max()) // self.thin + 1
        self.w = int(df.ix.max()) // self.thin + 1

    def index(self, df: pd.DataFrame):
        issues = pd.DatetimeIndex(df.issue)
        days = issues.unique().sort_values()
        return days, days.get_indexer(issues), df.iy.to_numpy() // self.thin, df.ix.to_numpy() // self.thin

    def scatter(self, df, cols, fill=np.nan):
        days, n, y, x = self.index(df)
        a = np.full((len(days), len(cols), self.h, self.w), fill, np.float32)
        a[n, :, y, x] = df[cols].to_numpy(np.float32)
        return a, (n, y, x)


def _net(c_in: int, width: int = 64):
    import torch.nn as nn

    def block(a, b, d=1):
        return nn.Sequential(nn.Conv2d(a, b, 3, padding=d, dilation=d), nn.BatchNorm2d(b), nn.GELU(),
                             nn.Conv2d(b, b, 3, padding=d, dilation=d), nn.BatchNorm2d(b), nn.GELU())

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1, self.e2 = block(c_in, width), block(width, 2 * width)
            self.mid = block(2 * width, 2 * width, d=2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d1 = block(3 * width, width)
            self.out = nn.Conv2d(width, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            h, w = x.shape[-2:]
            ph, pw = (-h) % 2, (-w) % 2
            x = nn.functional.pad(x, (0, pw, 0, ph))
            a = self.e1(x)
            b = self.mid(self.e2(self.pool(a)))
            y = self.out(self.d1(__import__("torch").cat([a, self.up(b)], 1)))
            return y[..., :h, :w].squeeze(1)

    return UNet()


class CNN:
    name = "M3"

    def __init__(self, epochs: int = 25, lr: float = 1e-3, batch: int = 16, width: int = 64, seed: int = 0, **_):
        self.epochs, self.lr, self.batch, self.width, self.seed = epochs, lr, batch, width, seed

    def _inputs(self, df):
        a, idx = self.grid.scatter(df, self.cols)
        a = (a - self.mu[None, :, None, None]) / self.sd[None, :, None, None]
        present = np.isfinite(a[:, :1]).astype(np.float32)
        return np.concatenate([np.nan_to_num(a, nan=0.0), present], 1), idx

    def fit(self, df: pd.DataFrame, label: str):
        import torch
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.cols = _features(df)
        self.grid = Grid(df)
        x = df[self.cols].to_numpy(np.float64)
        self.mu, self.sd = np.nanmean(x, 0).astype(np.float32), (np.nanstd(x, 0) + 1e-6).astype(np.float32)
        val_year = int(df.year.max())
        parts = {}
        for name, d in (("train", df[df.year != val_year]), ("val", df[df.year == val_year])):
            xin, idx = self._inputs(d)
            y, _ = self.grid.scatter(d, [label])
            parts[name] = (torch.tensor(xin), torch.tensor(np.nan_to_num(y[:, 0], nan=0.0)),
                           torch.tensor(np.isfinite(y[:, 0]).astype(np.float32)), d, idx)
        self.net = _net(parts["train"][0].shape[1], self.width).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")
        xt, yt, mt = parts["train"][:3]
        best, best_state = np.inf, None
        for _ in range(self.epochs):
            self.net.train()
            for k in torch.randperm(len(xt)).split(self.batch):
                xb, yb, mb = xt[k].to(self.dev), yt[k].to(self.dev), mt[k].to(self.dev)
                loss = (bce(self.net(xb), yb) * mb).sum() / mb.sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
            v = self._val_loss(parts["val"], bce)
            if v < best:
                best, best_state = v, {k: t.detach().clone() for k, t in self.net.state_dict().items()}
        self.net.load_state_dict(best_state)
        # Platt map on the held-back year: calibrated probabilities like M1/M2.
        from sklearn.linear_model import LogisticRegression
        z = self._logits(parts["val"][0])
        n, yy, xx = parts["val"][4]
        lv = z[n, yy, xx]
        yv = parts["val"][3][label].to_numpy()
        ok = np.isfinite(yv)
        self.platt = LogisticRegression(C=1e3).fit(lv[ok, None], yv[ok].astype(int))
        return self

    def _logits(self, xin):
        import torch
        self.net.eval()
        out = []
        with torch.no_grad():
            for k in torch.arange(len(xin)).split(64):
                out.append(self.net(xin[k].to(self.dev)).float().cpu().numpy())
        return np.concatenate(out)

    def _val_loss(self, part, bce) -> float:
        import torch
        xv, yv, mv = part[:3]
        self.net.eval()
        tot, cnt = 0.0, 0.0
        with torch.no_grad():
            for k in torch.arange(len(xv)).split(64):
                l = bce(self.net(xv[k].to(self.dev)), yv[k].to(self.dev)) * mv[k].to(self.dev)
                tot, cnt = tot + float(l.sum()), cnt + float(mv[k].sum())
        return tot / max(cnt, 1)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        import torch
        xin, (n, y, x) = self._inputs(df)
        z = self._logits(torch.tensor(xin))[n, y, x]
        return self.platt.predict_proba(z[:, None])[:, 1]
