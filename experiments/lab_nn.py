"""
Neural networks of the model lab (``experiments/lab.py``).

- ``mlp``: every row on its own, as the trees see it.
- ``tcn``: a temporal convolution over the hours of a bus, so a correction sees
  the model's state in the hours around it (about 2.5 days either way).
- ``tcn`` with ``europe``: the same, and every layer also sees the mean state
  of all buses of the run in the same hour.

All learn the correction in units of `SCALE` with the absolute-error loss, from
the default features unless others are given: numeric ones standardised by the
median and the interquartile range of the training rows, hour and weekday as
sines and cosines, the price setter one-hot. The settings were fixed before the first
run, none is chosen on a held-out year.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pypsa_price_forecaster.dataset import DEFAULT_FEATURES

SCALE = 20.0  # EUR/MWh
CLIP = 8.0  # standardised numeric features are clipped to this
CALENDAR = ["price_setter", "hour", "dayofweek"]


def inputs(data, fit: np.ndarray, features: list[str]) -> np.ndarray:
    """Network inputs of every row, scaled by the statistics of the `fit` rows."""
    X = data.X
    values = X[[f for f in features if f not in CALENDAR]].to_numpy(np.float32)
    sample = values[fit][::20]
    low, median, high = np.nanpercentile(sample, [25, 50, 75], axis=0)
    spread = np.where(
        high > low, (high - low) / 1.349, np.nanstd(sample, axis=0) + 1e-6
    )
    numeric = np.nan_to_num(np.clip((values - median) / spread, -CLIP, CLIP))
    hour = 2 * np.pi * X.hour.to_numpy() / 24
    day = X.dayofweek.to_numpy()
    week = 2 * np.pi * day / 7
    calendar = [np.sin(hour), np.cos(hour), np.sin(2 * hour), np.cos(2 * hour)]
    calendar += [np.sin(week), np.cos(week), day >= 5]
    codes = X.price_setter.cat.codes.to_numpy()
    setter = np.eye(len(X.price_setter.cat.categories) + 1)[codes + 1][:, 1:]
    return np.hstack([numeric, np.column_stack(calendar), setter]).astype(np.float32)


def schedule(optimizer, steps: int, rate: float):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer, rate, total_steps=steps, pct_start=0.1
    )


class MLP(nn.Module):
    def __init__(
        self, size: int, width: int = 256, depth: int = 3, dropout: float = 0.1
    ):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [
                nn.Linear(size if i == 0 else width, width),
                nn.SiLU(),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def mlp(x, y, fit, test, epochs: int = 10, batch: int = 4096, rate: float = 2e-3):
    model = MLP(x.shape[1])
    rows = torch.from_numpy(np.flatnonzero(fit))
    xs, ys = torch.from_numpy(x), torch.from_numpy(y)
    steps = epochs * (len(rows) // batch)
    optimizer = torch.optim.AdamW(model.parameters(), rate, weight_decay=1e-5)
    scheduler = schedule(optimizer, steps, rate)
    model.train()
    for _ in range(epochs):
        order = rows[torch.randperm(len(rows))]
        for i in range(len(rows) // batch):
            take = order[i * batch : (i + 1) * batch]
            loss = F.l1_loss(model(xs[take]), ys[take])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
    model.eval()
    with torch.no_grad():
        return model(xs[torch.from_numpy(np.flatnonzero(test))]).numpy()


class Block(nn.Module):
    def __init__(self, width: int, dilation: int, europe: bool):
        super().__init__()
        self.norm = nn.GroupNorm(8, width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation)
        self.europe = nn.Conv1d(width, width, 1) if europe else None
        self.mix = nn.Conv1d(width, width, 1)

    def forward(self, h, buses: int):
        out = self.conv(F.silu(self.norm(h)))
        if self.europe is not None:
            mean = out.view(-1, buses, *out.shape[1:]).mean(dim=1)
            out = out + self.europe(mean).repeat_interleave(buses, dim=0)
        return h + self.mix(F.silu(out))


class TCN(nn.Module):
    """Residual dilated convolutions over the hours of every bus, centred on the hour."""

    def __init__(
        self, size: int, europe: bool, width: int = 64, dilations=(1, 2, 4, 8, 16, 32)
    ):
        super().__init__()
        self.stem = nn.Conv1d(size, width, 1)
        self.blocks = nn.ModuleList(Block(width, d, europe) for d in dilations)
        self.head = nn.Conv1d(width, 1, 1)

    def forward(self, x):
        """`x`: runs x buses x features x hours, returns runs x buses x hours."""
        runs, buses, size, hours = x.shape
        h = self.stem(x.reshape(runs * buses, size, hours))
        for block in self.blocks:
            h = block(h, buses)
        return self.head(h).view(runs, buses, hours)


def tcn(
    x,
    y,
    train_runs,
    test_runs,
    europe: bool = False,
    epochs: int = 30,
    window: int = 336,
    batch: int = 4,
    rate: float = 2e-3,
):
    """`x`: runs x buses x hours x features, `y`: runs x buses x hours with NaN where unobserved."""
    xs = torch.from_numpy(x).permute(0, 1, 3, 2).contiguous()
    ys = torch.from_numpy(y)
    hours = xs.shape[-1]
    model = TCN(xs.shape[2], europe)
    steps = epochs * max(1, len(train_runs) * hours // (window * batch))
    optimizer = torch.optim.AdamW(model.parameters(), rate, weight_decay=1e-5)
    scheduler = schedule(optimizer, steps, rate)
    model.train()
    for _ in range(steps):
        runs = np.random.choice(train_runs, batch)
        starts = np.random.randint(0, hours - window, batch)
        xb = torch.stack(
            [xs[r, :, :, s : s + window] for r, s in zip(runs, starts, strict=True)]
        )
        yb = torch.stack(
            [ys[r, :, s : s + window] for r, s in zip(runs, starts, strict=True)]
        )
        observed = ~torch.isnan(yb)
        loss = F.l1_loss(model(xb)[observed], yb[observed])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(xs[r : r + 1]).numpy().ravel() for r in test_runs])


def fit_predict(
    kind,
    data,
    train,
    test,
    seed,
    threads,
    features: list[str] = DEFAULT_FEATURES,
    members: int = 1,
    **options,
):
    """Return the correction of the `test` rows from a network fitted on the `train` rows, the mean of `members` fits."""
    torch.set_num_threads(threads)
    fit = train & data.labelled
    x = inputs(data, fit, features)
    y = (data.y / SCALE).astype(np.float32)
    if kind == "tcn":
        # every run holds the same buses in the same order, each with all hours in order
        runs = data.table.run.to_numpy()
        names = list(dict.fromkeys(runs))
        bus = data.table.bus.to_numpy().reshape(len(names), -1)
        buses = len(set(bus[0]))
        hours = bus.shape[1] // buses
        assert (bus.reshape(len(names), buses, hours) == bus[:1, ::hours, None]).all()
        run_rows = runs.reshape(len(names), -1)[:, 0]
        x = x.reshape(len(names), buses, hours, -1)
        y = y.reshape(len(names), buses, hours)
        train_runs = np.flatnonzero(np.isin(run_rows, np.unique(runs[train])))
        test_runs = np.flatnonzero(np.isin(run_rows, np.unique(runs[test])))
    fits = []
    for member in range(members):
        torch.manual_seed(1000 * seed + member)
        np.random.seed(1000 * seed + member)
        if kind == "mlp":
            fits.append(mlp(x, y, fit, test, **options))
        else:
            fits.append(tcn(x, y, train_runs, test_runs, **options))
    return SCALE * np.mean(fits, axis=0)
