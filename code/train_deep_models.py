"""
Deep-learning post-processing for National Water Model streamflow forecasts.

Inputs expected:
  data/combined_modeling_table_complete.csv produced by prepare_data.py.

Models implemented:
  1. LSTM residual predictor
  2. GRU residual predictor
  3. Transformer encoder residual predictor

Target:
  residual_h1 ... residual_h18 = observed USGS flow at valid time - NWM forecast.
Corrected forecast:
  corrected_h = nwm_h + predicted_residual_h.
"""
from pathlib import Path
import argparse, json, random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

FEATURE_COLS = [f'nwm_h{i}' for i in range(1,19)] + [
    'usgs_flow', 'resid_h1_lag1', 'resid_h1_lag2', 'resid_h1_lag3',
    'hour_sin', 'hour_cos', 'doy_sin', 'doy_cos', 'estimated_flag'
]
TARGET_COLS = [f'resid_h{i}' for i in range(1,19)]
def timestamp_key(x):
    ts = pd.Timestamp(x)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return str(ts)

class SeqDataset(Dataset):
    def __init__(self, df, split, seq_len, x_scaler=None, y_scaler=None, fit_scalers=False):
        xs, ys, meta = [], [], []
        for station, sdf in df.sort_values(['station_name', 'timestamp']).groupby('station_name'):
            sdf = sdf.reset_index(drop=True)
            x = sdf[FEATURE_COLS].values.astype('float32')
            y = sdf[TARGET_COLS].values.astype('float32')
            splits = sdf['split'].values
            times = sdf['timestamp'].values
            for i in range(seq_len - 1, len(sdf)):
                if splits[i] != split:
                    continue
                xs.append(x[i-seq_len+1:i+1])
                ys.append(y[i])
                meta.append((station, timestamp_key(times[i])))
        self.raw_x = np.asarray(xs, dtype='float32')
        self.raw_y = np.asarray(ys, dtype='float32')
        self.meta = meta
        if fit_scalers:
            x_scaler = StandardScaler().fit(self.raw_x.reshape(-1, self.raw_x.shape[-1]))
            y_scaler = StandardScaler().fit(self.raw_y)
        self.x_scaler = x_scaler; self.y_scaler = y_scaler
        x_scaled = x_scaler.transform(self.raw_x.reshape(-1, self.raw_x.shape[-1])).reshape(self.raw_x.shape)
        y_scaled = y_scaler.transform(self.raw_y)
        self.x = torch.tensor(x_scaled, dtype=torch.float32)
        self.y = torch.tensor(y_scaled, dtype=torch.float32)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class LSTMResidualModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, output_size=18):
        super().__init__()
        self.rnn = nn.LSTM(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

class GRUResidualModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, output_size=18):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])

class TransformerResidualModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, output_size=18, nhead=4, num_layers=2):
        super().__init__()
        self.proj = nn.Linear(n_features, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*2,
            dropout=0.10, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, output_size))
    def forward(self, x):
        z = self.proj(x)
        z = self.encoder(z)
        return self.head(z[:, -1, :])

def metric_dict(obs, pred):
    obs = np.asarray(obs, dtype=float); pred = np.asarray(pred, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[mask]; pred = pred[mask]
    if len(obs) < 2:
        return {'CC': np.nan, 'RMSE': np.nan, 'PBIAS': np.nan, 'NSE': np.nan}
    return {
        'CC': float(np.corrcoef(obs, pred)[0, 1]),
        'RMSE': float(np.sqrt(np.mean((obs - pred) ** 2))),
        'PBIAS': float((np.sum(obs) - np.sum(pred)) / np.sum(obs) * 100),
        'NSE': float(1 - np.sum((pred - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2)),
    }

def train_one(model, train_ds, val_ds, output_dir, name, epochs, batch_size, lr, patience):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_loss = float('inf'); best_state = None; wait = 0; history = []
    for epoch in range(1, epochs + 1):
        model.train(); train_losses = []
        for xb, yb in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_losses.append(loss.item())
        model.eval(); val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(loss_fn(model(xb), yb).item())
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'val_loss': float(np.mean(val_losses))}
        history.append(row)
        print(name, row)
        if row['val_loss'] < best_loss:
            best_loss = row['val_loss']; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; wait = 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), output_dir / f'{name}.pt')
    pd.DataFrame(history).to_csv(output_dir / f'{name}_training_history.csv', index=False)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='../data/combined_modeling_table_complete.csv')
    parser.add_argument('--output', default='../results/deep_learning')
    parser.add_argument('--seq-len', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=6)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, parse_dates=['timestamp'])

    train_ds = SeqDataset(df, 'train', args.seq_len, fit_scalers=True)
    val_ds = SeqDataset(df, 'validation', args.seq_len, train_ds.x_scaler, train_ds.y_scaler)
    test_ds = SeqDataset(df, 'test', args.seq_len, train_ds.x_scaler, train_ds.y_scaler)
    print(f'Sequences: train={len(train_ds)}, validation={len(val_ds)}, test={len(test_ds)}')

    model_factories = {
        'LSTM': lambda: LSTMResidualModel(len(FEATURE_COLS)),
        'GRU': lambda: GRUResidualModel(len(FEATURE_COLS)),
        'Transformer': lambda: TransformerResidualModel(len(FEATURE_COLS)),
    }

    trained = {}
    for name, factory in model_factories.items():
        trained[name] = train_one(factory(), train_ds, val_ds, output_dir, name, args.epochs, args.batch_size, args.lr, args.patience)

    # Map metadata back to original rows for corrected forecast metrics.
    key_map = {(r.station_name, timestamp_key(r.timestamp)): r for r in df.itertuples(index=False)}
    rows = []
    loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    for name, model in trained.items():
        preds = []
        model.eval()
        with torch.no_grad():
            for xb, _ in loader:
                preds.append(model(xb).numpy())
        pred_resid = test_ds.y_scaler.inverse_transform(np.vstack(preds))
        for i, (station, ts) in enumerate(test_ds.meta):
            raw = key_map[(station, ts)]
            for h in range(1, 19):
                obs = getattr(raw, f'obs_h{h}')
                nwm = getattr(raw, f'nwm_h{h}')
                corrected = nwm + pred_resid[i, h-1]
                rows.append({'model': name, 'station': station, 'timestamp': ts, 'horizon': h,
                             'observed': obs, 'original_nwm': nwm, 'corrected_forecast': corrected})
    pred_df = pd.DataFrame(rows)
    pred_df.to_csv(output_dir / 'deep_learning_test_predictions.csv', index=False)

    metrics = []
    for (name, station, horizon), sdf in pred_df.groupby(['model', 'station', 'horizon']):
        metrics.append({'model': name, 'station': station, 'horizon': horizon,
                        **metric_dict(sdf['observed'], sdf['corrected_forecast'])})
    pd.DataFrame(metrics).to_csv(output_dir / 'deep_learning_test_metrics.csv', index=False)
    pd.DataFrame(metrics).groupby(['model', 'station'])[['CC', 'RMSE', 'PBIAS', 'NSE']].mean().reset_index().to_csv(output_dir / 'deep_learning_test_summary.csv', index=False)

if __name__ == '__main__':
    main()
