"""
RNN Web App - Flask
Datasets: Sin wave (univariate) | Multivariate time series
"""

import os, json
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
DEVICE = torch.device('cpu')

# Model Definitions 

class RNN_SinWave(nn.Module):
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, x.size(0), self.hidden_size)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out[:, -1, :]), hidden

    def init_hidden(self, batch=1):
        return torch.zeros(1, batch, self.hidden_size)


class RNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=32, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])


# Load Models

def load_model(cls, path, **kwargs):
    m = cls(**kwargs)
    if os.path.exists(path):
        try:
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"✅ {path}")
        except Exception as e:
            print(f"⚠️ {path}: {e}")
    else:
        print(f"⚠️ Not found: {path}")
    m.eval()
    return m

sinwave_model   = load_model(RNN_SinWave, 'models/rnn_sinwave.pt')
multivar_model  = load_model(RNNModel,    'models/rnn_multivar.pt', input_size=3, hidden_size=32)

# Routes

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/sinwave', methods=['POST'])
def predict_sinwave():
    """Generate a sin wave and predict next steps."""
    try:
        data = request.get_json()
        noise  = float(data.get('noise', 0.1))
        steps  = int(data.get('steps', 100))
        future = int(data.get('future', 20))

        x    = np.linspace(0, 20, steps)
        seq  = np.sin(x) + noise * np.random.randn(steps)
        seq  = (seq - seq.min()) / (seq.max() - seq.min() + 1e-8)

        SEQ_LEN = 10
        # Build input sequences
        inputs_list = []
        for i in range(len(seq) - SEQ_LEN):
            inputs_list.append(seq[i:i+SEQ_LEN])

        sinwave_model.eval()
        predictions = []
        with torch.no_grad():
            # Predict on existing data
            for inp in inputs_list:
                t = torch.FloatTensor(inp).unsqueeze(0).unsqueeze(-1)
                h = sinwave_model.init_hidden(1)
                out, h = sinwave_model(t, h)
                predictions.append(float(out.item()))

            # Future predictions
            last_seq = seq[-SEQ_LEN:].tolist()
            h = sinwave_model.init_hidden(1)
            fut = []
            for _ in range(future):
                t = torch.FloatTensor(last_seq).unsqueeze(0).unsqueeze(-1)
                out, h = sinwave_model(t, h)
                v = float(out.item())
                fut.append(v)
                last_seq = last_seq[1:] + [v]

        return jsonify({
            'success': True,
            'original': seq.tolist(),
            'predictions': [None]*SEQ_LEN + predictions,
            'future': fut,
            'x': x.tolist(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/predict/multivar', methods=['POST'])
def predict_multivar():
    """Multivariate time series prediction."""
    try:
        data   = request.get_json()
        steps  = int(data.get('steps', 300))
        future = int(data.get('future', 30))

        t = np.linspace(0, 30, steps)
        f1 = np.sin(t)
        f2 = np.cos(0.5 * t)
        f3 = 0.05 * t + 0.1 * np.random.randn(steps)
        target = 0.5*f1 + 0.3*f2 + 0.2*f3 + 0.05*np.random.randn(steps)

        # Normalize
        features = np.stack([f1, f2, f3], axis=1)
        feat_min, feat_max = features.min(0), features.max(0)
        feat_norm = (features - feat_min) / (feat_max - feat_min + 1e-8)
        tgt_min, tgt_max = target.min(), target.max()
        tgt_norm = (target - tgt_min) / (tgt_max - tgt_min + 1e-8)

        SEQ_LEN = 20
        Xs, ys = [], []
        for i in range(len(feat_norm) - SEQ_LEN):
            Xs.append(feat_norm[i:i+SEQ_LEN])
            ys.append(tgt_norm[i+SEQ_LEN])

        X_tensor = torch.FloatTensor(np.array(Xs))

        multivar_model.eval()
        with torch.no_grad():
            preds_norm = multivar_model(X_tensor).squeeze().numpy()

        # Denormalize
        preds = preds_norm * (tgt_max - tgt_min) + tgt_min

        # Future
        last_feat = feat_norm[-SEQ_LEN:].tolist()
        fut_preds = []
        with torch.no_grad():
            for _ in range(future):
                t_in = torch.FloatTensor([last_feat])
                out  = multivar_model(t_in).item()
                fut_preds.append(out * (tgt_max - tgt_min) + tgt_min)
                # Roll window (use last known features pattern)
                last_feat = last_feat[1:] + [last_feat[-1]]

        return jsonify({
            'success': True,
            'target': target.tolist(),
            'predictions': [None]*SEQ_LEN + preds.tolist(),
            'future': fut_preds,
            't': t.tolist(),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5003)
