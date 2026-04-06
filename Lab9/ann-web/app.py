"""
ANN Web App - Flask
Datasets: MNIST (chữ số viết tay) | Cats vs Dogs (ảnh)
"""

import io, os, base64
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)

DEVICE = torch.device('cpu')

# Model Definitions 

class ANN_MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ANN_CatDog(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


# Load Models

def load_model(model_class, path):
    model = model_class()
    if os.path.exists(path):
        try:
            state = torch.load(path, map_location=DEVICE)
            model.load_state_dict(state)
            model.eval()
            print(f"✅ Loaded: {path}")
        except Exception as e:
            print(f"⚠️  Could not load {path}: {e}. Using untrained model.")
    else:
        print(f"⚠️  Model file not found: {path}. Using untrained model.")
        model.eval()
    return model


mnist_model   = load_model(ANN_MNIST,  'models/ann_mnist.pt')
catdog_model  = load_model(ANN_CatDog, 'models/ann_catdog.pt')

# Transforms 

mnist_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

catdog_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

MNIST_CLASSES  = [str(i) for i in range(10)]
CATDOG_CLASSES = ['Cat 🐱', 'Dog 🐶']

# Prediction Helpers 

def predict(model, tensor, classes):
    model.eval()
    with torch.no_grad():
        output = model(tensor.unsqueeze(0).to(DEVICE))
        probs  = torch.softmax(output, dim=1)[0]
        idx    = probs.argmax().item()
    return {
        'label':      classes[idx],
        'confidence': round(probs[idx].item() * 100, 2),
        'all_probs':  {c: round(probs[i].item() * 100, 2) for i, c in enumerate(classes)},
    }

# Routes 

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict/mnist', methods=['POST'])
def predict_mnist():
    try:
        file = request.files['image']
        img  = Image.open(file.stream).convert('L')
        tensor = mnist_transform(img)
        result = predict(mnist_model, tensor, MNIST_CLASSES)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/predict/catdog', methods=['POST'])
def predict_catdog():
    try:
        file = request.files['image']
        img  = Image.open(file.stream).convert('RGB')
        tensor = catdog_transform(img)
        result = predict(catdog_model, tensor, CATDOG_CLASSES)
        return jsonify({'success': True, **result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5001)
