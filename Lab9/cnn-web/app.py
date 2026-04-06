"""
CNN Web App - Flask
Datasets: MNIST | Cats vs Dogs | CIFAR-10 | PlantVillage (38 classes)
"""

import io, os
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
DEVICE = torch.device('cpu')

# Model Definitions 

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.fc1   = nn.Linear(32 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, 10)
        self.relu  = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class CatDog_CNN(nn.Module):
    def __init__(self, dropout_max=0.4):
        super().__init__()
        def conv_block(in_c, out_c, drop):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(drop),
            )
        self.features = nn.Sequential(
            conv_block(3, 32, 0.1),
            conv_block(32, 64, 0.2),
            conv_block(64, 128, 0.3),
            conv_block(128, 256, dropout_max),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout_max),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


class CIFAR10_CNN(nn.Module):
    def __init__(self, num_classes=10, dropout=0.35):
        super().__init__()
        def block(in_c, out_c, drop=dropout):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2), nn.Dropout2d(drop),
            )
        self.features = nn.Sequential(
            block(3, 64, 0.2), block(64, 128, 0.25),
            block(128, 256, 0.3), block(256, 512, dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c), nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
        )
        self.skip = nn.Sequential(nn.Conv2d(in_c, out_c, 1, bias=False), nn.BatchNorm2d(out_c)) \
                    if in_c != out_c else nn.Identity()
        self.act  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(dropout)

    def forward(self, x):
        return self.drop(self.act(self.conv(x) + self.skip(x)))


class PlantVillage_CNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            ResBlock(3, 32, 0.1), nn.MaxPool2d(2),
            ResBlock(32, 64, 0.15), nn.MaxPool2d(2),
            ResBlock(64, 128, 0.2), nn.MaxPool2d(2),
            ResBlock(128, 256, 0.25), nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# Load Models 

def load_model(cls, path, **kwargs):
    m = cls(**kwargs)
    if os.path.exists(path):
        try:
            m.load_state_dict(torch.load(path, map_location=DEVICE))
            print(f"✅ {path}")
        except Exception as e:
            print(f"⚠️  {path}: {e}")
    else:
        print(f"⚠️  Not found: {path}")
    m.eval()
    return m


mnist_model       = load_model(MNIST_CNN,       'models/cnn_mnist.pt')
catdog_model      = load_model(CatDog_CNN,      'models/cnn_catdog.pt')
cifar10_model     = load_model(CIFAR10_CNN,     'models/cnn_cifar10.pt')
plantvillage_model = load_model(PlantVillage_CNN, 'models/cnn_plantvillage.pt', num_classes=38)

# Transforms & Labels

mnist_tf = transforms.Compose([
    transforms.Grayscale(), transforms.Resize((28, 28)), transforms.ToTensor(),
])
catdog_tf = transforms.Compose([
    transforms.Resize((160, 160)), transforms.CenterCrop(160), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
cifar10_tf = transforms.Compose([
    transforms.Resize((32, 32)), transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
])
plantvillage_tf = transforms.Compose([
    transforms.Resize((128, 128)), transforms.CenterCrop(128), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

MNIST_CLASSES  = [str(i) for i in range(10)]
CATDOG_CLASSES = ['Cat 🐱', 'Dog 🐶']
CIFAR10_CLASSES = ['airplane ✈️','automobile 🚗','bird 🐦','cat 🐱','deer 🦌',
                   'dog 🐶','frog 🐸','horse 🐴','ship 🚢','truck 🚛']
PLANTVILLAGE_CLASSES = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry___Powdery_mildew','Cherry___healthy',
    'Corn___Cercospora_leaf_spot','Corn___Common_rust','Corn___Northern_Leaf_Blight','Corn___healthy',
    'Grape___Black_rot','Grape___Esca','Grape___Leaf_blight','Grape___healthy',
    'Orange___Haunglongbing','Peach___Bacterial_spot','Peach___healthy',
    'Pepper___Bacterial_spot','Pepper___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites',
    'Tomato___Target_Spot','Tomato___Yellow_Leaf_Curl_Virus','Tomato___Mosaic_virus',
    'Tomato___healthy',
]

# Predict Helper

def predict(model, tensor, classes, top_k=5):
    with torch.no_grad():
        out   = model(tensor.unsqueeze(0).to(DEVICE))
        probs = torch.softmax(out, dim=1)[0]
        idx   = probs.argmax().item()
        topk  = probs.topk(min(top_k, len(classes)))
    return {
        'label':      classes[idx],
        'confidence': round(probs[idx].item() * 100, 2),
        'top_probs':  {classes[i]: round(p.item()*100,2) for i, p in zip(topk.indices, topk.values)},
    }

# Routes

@app.route('/')
def index():
    return render_template('index.html')

def handle(file_key, transform, model, classes, color_mode='RGB', top_k=5):
    try:
        f = request.files[file_key]
        img = Image.open(f.stream).convert(color_mode)
        t   = transform(img)
        res = predict(model, t, classes, top_k)
        return jsonify({'success': True, **res})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/predict/mnist',       methods=['POST'])
def p_mnist():       return handle('image', mnist_tf,       mnist_model,       MNIST_CLASSES,  'L')
@app.route('/predict/catdog',      methods=['POST'])
def p_catdog():      return handle('image', catdog_tf,      catdog_model,      CATDOG_CLASSES)
@app.route('/predict/cifar10',     methods=['POST'])
def p_cifar10():     return handle('image', cifar10_tf,     cifar10_model,     CIFAR10_CLASSES)
@app.route('/predict/plantvillage',methods=['POST'])
def p_plant():       return handle('image', plantvillage_tf, plantvillage_model, PLANTVILLAGE_CLASSES)

if __name__ == '__main__':
    app.run(debug=True, port=5002)
