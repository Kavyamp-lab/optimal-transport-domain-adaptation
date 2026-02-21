import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import ot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE

# =============================
# Setup
# =============================

os.makedirs('data', exist_ok=True)
os.makedirs('results', exist_ok=True)

sns.set_theme(style="whitegrid")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================
# Transforms
# =============================

transform_source = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

transform_target = transforms.Compose([
    transforms.RandomRotation(degrees=(45, 60)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    AddGaussianNoise(0., 0.5)
])

# =============================
# Load Data
# =============================

source_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform_source)
target_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform_target)

np.random.seed(42)
source_indices = np.random.choice(len(source_dataset), 2000, replace=False)
target_indices = np.random.choice(len(target_dataset), 2000, replace=False)

source_subset = Subset(source_dataset, source_indices)
target_subset = Subset(target_dataset, target_indices)

source_loader = DataLoader(source_subset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_subset, batch_size=64, shuffle=False)

# =============================
# Save Sample Images
# =============================

def save_sample_images(dataset, title, filename):
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        img, _ = dataset[i]
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

save_sample_images(source_subset, "Source Domain (Standard MNIST)", "source_domain.png")
save_sample_images(target_subset, "Target Domain (Rotated + Noisy MNIST)", "target_domain.png")

# =============================
# Model Definition
# =============================

class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 5 * 5, 128)
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.classifier = nn.Linear(128, 10)
        
    def forward(self, x):
        features = self.extractor(x)
        return self.classifier(torch.relu(features)), features

model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================
# Train on Source
# =============================

print("Training Baseline Model on Source Domain...")

model.train()
for epoch in range(5):
    for imgs, labels in source_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

print("Training Complete.")

# =============================
# Feature Extraction
# =============================

def extract_features(loader):
    model.eval()
    features, labels_list = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            _, feats = model(imgs)
            features.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(features), np.concatenate(labels_list)

Xs, ys = extract_features(source_loader)
Xt, yt = extract_features(target_loader)

print(f"Source Features Shape: {Xs.shape}")
print(f"Target Features Shape: {Xt.shape}")

# =============================
# Baseline Prediction
# =============================

weights = model.classifier.weight.detach().cpu().numpy()
bias = model.classifier.bias.detach().cpu().numpy()

def predict(features):
    logits = np.dot(features, weights.T) + bias
    return np.argmax(logits, axis=1)

pred_t_baseline = predict(Xt)
acc_baseline = accuracy_score(yt, pred_t_baseline)

# =============================
# Z-Score Normalization
# =============================

scaler_s = StandardScaler().fit(Xs)
scaler_t = StandardScaler().fit(Xt)

Xs_norm = scaler_s.transform(Xs)
Xt_norm = scaler_t.transform(Xt)

clf_norm = LogisticRegression(max_iter=1000).fit(Xs_norm, ys)
acc_norm = accuracy_score(yt, clf_norm.predict(Xt_norm))

# =============================
# Stable Sinkhorn OT
# =============================

print("\nApplying Stable Sinkhorn Optimal Transport...")

scaler_ot = StandardScaler().fit(Xs)
Xs_scaled = scaler_ot.transform(Xs)
Xt_scaled = scaler_ot.transform(Xt)

reg_e = 1.0

ot_sinkhorn = ot.da.SinkhornTransport(
    reg_e=reg_e,
    max_iter=1000,
    tol=1e-9,
    verbose=False
)

ot_sinkhorn.fit(Xs=Xt_scaled, Xt=Xs_scaled)
Xt_aligned_scaled = ot_sinkhorn.transform(Xs=Xt_scaled)

pred_t_ot = predict(scaler_ot.inverse_transform(Xt_aligned_scaled))
acc_ot = accuracy_score(yt, pred_t_ot)

print(f"Target Accuracy (Before Adaptation): {acc_baseline * 100:.2f}%")
print(f"Target Accuracy (Z-Score Normalization): {acc_norm * 100:.2f}%")
print(f"Target Accuracy (After Sinkhorn OT): {acc_ot * 100:.2f}%")

# =============================
# Wasserstein Distance
# =============================

n_samples = 500
a = np.ones((n_samples,)) / n_samples
b = np.ones((n_samples,)) / n_samples

M_before = ot.dist(Xs_scaled[:n_samples], Xt_scaled[:n_samples], metric='sqeuclidean')
w_dist_before = ot.emd2(a, b, M_before)

M_after = ot.dist(Xs_scaled[:n_samples], Xt_aligned_scaled[:n_samples], metric='sqeuclidean')
w_dist_after = ot.emd2(a, b, M_after)

print(f"\nWasserstein Distance BEFORE: {w_dist_before:.4f}")
print(f"Wasserstein Distance AFTER:  {w_dist_after:.4f}")

# =============================
# Save Accuracy Bar Plot
# =============================

plt.figure()
methods = ["Baseline", "Z-Score", "Sinkhorn OT"]
accuracies = [acc_baseline, acc_norm, acc_ot]

plt.bar(methods, accuracies)
plt.ylabel("Accuracy")
plt.title("Target Domain Accuracy Comparison")
plt.savefig("results/accuracy_comparison.png")
plt.close()

# =============================
# t-SNE Visualization
# =============================

tsne = TSNE(n_components=2, random_state=42)

combined_before = np.vstack((Xs_scaled[:500], Xt_scaled[:500]))
combined_after = np.vstack((Xs_scaled[:500], Xt_aligned_scaled[:500]))

tsne_before = tsne.fit_transform(combined_before)
tsne_after = tsne.fit_transform(combined_after)

plt.figure()
plt.scatter(tsne_before[:500,0], tsne_before[:500,1], s=5)
plt.scatter(tsne_before[500:,0], tsne_before[500:,1], s=5)
plt.title("t-SNE Before OT")
plt.savefig("results/tsne_before_ot.png")
plt.close()

plt.figure()
plt.scatter(tsne_after[:500,0], tsne_after[:500,1], s=5)
plt.scatter(tsne_after[500:,0], tsne_after[500:,1], s=5)
plt.title("t-SNE After OT")
plt.savefig("results/tsne_after_ot.png")
plt.close()

print("\nAll results saved inside 'results/' folder.")
print("Done Successfully 🚀")