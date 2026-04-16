import os
import torch
import numpy as np
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from cleanfid.fid import frechet_distance
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
import faiss
from sklearn.metrics import pairwise_distances
import argparse
import warnings
warnings.filterwarnings("ignore")

MODELS = ["mobilenet", "densenet", "resnet50"]

def initialize_model(model_name, device):
    print(model_name)
    if model_name == "mobilenet":
        model = models.mobilenet_v2(pretrained=True).to(device)
        feature_extractor = model.features
        img_size = 224
    elif model_name == "densenet":
        model = models.densenet201(pretrained=True).to(device)
        feature_extractor = model.features
        img_size = 224
    else:
        model = models.resnet50(pretrained=True).to(device)
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        img_size = 224

    
    return feature_extractor, img_size

def extract_patches(inputs, patch_size: int, stride_r: float):
    strides = int(patch_size * stride_r)
    if strides < 1:
        raise ValueError(f"stride=int(patch_size*stride_r) must be >= 1; got patch_size={patch_size}, stride_r={stride_r}")
    patches = torch.nn.functional.unfold(inputs, kernel_size=patch_size, stride=strides)
    patches = patches.transpose(1, 2).contiguous().view(-1, 3, patch_size, patch_size)
    return patches

def extract_features(
    loader, feature_extractor, device, patch_size: int = 80, img_size: int = 224, stride_r: float = 0.8
):
    features = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            patches = extract_patches(imgs, patch_size=patch_size, stride_r=stride_r)
            patches_resized = torch.nn.functional.interpolate(
                patches, size=img_size, mode='bilinear', align_corners=False
            )
            fmap = feature_extractor(patches_resized)
            fmap = torch.nn.functional.adaptive_avg_pool2d(fmap, (1, 1))
            fmap = fmap.view(fmap.size(0), -1)
            features.append(fmap.cpu().numpy())
    return np.vstack(features)


# evaluation metrics
def compute_rel_l2(X, X_rec):
    return np.linalg.norm(X - X_rec, ord='fro') / np.linalg.norm(X, ord='fro')

def compute_sparsity(U):
    return 1 - (np.count_nonzero(U) / U.size)

def compute_fid(X, Y):
    cost_matrix = pairwise_distances(X, Y, metric='euclidean')
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    emd = cost_matrix[row_ind, col_ind].mean()
    return emd

def compute_fid_cleanfid(X, Y):
    mu1, sigma1 = X.mean(0), np.cov(X, rowvar=False)
    mu2, sigma2 = Y.mean(0), np.cov(Y, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6)
    return fid

def deepknn_score(A, recon, k=5, use_gpu=False):
    A = A.astype(np.float32)
    recon = recon.astype(np.float32)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(A.shape[1])
        index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    else:
        index = faiss.IndexFlatL2(A.shape[1])
    index.add(A)
    D, _ = index.search(recon, k)
    return np.mean(D[:, 0])

def compute_stability(A, extractor_fn, method_name, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    concept_banks = []
    for _, idx in kf.split(A):
        A_fold = A[idx]
        U, V = extractor_fn(A_fold)
        concept_banks.append(V)

    similarities = []
    for i in range(n_folds):
        for j in range(i+1, n_folds):
            sim_matrix = cosine_similarity(concept_banks[i], concept_banks[j])
            row_ind, col_ind = linear_sum_assignment(-sim_matrix)
            matched_sims = sim_matrix[row_ind, col_ind]
            similarities.append(np.mean(matched_sims))

    stability = 1 - np.mean(similarities)
    return stability


# concept extraction functions
def extract_pca(A, num_concepts=25):
    pca = PCA(n_components=num_concepts)
    U = pca.fit_transform(A)
    V = pca.components_
    return U, V

def extract_kmeans(A, num_concepts=25):
    kmeans = KMeans(n_clusters=num_concepts, random_state=0).fit(A)
    labels = kmeans.labels_
    V = kmeans.cluster_centers_
    U = np.eye(num_concepts)[labels]
    return U, V

def extract_nmf(A, num_concepts=25):
    nmf = NMF(n_components=num_concepts, init='nndsvd', random_state=0, max_iter=1000)
    U = nmf.fit_transform(A)
    V = nmf.components_
    return U, V


# evaluation wrapper
def evaluate_method(name, A, extractor_fn, scaler, num_concepts=25):
    U, V = extractor_fn(A)
    A_rec = np.dot(U, V)
    
    if name in ["PCA"]:
        A_rec = scaler.inverse_transform(A_rec)
        A = scaler.inverse_transform(A)
    
    rel_l2 = compute_rel_l2(A, A_rec)
    sparsity = compute_sparsity(U)
    stability = compute_stability(A, extractor_fn, name)
    fid = compute_fid(A, A_rec)

    A_knn = A / np.linalg.norm(A, axis=1, keepdims=True)
    A_rec_knn = A_rec / np.linalg.norm(A_rec, axis=1, keepdims=True)
    ood = deepknn_score(A_knn, A_rec_knn, k=5, use_gpu=False)

    return rel_l2, sparsity, fid, ood, stability



def main():
    ap = argparse.ArgumentParser(description="Concept Basis Quality Evaluation")
    ap.add_argument("--img_folder_dir", type=str, default="datasets/imagenet/train", help="Path to image directory")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader")
    ap.add_argument("--num_concepts", type=int, default=25, help="Number of concepts")
    ap.add_argument("--n_folds", type=int, default=5, help="Number of folds for stability evaluation")
    ap.add_argument("--stride_r", type=float, default=0.8, help="Stride ratio for patches")
    ap.add_argument("--patch_size", type=int, default=80, help="Patch size")
    args = ap.parse_args()

    current_dir = os.getcwd()
    print(current_dir)
    args.img_folder_dir = os.path.join(current_dir, args.img_folder_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"


    for model_name in MODELS:
        print(f"\n==================== MODEL: {model_name} ====================")
    
        feature_extractor, img_size = initialize_model(model_name, device)
        transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        dataset = ImageFolder(args.img_folder_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        print("extracting features...")
        scaler = StandardScaler()
        A = extract_features(
            loader,
            feature_extractor,
            device,
            patch_size=args.patch_size,
            img_size=img_size,
            stride_r=args.stride_r,
        )
        A = np.maximum(A, 0)
        A_normalized = scaler.fit_transform(A)

        results = {}
        results['PCA'] = evaluate_method("PCA", A_normalized, extract_pca, scaler, num_concepts=args.num_concepts)
        results['KMeans'] = evaluate_method("KMeans", A, extract_kmeans, scaler, num_concepts=args.num_concepts)
        results['NMF'] = evaluate_method("NMF", A, extract_nmf, scaler, num_concepts=args.num_concepts)

        print(f"\n\n===== Summary Table for: {model_name} =====")
        print(f"{'Method':<8} | {'Rel_L2 ↓':>7} | {'Sparsity ↑':>8} | {'Stability ↓':>9} | {'FID ↓':>7} | {'OOD ↓':>7}")
        print("-" * 60)
        for method, (r, s, f, o, st) in results.items():
            print(f"{method:<8} | {r:7.4f} | {s:8.4f} | {st:9.4f} | {f:7.4f} | {o:7.4f}")

        print()

if __name__ == "__main__":
    main()