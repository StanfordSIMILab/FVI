import cv2
import numpy as np
from sklearn.cluster import KMeans
from umap import UMAP
import os


def extract_frames(video_path, output_dir="frames"):
    """
    Extract frames from a video and save them as images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_paths = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_paths.append(frame_path)
        frame_idx += 1

    cap.release()
    return frame_paths


def compute_frame_features(frame_paths):
    """
    Compute flattened pixel features for each frame.
    """
    features = []
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
        features.append(frame.flatten())
    return np.array(features)


def apply_umap(features, n_components=2):
    """
    Apply UMAP dimensionality reduction on features.
    """
    umap_model = UMAP(n_components=n_components, random_state=42)
    reduced_features = umap_model.fit_transform(features)
    return reduced_features


def cluster_frames(reduced_features, n_clusters=5):
    """
    Perform clustering on reduced UMAP features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(reduced_features)
    return cluster_labels, kmeans.cluster_centers_


def select_top_k_from_clusters(frame_paths, reduced_features, cluster_labels, k=3):
    """
    Select top-k frames from each cluster based on proximity to cluster centroids.
    """
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((frame_paths[i], reduced_features[i]))

    top_k_frames = []
    for label, frames in clusters.items():
        # Sort frames by distance to cluster centroid
        frames.sort(key=lambda x: np.linalg.norm(x[1]))
        top_k_frames.extend([frame[0] for frame in frames[:k]])

    return top_k_frames


def compute_fvi(frame_paths):
    """
    Compute Frame Variation Index (FVI) for consecutive frames.
    """
    fvi_values = []

    for i in range(len(frame_paths) - 1):
        # Load consecutive frames
        frame1 = cv2.imread(frame_paths[i], cv2.IMREAD_GRAYSCALE)
        frame2 = cv2.imread(frame_paths[i + 1], cv2.IMREAD_GRAYSCALE)
        
        # Compute absolute difference
        diff = np.abs(frame2.astype(np.float32) - frame1.astype(np.float32))
        
        # Sum up the pixel differences
        fvi_values.append(np.sum(diff))

    return fvi_values


if __name__ == "__main__":
    video_path = "input_video.mp4"  # replace with actual video path
    output_dir = "extracted_frames"

    # Step 1: Extract frames
    frame_paths = extract_frames(video_path, output_dir)

    # Step 2: Compute frame features
    features = compute_frame_features(frame_paths)

    # Step 3: Apply UMAP
    reduced_features = apply_umap(features, n_components=2)

    # Step 4: Cluster frames
    cluster_labels, cluster_centers = cluster_frames(reduced_features, n_clusters=5)

    # Step 5: Select top-k frames per cluster
    top_k_frames = select_top_k_from_clusters(frame_paths, reduced_features, cluster_labels, k=3)

    # Step 6: Compute FVI for selected frames
    fvi_values = compute_fvi(top_k_frames)

    # Display results
    print("Top-k Frames:", top_k_frames)
    print("FVI Values for Selected Frames:", fvi_values)
