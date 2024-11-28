import cv2
import numpy as np
import os

def extract_frames(video_path, output_dir="frames"):
    """
    Extract frames from a video and save them as images.
    
    Args:
        video_path (str): Path to the input video.
        output_dir (str): Directory to save extracted frames.
    
    Returns:
        list: List of file paths to extracted frames.
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


def compute_fvi(frame_paths):
    """
    Compute Frame Variation Index (FVI) based on pixel intensity differences.
    
    Args:
        frame_paths (list): List of file paths to consecutive frames.
    
    Returns:
        list: List of FVI values for each pair of consecutive frames.
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


def normalize_fvi(fvi_values):
    """
    Normalize FVI values to the range [0, 1].
    
    Args:
        fvi_values (list): List of raw FVI values.
    
    Returns:
        list: List of normalized FVI values.
    """
    fvi_array = np.array(fvi_values)
    normalized = (fvi_array - np.min(fvi_array)) / (np.max(fvi_array) - np.min(fvi_array))
    return normalized.tolist()


def select_high_variation_frames(frame_paths, fvi_values, top_k=5):
    """
    Select frames with the highest variation based on FVI values.
    
    Args:
        frame_paths (list): List of file paths to frames.
        fvi_values (list): List of FVI values for consecutive frames.
        top_k (int): Number of top frames to select.
    
    Returns:
        list: List of file paths to the top-k high-variation frames.
    """
    # Get indices of top-k FVI values
    top_k_indices = np.argsort(fvi_values)[-top_k:]
    return [frame_paths[i] for i in top_k_indices]


if __name__ == "__main__":
    # replace with input video path
    video_path = "input_video.mp4" 
    output_dir = "extracted_frames"

    # Step 1: Extract frames from the video
    frame_paths = extract_frames(video_path, output_dir)

    # Step 2: Compute FVI values
    fvi_values = compute_fvi(frame_paths)

    # Step 3: Normalize FVI values
    normalized_fvi = normalize_fvi(fvi_values)

    # Step 4: Select top-k high-variation frames
    top_k_frames = select_high_variation_frames(frame_paths, normalized_fvi, top_k=5)

    # Display results
    print("FVI Values:", fvi_values)
    print("Normalized FVI:", normalized_fvi)
    print("Top-k High-Variation Frames:", top_k_frames)
