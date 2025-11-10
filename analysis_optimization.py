"""Intelligent frame extraction and parallel processing."""
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def intelligent_frame_extraction(video_path: str, metadata: dict = None) -> dict:
    """
    Extract key frames based on importance, not just uniformly.
    Always gets: first 3 seconds (hook), frames with text, scene changes, last 2 seconds (CTA).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        print("[ERROR] Could not open video for intelligent extraction")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    key_frames = {
        'hook_frames': [],      # First 3 seconds
        'text_frames': [],       # Frames with text overlays
        'scene_changes': [],     # Major visual transitions
        'ending_frames': []      # Last 2 seconds
    }

    # 1. ALWAYS extract first 3 seconds (critical for hooks)
    hook_frame_count = min(int(fps * 3), total_frames)
    for i in range(0, hook_frame_count, max(1, int(fps / 5))):  # 5 frames per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            key_frames['hook_frames'].append((timestamp, frame))

    # 2. Sample throughout video for text detection
    sample_interval = max(1, int(total_frames / 30))  # Check 30 points
    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and detect_text_regions(frame):
            timestamp = i / fps
            key_frames['text_frames'].append((timestamp, frame))

    # 3. ALWAYS extract last 2 seconds (CTA/payoff)
    start_frame = max(0, total_frames - int(fps * 2))
    for i in range(start_frame, total_frames, max(1, int(fps / 5))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            key_frames['ending_frames'].append((timestamp, frame))

    cap.release()

    print(f"[INFO] Intelligent extraction complete:")
    print(f"  Hook frames: {len(key_frames['hook_frames'])}")
    print(f"  Text frames: {len(key_frames['text_frames'])}")
    print(f"  Ending frames: {len(key_frames['ending_frames'])}")

    return key_frames


def detect_text_regions(frame: np.ndarray) -> bool:
    """Quick text detection using edge detection."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Check top and bottom thirds where text usually appears
        h, w = gray.shape
        top_region = edges[0:h//3, :]
        bottom_region = edges[2*h//3:, :]

        # High edge density suggests text
        return np.sum(top_region) > 10000 or np.sum(bottom_region) > 10000
    except Exception as e:
        print(f"[WARNING] Text detection failed: {e}")
        return False


def parallel_video_processing(tiktok_url: str, strategy: str, frames_per_minute: int, cap: int, scene_threshold: float):
    """Process video components in parallel for speed."""
    from processing import extract_video_metadata, extract_audio_and_frames
    import traceback

    results = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Start tasks simultaneously
        futures = {}
        futures['metadata'] = executor.submit(extract_video_metadata, tiktok_url)
        futures['extraction'] = executor.submit(
            extract_audio_and_frames,
            tiktok_url, strategy, frames_per_minute, cap, scene_threshold
        )

        # Wait for results
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=120)  # 2 minute timeout
                print(f"[PARALLEL] {name} complete")
            except Exception as e:
                print(f"[ERROR] {name} failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                results[name] = None

    return results


def optimize_frame_selection(frame_paths: list, max_frames: int = 30) -> list:
    """
    Select the most diverse and informative frames from a larger set.
    Uses color histogram and edge density to pick varied frames.
    """
    if len(frame_paths) <= max_frames:
        return frame_paths

    print(f"[INFO] Optimizing frame selection: {len(frame_paths)} -> {max_frames}")

    frame_features = []
    valid_paths = []

    for path in frame_paths:
        try:
            img = cv2.imread(path)
            if img is None:
                continue

            # Calculate features
            # 1. Color histogram
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # 2. Edge density
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges) / edges.size

            # 3. Brightness variance
            brightness_var = np.var(gray)

            feature = np.concatenate([hist, [edge_density, brightness_var]])
            frame_features.append(feature)
            valid_paths.append(path)

        except Exception as e:
            print(f"[WARNING] Could not analyze frame {path}: {e}")
            continue

    if len(valid_paths) <= max_frames:
        return valid_paths

    # Use k-means clustering to find diverse frames
    try:
        from sklearn.cluster import KMeans

        # Cluster frames
        n_clusters = min(max_frames, len(valid_paths))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(frame_features)

        # Pick one frame from each cluster (closest to centroid)
        selected_indices = []
        for i in range(n_clusters):
            cluster_indices = np.where(labels == i)[0]
            if len(cluster_indices) > 0:
                # Find frame closest to cluster center
                cluster_features = [frame_features[idx] for idx in cluster_indices]
                distances = [np.linalg.norm(feat - kmeans.cluster_centers_[i]) for feat in cluster_features]
                closest_idx = cluster_indices[np.argmin(distances)]
                selected_indices.append(closest_idx)

        selected_paths = [valid_paths[idx] for idx in sorted(selected_indices)]
        print(f"[SUCCESS] Selected {len(selected_paths)} diverse frames")
        return selected_paths

    except Exception as e:
        print(f"[WARNING] Clustering failed: {e}, using first {max_frames} frames")
        return valid_paths[:max_frames]
