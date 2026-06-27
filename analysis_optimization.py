"""Intelligent frame extraction and parallel processing."""
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor, as_completed


def intelligent_frame_extraction(video_path: str, output_dir: str, max_frames: int = 40) -> list:
    """
    Extract key frames based on importance, not just uniformly.
    Prioritizes: first 3 seconds (hook), frames with text, last 2 seconds (CTA/payoff).

    Returns list of saved frame paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap or not cap.isOpened():
        print("[ERROR] Could not open video for intelligent extraction")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    all_frames = []  # List of (timestamp, frame, priority) tuples

    # 1. ALWAYS extract first 3 seconds (CRITICAL for hooks) - HIGH priority
    hook_frame_count = min(int(fps * 3), total_frames)
    for i in range(0, hook_frame_count, max(1, int(fps / 5))):  # 5 frames per second
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            all_frames.append((timestamp, frame, 'hook'))

    # 2. Sample throughout video for text detection - MEDIUM priority
    sample_interval = max(1, int(total_frames / 30))  # Check 30 points
    for i in range(0, total_frames, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret and detect_text_regions(frame):
            timestamp = i / fps
            # Avoid duplicates from hook section
            if timestamp > 3.0 and timestamp < (duration - 2.0):
                all_frames.append((timestamp, frame, 'text'))

    # 3. ALWAYS extract last 2 seconds (CTA/payoff) - HIGH priority
    start_frame = max(0, total_frames - int(fps * 2))
    for i in range(start_frame, total_frames, max(1, int(fps / 5))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            timestamp = i / fps
            all_frames.append((timestamp, frame, 'ending'))

    cap.release()

    # Sort by timestamp to maintain chronological order
    all_frames.sort(key=lambda x: x[0])

    # Limit to max_frames (prioritize hooks and endings)
    if len(all_frames) > max_frames:
        # Keep all hooks and endings, sample text frames
        hooks = [f for f in all_frames if f[2] == 'hook']
        endings = [f for f in all_frames if f[2] == 'ending']
        text = [f for f in all_frames if f[2] == 'text']

        # Calculate how many text frames we can keep
        priority_count = len(hooks) + len(endings)
        text_budget = max_frames - priority_count

        if text_budget > 0 and len(text) > text_budget:
            # Sample text frames evenly
            step = len(text) / text_budget
            text = [text[int(i * step)] for i in range(text_budget)]
        elif text_budget <= 0:
            text = []

        all_frames = hooks + text + endings
        all_frames.sort(key=lambda x: x[0])

    # Save frames to disk
    saved_paths = []
    os.makedirs(output_dir, exist_ok=True)

    for idx, (timestamp, frame, priority) in enumerate(all_frames):
        frame_path = os.path.join(output_dir, f"frame_{idx:04d}_t{timestamp:.2f}_{priority}.jpg")
        cv2.imwrite(frame_path, frame)
        saved_paths.append(frame_path)

    print(f"[INTELLIGENT] Extracted {len(saved_paths)} frames:")
    hook_count = sum(1 for f in all_frames if f[2] == 'hook')
    text_count = sum(1 for f in all_frames if f[2] == 'text')
    ending_count = sum(1 for f in all_frames if f[2] == 'ending')
    print(f"  Hooks (0-3s): {hook_count}")
    print(f"  Text frames: {text_count}")
    print(f"  Endings (last 2s): {ending_count}")

    return saved_paths


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
