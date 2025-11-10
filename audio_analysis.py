"""Enhanced audio analysis with ACRCloud viral sound detection."""
import os
import json
import hashlib
import hmac
import base64
import time
import requests


class ViralSoundDetector:
    """ACRCloud integration for viral sound detection."""

    def __init__(self):
        self.host = "identify-us-west-2.acrcloud.com"
        self.access_key = os.getenv("ACRCLOUD_ACCESS_KEY")
        self.access_secret = os.getenv("ACRCLOUD_ACCESS_SECRET")
        self.cache_file = "cache/audio_identifications.json"
        self.load_cache()
        self.enabled = bool(self.access_key and self.access_secret)

        if self.enabled:
            print("[INFO] ACRCloud configured for viral sound detection")
        else:
            print("[WARNING] ACRCloud not configured - skipping viral sound detection")
            print("         Set ACRCLOUD_ACCESS_KEY and ACRCLOUD_ACCESS_SECRET to enable")

    def identify(self, audio_path: str) -> dict:
        """Identify if audio is a known viral sound."""
        if not self.enabled:
            return {'is_viral': False, 'reason': 'ACRCloud not configured'}

        # Check cache first
        audio_hash = self.get_audio_hash(audio_path)
        if audio_hash in self.cache:
            print("[CACHE HIT] Using cached audio identification")
            return self.cache[audio_hash]

        try:
            # Read audio file
            with open(audio_path, 'rb') as f:
                audio_data = f.read()

            # Create ACRCloud signature
            timestamp = str(int(time.time()))
            string_to_sign = f"POST\n/v1/identify\n{self.access_key}\n{timestamp}"
            signature = base64.b64encode(
                hmac.new(
                    self.access_secret.encode(),
                    string_to_sign.encode(),
                    hashlib.sha1
                ).digest()
            ).decode()

            # API request
            files = {'sample': audio_data}
            params = {
                'access_key': self.access_key,
                'signature': signature,
                'timestamp': timestamp,
                'data_type': 'audio',
                'sample_bytes': len(audio_data)
            }

            response = requests.post(
                f"https://{self.host}/v1/identify",
                files=files,
                params=params,
                timeout=10
            )

            result = response.json()

            if result.get('status', {}).get('code') == 0:
                music = result.get('metadata', {}).get('music', [{}])[0]
                identified = {
                    'is_viral': True,
                    'sound_name': music.get('title', 'Unknown'),
                    'artist': music.get('artists', [{}])[0].get('name', 'Unknown'),
                    'confidence': result.get('status', {}).get('confidence', 0)
                }
                print(f"[SUCCESS] Viral sound detected: {identified['sound_name']} by {identified['artist']}")
            else:
                identified = {'is_viral': False, 'reason': 'Original audio'}

            # Cache result
            self.cache[audio_hash] = identified
            self.save_cache()

            return identified

        except Exception as e:
            print(f"[ERROR] ACRCloud failed: {e}")
            return {'is_viral': False, 'error': str(e)}

    def get_audio_hash(self, audio_path: str) -> str:
        """Generate hash for audio file."""
        try:
            with open(audio_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            print(f"[ERROR] Could not hash audio file: {e}")
            return str(time.time())

    def load_cache(self):
        """Load cached audio identifications."""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
            else:
                self.cache = {}
        except Exception as e:
            print(f"[WARNING] Could not load audio cache: {e}")
            self.cache = {}

    def save_cache(self):
        """Save audio identifications to cache."""
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"[WARNING] Could not save audio cache: {e}")


def enhanced_audio_analysis(audio_path: str) -> dict:
    """Comprehensive audio analysis with viral sound check."""
    import time
    start_time = time.time()

    analysis = {
        'viral_sound': {'is_viral': False},
        'tempo': 0,
        'energy': 0,
        'is_music': False,
        'error': None
    }

    try:
        # Try to load librosa for advanced audio analysis
        try:
            import librosa
            import numpy as np

            librosa_start = time.time()
            # Load audio (reduced to 15s for faster analysis)
            y, sr = librosa.load(audio_path, duration=15)

            # Basic audio features
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            energy = np.mean(librosa.feature.rms(y=y))

            # Convert to Python floats (tempo can be numpy array)
            tempo_float = float(np.asarray(tempo).item() if hasattr(tempo, 'item') else tempo)
            energy_float = float(energy)

            analysis['tempo'] = tempo_float
            analysis['energy'] = energy_float
            analysis['is_music'] = energy_float > 0.05

            librosa_time = time.time() - librosa_start
            print(f"[INFO] Audio features: tempo={tempo_float:.1f}bpm, energy={energy_float:.3f} ({librosa_time:.1f}s)")

        except ImportError:
            print("[INFO] librosa not available, skipping advanced audio analysis")
        except Exception as e:
            print(f"[WARNING] Audio feature extraction failed: {e}")

        # Check for viral sound
        acr_start = time.time()
        detector = ViralSoundDetector()
        viral_check = detector.identify(audio_path)
        analysis['viral_sound'] = viral_check
        acr_time = time.time() - acr_start

        if viral_check.get('is_viral'):
            print(f"[VIRAL SOUND] {viral_check.get('sound_name')} by {viral_check.get('artist')} (ACRCloud: {acr_time:.1f}s)")
        else:
            print(f"[INFO] ACRCloud viral check complete ({acr_time:.1f}s)")

    except Exception as e:
        print(f"[ERROR] Audio analysis failed: {e}")
        analysis['error'] = str(e)

    total_time = time.time() - start_time
    print(f"[TIMING] Total audio analysis: {total_time:.1f}s")
    return analysis


def analyze_audio_characteristics(audio_path: str) -> dict:
    """
    Analyze audio characteristics for content strategy insights.
    Returns info about music type, speech patterns, etc.
    """
    characteristics = {
        'has_background_music': False,
        'has_voiceover': False,
        'audio_type': 'unknown',
        'energy_level': 'medium'
    }

    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(audio_path)

        # Energy analysis
        rms = librosa.feature.rms(y=y)[0]
        avg_energy = np.mean(rms)

        if avg_energy > 0.1:
            characteristics['energy_level'] = 'high'
        elif avg_energy < 0.03:
            characteristics['energy_level'] = 'low'

        # Spectral analysis
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(spectral_centroids)

        # Rough heuristics for content type
        if avg_centroid > 3000 and avg_energy > 0.05:
            characteristics['audio_type'] = 'music_with_speech'
            characteristics['has_background_music'] = True
            characteristics['has_voiceover'] = True
        elif avg_energy > 0.05:
            characteristics['audio_type'] = 'music'
            characteristics['has_background_music'] = True
        elif avg_centroid > 2000:
            characteristics['audio_type'] = 'speech'
            characteristics['has_voiceover'] = True
        else:
            characteristics['audio_type'] = 'ambient'

        print(f"[INFO] Audio type: {characteristics['audio_type']}, energy: {characteristics['energy_level']}")

    except ImportError:
        print("[INFO] librosa not available for audio characteristics")
    except Exception as e:
        print(f"[WARNING] Audio characteristics analysis failed: {e}")

    return characteristics
