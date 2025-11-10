"""Track prediction accuracy for continuous improvement."""
import sqlite3
import json
from datetime import datetime
import os


class AnalysisPerformanceTracker:
    def __init__(self, db_path: str = "./data/predictions.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self):
        """Initialize the database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                video_url TEXT PRIMARY KEY,
                timestamp DATETIME,
                predicted_performance TEXT,
                actual_views INTEGER,
                actual_likes INTEGER,
                actual_saves INTEGER,
                actual_comments INTEGER,
                actual_shares INTEGER,
                save_rate REAL,
                engagement_rate REAL,
                metadata TEXT,
                analysis_summary TEXT
            )
        ''')
        conn.commit()
        conn.close()
        print("[INFO] Performance tracking database initialized")

    def record_prediction(self, video_url: str, metadata: dict, analysis: dict):
        """Store prediction with save data and full metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Extract metrics
            view_count = metadata.get('view_count', 0)
            like_count = metadata.get('like_count', 0)
            save_count = metadata.get('save_count', 0)
            comment_count = metadata.get('comment_count', 0)
            share_count = metadata.get('share_count', 0)

            # Calculate rates
            engagement_metrics = metadata.get('engagement_metrics', {})
            save_rate = engagement_metrics.get('save_rate', 0)
            engagement_rate = engagement_metrics.get('total_engagement_rate', 0)

            # Create analysis summary
            analysis_summary = {
                'performance_level': analysis.get('performance_level', 'unknown'),
                'hook_score': analysis.get('scores', {}).get('hook_strength', 0),
                'viral_potential': analysis.get('scores', {}).get('viral_potential', 0),
                'predicted_category': metadata.get('performance_level', 'unknown')
            }

            cursor.execute('''
                INSERT OR REPLACE INTO predictions
                (video_url, timestamp, predicted_performance, actual_views, actual_likes,
                 actual_saves, actual_comments, actual_shares, save_rate, engagement_rate,
                 metadata, analysis_summary)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                video_url,
                datetime.now().isoformat(),
                analysis.get('performance_level', 'unknown'),
                view_count,
                like_count,
                save_count,
                comment_count,
                share_count,
                save_rate,
                engagement_rate,
                json.dumps(metadata),
                json.dumps(analysis_summary)
            ))

            conn.commit()
            conn.close()
            print(f"[TRACKER] Recorded prediction for {video_url[:50]}...")

        except Exception as e:
            print(f"[ERROR] Failed to record prediction: {e}")

    def get_prediction_accuracy(self) -> dict:
        """Analyze prediction accuracy over time."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT predicted_performance, actual_views, save_rate, engagement_rate
                FROM predictions
                WHERE actual_views > 0
            ''')

            results = cursor.fetchall()
            conn.close()

            if not results:
                return {'total_predictions': 0, 'message': 'No predictions recorded yet'}

            # Analyze accuracy
            accuracy_stats = {
                'total_predictions': len(results),
                'viral_predicted': 0,
                'viral_actual': 0,
                'high_save_rate_videos': 0,
                'avg_save_rate': 0,
                'avg_engagement_rate': 0
            }

            save_rates = []
            engagement_rates = []

            for predicted, views, save_rate, engagement_rate in results:
                if predicted == 'viral':
                    accuracy_stats['viral_predicted'] += 1
                if views >= 1000000:
                    accuracy_stats['viral_actual'] += 1
                if save_rate > 1.5:
                    accuracy_stats['high_save_rate_videos'] += 1

                save_rates.append(save_rate or 0)
                engagement_rates.append(engagement_rate or 0)

            accuracy_stats['avg_save_rate'] = sum(save_rates) / len(save_rates) if save_rates else 0
            accuracy_stats['avg_engagement_rate'] = sum(engagement_rates) / len(engagement_rates) if engagement_rates else 0

            return accuracy_stats

        except Exception as e:
            print(f"[ERROR] Failed to get accuracy stats: {e}")
            return {'error': str(e)}

    def get_top_performers(self, limit: int = 10) -> list:
        """Get videos with best performance."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT video_url, actual_views, actual_saves, save_rate, engagement_rate, timestamp
                FROM predictions
                ORDER BY actual_views DESC
                LIMIT ?
            ''', (limit,))

            results = cursor.fetchall()
            conn.close()

            performers = []
            for url, views, saves, save_rate, engagement_rate, timestamp in results:
                performers.append({
                    'url': url,
                    'views': views,
                    'saves': saves,
                    'save_rate': save_rate,
                    'engagement_rate': engagement_rate,
                    'analyzed_at': timestamp
                })

            return performers

        except Exception as e:
            print(f"[ERROR] Failed to get top performers: {e}")
            return []

    def export_data(self, output_path: str = "./data/predictions_export.json"):
        """Export all prediction data for analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('SELECT * FROM predictions')
            columns = [description[0] for description in cursor.description]
            results = cursor.fetchall()
            conn.close()

            data = []
            for row in results:
                data.append(dict(zip(columns, row)))

            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[TRACKER] Exported {len(data)} predictions to {output_path}")
            return True

        except Exception as e:
            print(f"[ERROR] Failed to export data: {e}")
            return False
