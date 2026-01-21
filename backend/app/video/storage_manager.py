"""Video file storage and management utilities."""

import os
import shutil
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)


class VideoStorageManager:
    """Manages storage of uploaded videos and analysis results."""

    def __init__(self, base_path: str = "uploads/videos"):
        """
        Initialize storage manager.

        Args:
            base_path: Base directory for video storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_uploaded_video(
        self,
        video_data: bytes,
        manual_id: str,
        original_filename: str
    ) -> tuple[str, str]:
        """
        Save an uploaded video file.

        Args:
            video_data: Video file bytes
            manual_id: Manual ID this video belongs to
            original_filename: Original filename from upload

        Returns:
            (video_id, saved_path) tuple
        """
        # Generate unique video ID
        video_id = str(uuid.uuid4())[:8]

        # Create manual-specific directory
        manual_dir = self.base_path / manual_id
        manual_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        ext = Path(original_filename).suffix.lower()
        if not ext:
            ext = '.mp4'  # Default to mp4

        # Save file
        filename = f"{video_id}_assembly{ext}"
        video_path = manual_dir / filename

        with open(video_path, 'wb') as f:
            f.write(video_data)

        logger.info(f"Saved video {video_id} to {video_path}")
        return video_id, str(video_path)

    def get_video_path(self, manual_id: str, video_id: str) -> Optional[str]:
        """
        Get the path to a stored video.

        Args:
            manual_id: Manual ID
            video_id: Video ID

        Returns:
            Path to video file, or None if not found
        """
        manual_dir = self.base_path / manual_id

        if not manual_dir.exists():
            return None

        # Search for file with this video_id prefix
        for file in manual_dir.glob(f"{video_id}_*"):
            return str(file)

        return None

    def save_overlay_video(
        self,
        overlay_path: str,
        manual_id: str,
        video_id: str
    ) -> str:
        """
        Save a generated overlay video.

        Args:
            overlay_path: Path to temporary overlay video
            manual_id: Manual ID
            video_id: Original video ID

        Returns:
            Path to saved overlay video
        """
        manual_dir = self.base_path / manual_id
        manual_dir.mkdir(parents=True, exist_ok=True)

        # Save with _overlay suffix
        overlay_filename = f"{video_id}_overlay.mp4"
        dest_path = manual_dir / overlay_filename

        shutil.copy2(overlay_path, dest_path)

        logger.info(f"Saved overlay video to {dest_path}")
        return str(dest_path)

    def get_overlay_path(self, manual_id: str, video_id: str) -> Optional[str]:
        """
        Get the path to an overlay video.

        Args:
            manual_id: Manual ID
            video_id: Video ID

        Returns:
            Path to overlay video, or None if not found
        """
        manual_dir = self.base_path / manual_id
        overlay_path = manual_dir / f"{video_id}_overlay.mp4"

        if overlay_path.exists():
            return str(overlay_path)
        return None

    def cleanup_old_videos(self, days: int = 7) -> int:
        """
        Remove videos older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for manual_dir in self.base_path.iterdir():
            if not manual_dir.is_dir():
                continue

            for video_file in manual_dir.glob("*_assembly.*"):
                if datetime.fromtimestamp(video_file.stat().st_mtime) < cutoff_time:
                    video_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old video: {video_file}")

        return deleted_count

    def get_storage_stats(self) -> dict:
        """
        Get storage statistics.

        Returns:
            Dictionary with storage info
        """
        total_size = 0
        video_count = 0
        overlay_count = 0

        for manual_dir in self.base_path.iterdir():
            if not manual_dir.is_dir():
                continue

            for file in manual_dir.iterdir():
                if file.is_file():
                    total_size += file.stat().st_size
                    if "_overlay" in file.name:
                        overlay_count += 1
                    else:
                        video_count += 1

        return {
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "video_count": video_count,
            "overlay_count": overlay_count,
            "storage_path": str(self.base_path)
        }


class AnalysisStorageManager:
    """Manages storage of video analysis results."""

    def __init__(self, base_path: str = "output/video_analysis"):
        """
        Initialize analysis storage manager.

        Args:
            base_path: Base directory for analysis results
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_analysis_results(
        self,
        results: dict,
        manual_id: str,
        analysis_id: str
    ) -> str:
        """
        Save video analysis results.

        Args:
            results: Analysis results dictionary
            manual_id: Manual ID
            analysis_id: Analysis ID

        Returns:
            Path to saved results file
        """
        import json

        manual_dir = self.base_path / manual_id
        manual_dir.mkdir(parents=True, exist_ok=True)

        results_file = manual_dir / f"{analysis_id}_results.json"

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved analysis results to {results_file}")
        return str(results_file)

    def load_analysis_results(
        self,
        manual_id: str,
        analysis_id: str
    ) -> Optional[dict]:
        """
        Load analysis results.

        Args:
            manual_id: Manual ID
            analysis_id: Analysis ID

        Returns:
            Results dictionary, or None if not found
        """
        import json

        results_file = self.base_path / manual_id / f"{analysis_id}_results.json"

        if not results_file.exists():
            return None

        with open(results_file, 'r') as f:
            return json.load(f)

    def cleanup_old_analyses(self, days: int = 30) -> int:
        """
        Remove analysis results older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of files deleted
        """
        cutoff_time = datetime.now() - timedelta(days=days)
        deleted_count = 0

        for manual_dir in self.base_path.iterdir():
            if not manual_dir.is_dir():
                continue

            for results_file in manual_dir.glob("*_results.json"):
                if datetime.fromtimestamp(results_file.stat().st_mtime) < cutoff_time:
                    results_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old analysis: {results_file}")

        return deleted_count
