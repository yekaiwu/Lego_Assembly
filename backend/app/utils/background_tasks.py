"""Background task management for long-running operations."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


class TaskManager:
    """Manages status tracking for background tasks."""

    def __init__(self, cache_dir: str = "backend/cache/analysis_status"):
        """
        Initialize task manager.

        Args:
            cache_dir: Directory for task status files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def create_task(
        self,
        task_id: str,
        task_type: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Create a new task.

        Args:
            task_id: Unique task identifier
            task_type: Type of task (analysis, overlay, etc.)
            metadata: Optional metadata

        Returns:
            Task info dictionary
        """
        task_info = {
            "task_id": task_id,
            "task_type": task_type,
            "status": TaskStatus.PENDING,
            "progress_percentage": 0,
            "current_step": "Initializing",
            "created_at": datetime.now().isoformat(),
            "started_at": None,
            "completed_at": None,
            "error": None,
            "metadata": metadata or {}
        }

        self._save_task(task_id, task_info)
        logger.info(f"Created task {task_id} of type {task_type}")
        return task_info

    def update_task(
        self,
        task_id: str,
        status: Optional[TaskStatus] = None,
        progress: Optional[int] = None,
        current_step: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Update task status.

        Args:
            task_id: Task identifier
            status: New status
            progress: Progress percentage (0-100)
            current_step: Description of current step
            error: Error message if failed
        """
        task_info = self.get_task(task_id)
        if not task_info:
            logger.warning(f"Task {task_id} not found for update")
            return

        if status:
            task_info["status"] = status

            if status == TaskStatus.PROCESSING and not task_info.get("started_at"):
                task_info["started_at"] = datetime.now().isoformat()

            if status in (TaskStatus.COMPLETED, TaskStatus.ERROR):
                task_info["completed_at"] = datetime.now().isoformat()

                # Calculate duration
                if task_info.get("started_at"):
                    start = datetime.fromisoformat(task_info["started_at"])
                    end = datetime.fromisoformat(task_info["completed_at"])
                    task_info["duration_sec"] = (end - start).total_seconds()

        if progress is not None:
            task_info["progress_percentage"] = max(0, min(100, progress))

        if current_step:
            task_info["current_step"] = current_step

        if error:
            task_info["error"] = error
            task_info["status"] = TaskStatus.ERROR

        self._save_task(task_id, task_info)
        logger.debug(f"Updated task {task_id}: {status}, {progress}%, {current_step}")

    def get_task(self, task_id: str) -> Optional[Dict]:
        """
        Get task information.

        Args:
            task_id: Task identifier

        Returns:
            Task info dict or None
        """
        task_file = self.cache_dir / f"{task_id}.json"

        if not task_file.exists():
            return None

        try:
            with open(task_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading task {task_id}: {e}")
            return None

    def delete_task(self, task_id: str):
        """
        Delete task status file.

        Args:
            task_id: Task identifier
        """
        task_file = self.cache_dir / f"{task_id}.json"

        if task_file.exists():
            task_file.unlink()
            logger.info(f"Deleted task {task_id}")

    def _save_task(self, task_id: str, task_info: Dict):
        """Save task info to file."""
        task_file = self.cache_dir / f"{task_id}.json"

        with open(task_file, 'w') as f:
            json.dump(task_info, f, indent=2)

    def cleanup_old_tasks(self, days: int = 7) -> int:
        """
        Remove task files older than specified days.

        Args:
            days: Age threshold in days

        Returns:
            Number of tasks deleted
        """
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(days=days)
        deleted = 0

        for task_file in self.cache_dir.glob("*.json"):
            try:
                with open(task_file, 'r') as f:
                    task_info = json.load(f)

                created_at = task_info.get("created_at")
                if created_at:
                    created = datetime.fromisoformat(created_at)
                    if created < cutoff:
                        task_file.unlink()
                        deleted += 1
                        logger.info(f"Deleted old task: {task_file.stem}")
            except Exception as e:
                logger.error(f"Error processing {task_file}: {e}")

        return deleted
