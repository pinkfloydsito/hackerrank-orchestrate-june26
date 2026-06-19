"""Roboflow dataset downloader with error handling and logging."""

import logging
import time
from pathlib import Path
from typing import List, Tuple

import dotenv

from hackerrank_orchestrate.config import PROJECT_ROOT, RAW_DATA_DIR, ROBOFLOW_DATASETS
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def download_dataset(
    workspace: str,
    project: str,
    version: int,
    target_dir: Path,
) -> None:
    """Download a single dataset from Roboflow.
    
    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        target_dir: Directory to save the downloaded dataset.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        logger.error("roboflow package not installed. Run: pip install roboflow")
        raise

    # Load API key from .env
    dotenv.load_dotenv(PROJECT_ROOT / ".env")
    import os

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY not found in .env. "
            "Create .env file with: ROBOFLOW_API_KEY=your_key"
        )

    logger.info(f"Connecting to Roboflow (key prefix: {api_key[:4]}...)")
    rf = Roboflow(api_key=api_key)

    logger.info(f"Accessing workspace: {workspace}, project: {project}")
    project_obj = rf.workspace(workspace).project(project)

    logger.info(f"Downloading version {version} in COCO format to {target_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Change to target directory for download
    original_dir = Path.cwd()
    try:
        import os as os_module
        os_module.chdir(target_dir)
        dataset = project_obj.version(version).download(model_format="coco")
        logger.info(f"Download complete: {workspace}/{project} v{version}")
        
        if hasattr(dataset, "location") and dataset.location:
            logger.info(f"Dataset location: {dataset.location}")
            loc = Path(dataset.location)
            if loc.exists():
                files = list(loc.rglob("*"))
                logger.info(f"Files downloaded: {len(files)}")
                for f in files[:10]:
                    logger.info(f"  - {f.relative_to(loc)}")
                if len(files) > 10:
                    logger.info(f"  ... and {len(files) - 10} more files")
    finally:
        os_module.chdir(original_dir)


def download_all_datasets() -> None:
    """Download all configured Roboflow datasets."""
    logger.info(f"Starting dataset downloads. Target directory: {RAW_DATA_DIR}")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for workspace, project, version, object_type in ROBOFLOW_DATASETS:
        target_dir = RAW_DATA_DIR / object_type
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Downloading: {workspace}/{project} v{version} -> {object_type}")
        logger.info(f"{'='*60}")

        try:
            download_dataset(workspace, project, version, target_dir)
            logger.info(f"SUCCESS: {workspace}/{project}")
        except Exception as e:
            logger.error(f"FAILED to download {workspace}/{project}: {e}")
            import traceback

            logger.error(traceback.format_exc())

        # Rate limit between downloads
        time.sleep(2)

    logger.info("\n" + "="*60)
    logger.info("All downloads attempted. Check logs for details.")


def verify_downloads() -> List[Tuple[str, bool, int]]:
    """Verify downloaded datasets and return status.
    
    Returns:
        List of tuples: (object_type, has_data, file_count)
    """
    results = []
    for object_type in ["car", "laptop", "package"]:
        dataset_dir = RAW_DATA_DIR / object_type
        if dataset_dir.exists():
            files = list(dataset_dir.rglob("*"))
            image_files = [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            has_data = len(image_files) > 0
            results.append((object_type, has_data, len(image_files)))
            logger.info(f"{object_type}: {len(image_files)} images found")
        else:
            results.append((object_type, False, 0))
            logger.warning(f"{object_type}: No data directory found")
    return results


def main() -> None:
    """CLI entry point for downloading datasets."""
    download_all_datasets()
    verify_downloads()


if __name__ == "__main__":
    main()
