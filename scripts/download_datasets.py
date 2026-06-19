#!/usr/bin/env python3
"""Download Roboflow datasets for the challenge."""

from hackerrank_orchestrate.data.roboflow_downloader import download_all_datasets, verify_downloads

if __name__ == "__main__":
    download_all_datasets()
    verify_downloads()
