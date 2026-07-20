import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Pi0Downloader")

def download_pi0():
    logger.info("Starting download of pi0_base from Google Cloud Storage...")
    
    # Add openpi to path so we can import it
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../openpi/src"))
    
    from openpi.shared import download
    
    # Set the cache dir
    os.environ["OPENPI_DATA_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights")
    
    # Fix for gsutil read-only file system error
    os.environ["BOTO_CONFIG"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/.boto")
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../weights/.gsutil"), exist_ok=True)
    
    checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_base")
    logger.info(f"Download complete! Weights saved to {checkpoint_dir}")

if __name__ == "__main__":
    download_pi0()