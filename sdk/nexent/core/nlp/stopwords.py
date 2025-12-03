import logging
import os
import tempfile

import requests
from jieba import analyse

logger = logging.getLogger("nlp.stopwords")


def download_stopwords(url: str, save_path: str) -> bool:
    """Download stopwords file"""
    try:
        logger.info(f"Downloading stopwords: {url}")
        response = requests.get(url, timeout=10)
        response.encoding = 'utf-8'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        logger.info(f"Stopwords saved to: {os.path.abspath(save_path)}")
        return True
    except Exception as e:
        logger.info(f"Failed to download stopwords: {str(e)}")
        return False


def load_stopwords(stopwords_name='baidu_stopwords.txt',
                   backup_url='https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt'):
    """
    Load stopwords (automatically download to temporary file)

    Args:
        stopwords_name: Name of the stopwords file
        backup_url: Backup download URL
    """
    # Define the path where the stopwords file should be in Docker container
    docker_stopwords_path = "/opt/baidu_stopwords.txt"
    
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    stopwords_path = os.path.join(temp_dir, stopwords_name)
    
    # Check if the stopwords file exists in the Docker container
    if os.path.exists(docker_stopwords_path):
        try:
            # Copy the Docker file to temp directory
            with open(docker_stopwords_path, 'r', encoding='utf-8') as src, \
                 open(stopwords_path, 'w', encoding='utf-8') as dst:
                dst.write(src.read())
            logger.info(f"Using Docker stopwords file: {docker_stopwords_path}")
            logger.info(f"Copied to temporary location: {stopwords_path}")
        except Exception as e:
            logger.error(f"Failed to copy Docker stopwords file: {e}")
            # Fall back to downloading
            if not download_stopwords(backup_url, stopwords_path):
                logger.warning(f"Unable to download stopwords from: {backup_url}")
                # Create a minimal stopwords file as fallback
                logger.info("Creating minimal fallback stopwords file")
                with open(stopwords_path, 'w', encoding='utf-8') as f:
                    f.write("# Minimal stopwords list\n")
                    f.write("的\n")
                    f.write("了\n")
                    f.write("在\n")
                    f.write("是\n")
                    f.write("我\n")
                    f.write("有\n")
                    f.write("和\n")
                    f.write("就\n")
                    f.write("不\n")
                    f.write("人\n")
                    f.write("都\n")
                    f.write("一\n")
                    f.write("一个\n")
    else:
        # Docker file doesn't exist, try to download
        logger.warning(f"Docker stopwords file not found at: {docker_stopwords_path}")
        if not download_stopwords(backup_url, stopwords_path):
            logger.warning(f"Unable to download stopwords from: {backup_url}")
            # Check if we have a local copy from a previous run
            if os.path.exists(stopwords_path):
                logger.info("Using existing local copy of stopwords")
            else:
                # Create a minimal stopwords file as fallback
                logger.info("Creating minimal fallback stopwords file")
                with open(stopwords_path, 'w', encoding='utf-8') as f:
                    f.write("# Minimal stopwords list\n")
                    f.write("的\n")
                    f.write("了\n")
                    f.write("在\n")
                    f.write("是\n")
                    f.write("我\n")
                    f.write("有\n")
                    f.write("和\n")
                    f.write("就\n")
                    f.write("不\n")
                    f.write("人\n")
                    f.write("都\n")
                    f.write("一\n")
                    f.write("一个\n")

    # Validate file content
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            if not first_line or len(first_line) > 100:  # Simple format validation
                # Don't delete the file, just warn about it
                logger.warning("Stopwords file format might be invalid, continuing anyway")
    except Exception as e:
        logger.error(f"Error validating stopwords file: {e}")

    # Configure to jieba
    try:
        analyse.set_stop_words(stopwords_path)
        logger.info("Successfully loaded stopwords")
    except Exception as e:
        logger.error(f"Failed to set stopwords for jieba: {e}")