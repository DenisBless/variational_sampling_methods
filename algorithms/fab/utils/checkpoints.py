"""Code builds on https://github.com/lollcat/fab-jax"""
import os

def get_latest_checkpoint(dir_path: str, key: str = ''):
    """Get path to latest checkpoint in directory

    Args:
        dir_path: Path to directory to search for checkpoints
        key: Key which has to be in checkpoint name

    Returns:
        Path to latest checkpoint
    """
    if not os.path.exists(dir_path):
        return None
    checkpoints = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if
                   os.path.isfile(os.path.join(dir_path, f)) and key in f]
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return checkpoints[-1]