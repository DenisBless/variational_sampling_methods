import os
from datetime import datetime


PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))


def project_path(*args) -> str:
    """
    Abstraction from os.path.join()
    Builds absolute paths from relative path strings with ilfvrm package as root.
    If args already contains an absolute path, it is used as root for the subsequent joins
    Args:
        *args:

    Returns:
        absolute path

    """

    return os.path.abspath(os.path.join(PACKAGE_DIR, *args))


def make_model_dir(alg, exp, seed):
    base_dir = project_path('models')
    if not os.path.exists(base_dir):
        # Create the directory if it doesn't exist
        os.makedirs(base_dir)
    # Get the current date and time
    current_datetime = datetime.now()

    # Format the date and time to use in the directory name
    timestamp = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

    # Combine the base directory, prefix, and timestamp to create the directory path
    directory_path = os.path.join(base_dir, f'{alg}_{exp}_seed{seed}_{timestamp}')

    # Check if the directory already exists
    if not os.path.exists(directory_path):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path)

    return directory_path

if __name__ == '__main__':
    make_model_dir('mfvi', 'gmm', 0, 0)
