import os
import sys

def get_project_root():
    """
    Get the project root directory
    
    Returns:
        str: Path to the project root directory
    """
    # Check if IB_PROJECT_ROOT is set in environment
    if 'IB_PROJECT_ROOT' in os.environ:
        return os.environ['IB_PROJECT_ROOT']
    
    # Try to find the project root by looking for CMakeLists.txt
    # Start from the current file's directory and go up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up until we find CMakeLists.txt or hit the filesystem root
    while current_dir != os.path.dirname(current_dir):  # Stop at filesystem root
        if os.path.isfile(os.path.join(current_dir, 'CMakeLists.txt')):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # If we couldn't find it, use a reasonable default
    print("Warning: Could not find project root. Using current working directory.", file=sys.stderr)
    return os.getcwd()

def get_data_dir():
    """
    Get the data directory
    
    Returns:
        str: Path to the data directory
    """
    # Check if IB_DATA_DIR is set in environment
    if 'IB_DATA_DIR' in os.environ:
        return os.environ['IB_DATA_DIR']
    
    # Default to ~/Data
    return os.path.join(os.path.expanduser("~"), "Data")

def resolve_path(relative_path):
    """
    Resolve a path relative to the project root
    
    Args:
        relative_path (str): Path relative to project root
        
    Returns:
        str: Absolute path
    """
    if os.path.isabs(relative_path):
        return relative_path
        
    return os.path.join(get_project_root(), relative_path)

def resolve_data_path(filename):
    """
    Resolve a path to a data file
    
    Args:
        filename (str): Name of the data file
        
    Returns:
        str: Absolute path to the data file
    """
    if os.path.isabs(filename):
        return filename
        
    return os.path.join(get_data_dir(), filename)