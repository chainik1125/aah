"""
Utilities for better progress bar handling with tqdm.
"""
import sys
from contextlib import contextmanager
from tqdm import tqdm as original_tqdm


class TqdmPrintRedirect:
    """Redirect print statements to tqdm.write() to avoid breaking progress bars."""
    
    def __init__(self):
        self.original_print = print
        
    def tqdm_print(*args, **kwargs):
        """Print that works nicely with tqdm progress bars."""
        # Convert args to string like normal print would
        output = ' '.join(str(arg) for arg in args)
        # Use tqdm.write which handles progress bars properly
        original_tqdm.write(output)
    
    def __enter__(self):
        # Monkey-patch the built-in print
        import builtins
        builtins.print = self.tqdm_print
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original print
        import builtins
        builtins.print = self.original_print


@contextmanager
def tqdm_redirect_print():
    """Context manager to redirect prints through tqdm.write()."""
    redirector = TqdmPrintRedirect()
    with redirector:
        yield


def create_progress_bars(n_outer, n_inner, outer_desc="Outer", inner_desc="Inner"):
    """
    Create nested progress bars that stay visible.
    
    Returns:
        outer_pbar, inner_pbar
    """
    # Create progress bars with explicit positions
    outer_pbar = original_tqdm(
        total=n_outer,
        desc=outer_desc,
        position=0,
        leave=True,
        ncols=100,
        file=sys.stdout,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    inner_pbar = original_tqdm(
        total=n_inner,
        desc=inner_desc,
        position=1,
        leave=True,
        ncols=100,
        file=sys.stdout,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
    )
    
    return outer_pbar, inner_pbar


def safe_print(*args, **kwargs):
    """Print that won't break tqdm progress bars."""
    original_tqdm.write(' '.join(str(arg) for arg in args))