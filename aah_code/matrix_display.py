import numpy as np
import pandas as pd

try:
    # Optional, but gives nicer ASCII/Unicode tables
    from tabulate import tabulate  # pip install tabulate
    _HAVE_TABULATE = True
except ImportError:
    _HAVE_TABULATE = False


def matrix_to_dataframe(H, basis_labels=None, precision=3):
    """
    Wrap an (n×n) array in a labelled pandas.DataFrame.

    Parameters
    ----------
    H : array_like, shape (n, n)
        The Hamiltonian (or any numeric matrix).
    basis_labels : list[str] | None
        Labels for each basis ket (uses |0>, |1>, … if None).
    precision : int
        Number of decimal places shown when printed.

    Returns
    -------
    df : pandas.DataFrame
    """
    H = np.asarray(H)
    n = H.shape[0]
    if basis_labels is None:
        basis_labels = [f"|{i}⟩" for i in range(n)]

    # Round elements just for display (keeps original array untouched)
    df = pd.DataFrame(np.round(H, precision), index=basis_labels, columns=basis_labels)
    pd.set_option("display.precision", precision)  # affects all future prints  📏
    return df


def print_matrix(df, style="pandas", **tabulate_kwargs):
    """
    Pretty‑print the DataFrame in one of three styles.

    Parameters
    ----------
    df : pandas.DataFrame
    style : {"pandas", "tabulate", "latex"}
        - "pandas"   → plain DataFrame.to_string()  (no extra deps)
        - "tabulate" → ASCII/Unicode table via tabulate
        - "latex"    → df.to_latex() (copy‑paste into TeX)
    **tabulate_kwargs
        Extra keyword args forwarded to tabulate(), e.g. tablefmt="psql".
    """
    if style == "pandas":
        print(df.to_string())                                         # :contentReference[oaicite:0]{index=0}
    elif style == "tabulate":
        if not _HAVE_TABULATE:
            raise ImportError("pip install tabulate to use this style")
        print(tabulate(df, headers="keys",
                       tablefmt=tabulate_kwargs.get("tablefmt", "psql")))  # :contentReference[oaicite:1]{index=1}
    elif style == "latex":
        print(df.to_latex())                                          # :contentReference[oaicite:2]{index=2}
    else:
        raise ValueError("style must be 'pandas', 'tabulate', or 'latex'.")


# ------------------- demo usage --------------------------------------------
if __name__ == "__main__":
    # Dummy 4×4 matrix just to illustrate
    H_demo = np.array([[0, 1, 0.5, 0],
                       [1, 0, 1,   0.5],
                       [0.5, 1, 0, 1],
                       [0, 0.5, 1, 0]])

    labels = [r"$|\!\uparrow,1⟩$", r"$|\!\uparrow,2⟩$",
              r"$|\!\downarrow,1⟩$", r"$|\!\downarrow,2⟩$"]

    df = matrix_to_dataframe(H_demo, labels, precision=2)  # build table

    # Choose one:
    print_matrix(df, style="pandas")                 # simple built‑in table
    # print_matrix(df, style="tabulate", tablefmt="fancy_grid")   # needs tabulate
    # print_matrix(df, style="latex")                 # copy into LaTeX
