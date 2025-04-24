import pandas as pd
from typing import List, Tuple

def get_true_ranges(df: pd.DataFrame, condition_col: str, index_col: str) -> List[Tuple[int, int]]:
    """
    Extracts contiguous ranges where a boolean column is True and maps those to corresponding
    values from another column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the columns to analyze.
    condition_col : str
        Name of the boolean condition column (e.g., 'gt_detection_win').
    index_col : str
        Column from which to extract the start/end values (e.g., 'SCLK').

    Returns
    -------
    List[Tuple[int, int]]
        A list of (start_value, end_value) tuples where the condition was True.
    """
    ranges = []
    in_range = False
    start_value = None

    for condition, value in zip(df[condition_col], df[index_col]):
        if condition and not in_range:
            start_value = int(value)
            in_range = True
        elif not condition and in_range:
            end_value = int(value)
            ranges.append((start_value, end_value))
            in_range = False

    # Catch edge case if the last condition stays True until the end
    if in_range and start_value is not None:
        ranges.append((start_value, int(df[index_col].iloc[-1])))

    return ranges


if __name__ == "__main__":
    # Load data and run example
    df = pd.read_csv("ml_ready_vortex_data.csv")
    true_ranges = get_true_ranges(df, "gt_detection_win", "SCLK")

    print("Detected Ranges (start SCLK, end SCLK):")
    for r in true_ranges[:10]:
        print(r)
    
    print(f"\nTotal detected true ranges: {len(true_ranges)}")
