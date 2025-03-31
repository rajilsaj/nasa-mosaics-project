import pandas as pd

def get_true_ranges(df, column1, column2):
    """
    Returns a list of tuples (start_index, end_index) for each contiguous block of True values in the specified column.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the boolean column.
        column (str): Name of the boolean column in the DataFrame.
    
    Returns:
        list of tuples: Each tuple represents (start_index, end_index) for a block of True values.
    """
    ranges = []
    in_range = False
    start_index = None
    i = 0
    for val1, val2 in zip(df[column1], df[column2]):
        if val1 and not in_range:
            start = int(val2)
            # Start of a new True block
            in_range = True
        elif not val1 and in_range:
            # End of a True block
            end = int(val2) - 1
            ranges.append((start, end))
            in_range = False
        i += 1
    
    # If the DataFrame ends while still in a True block, capture that block
    if in_range:
        ranges.append((start_index, df.index[-1]))
    
    return ranges

if __name__ == '__main__':
    # Example usage:
    
    df = pd.read_csv('ml_ready_vortex_data.csv')
    
    true_ranges = get_true_ranges(df, 'gt_detection_win', "SCLK")
    print(true_ranges)
    print(len(true_ranges))