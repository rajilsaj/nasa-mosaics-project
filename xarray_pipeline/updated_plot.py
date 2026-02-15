from pds4_tools import pds4_read
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LogNorm
import datetime

structure = pds4_read('ELS_201215406_V01.xml')
table = structure[0]
df = table.to_dataframe()
df['UTC'] = pd.to_datetime(df['UTC'])

# Extract DATA and energy bins
data_matrix = np.vstack(df['GROUP_1, DATA'].apply(lambda x: x.reshape(8,63))).T
energy = df['GROUP_2, DIM1_E'].iloc[0]

# Use raw data (not log-transformed) - will use log scale on colorbar
data_counts = data_matrix

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Determine normalization for colorbar - use percentile to remove outliers
data_positive = data_counts[data_counts > 0]
if len(data_positive) > 0:
    vmin = max(1e0, data_positive.min())
    # Use 95th percentile to cap outliers and reduce yellow spots
    vmax = np.percentile(data_positive, 95)
    norm = LogNorm(vmin=vmin, vmax=vmax)
else:
    norm = None

# Plot using pcolormesh
pcm = ax.pcolormesh(df['UTC'], energy, data_counts, shading='auto', cmap='viridis', norm=norm)

# Customize axes
ax.set_xlabel('Date/Time')
ax.set_ylabel('Energy (eV/q)')
ax.set_yscale('log')
ax.set_ylim(bottom=1.0)  # Set y-axis minimum to 10^0 (1)

# Format time axis - DD-MM-YYYY/HH:MM format with rotation
ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y/%H:%M'))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

# Add vertical bars (red at ~19:00, black at ~22:45)
# Use the first date from the time series
if len(df['UTC']) > 0:
    first_date = df['UTC'].iloc[0].date()
    red_bar_time = datetime.datetime.combine(first_date, datetime.time(19, 0))
    black_bar_time = datetime.datetime.combine(first_date, datetime.time(22, 45))
    
    # Convert to matplotlib date numbers
    time_start = mdates.date2num(df['UTC'].iloc[0])
    time_end = mdates.date2num(df['UTC'].iloc[-1])
    red_bar_num = mdates.date2num(red_bar_time)
    black_bar_num = mdates.date2num(black_bar_time)
    
    # Draw vertical bars if within data range
    if time_start <= red_bar_num <= time_end:
        ax.axvline(red_bar_num, color='red', linewidth=2, alpha=0.8)
    if time_start <= black_bar_num <= time_end:
        ax.axvline(black_bar_num, color='black', linewidth=2, alpha=0.8)

# Colorbar with log scale
cbar = fig.colorbar(pcm, ax=ax)
cbar.set_label('Interpolated Counts / s')
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0f}'))

plt.tight_layout()
plt.show()

