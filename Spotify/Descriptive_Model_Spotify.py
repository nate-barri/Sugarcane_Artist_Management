import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

# Load the time series data
df = pd.read_csv('Spotify/2023-2025.csv')
df['date'] = pd.to_datetime(df['date'])

# Calculate growth percentages from first data point
df['followers_growth'] = ((df['followers'] - df['followers'].iloc[0]) / df['followers'].iloc[0]) * 100
df['streams_growth'] = ((df['streams'] - df['streams'].iloc[0]) / df['streams'].iloc[0]) * 100
df['listeners_growth'] = ((df['listeners'] - df['listeners'].iloc[0]) / df['listeners'].iloc[0]) * 100

# Song data
songs = pd.DataFrame({
    'song': ['Leonora', 'Kung Maging Akin Ka', 'Paruparo', 'Tanging Ikaw', 
             'Paalam, Leonora', 'Gunita', 'Dalangin', 'Maria Clara', 'Gabi', 
             'Bituin', 'Sinehan', 'Set You Free', 'Reside', 'Pagbigyan', 
             'Lagi Na Lang', 'Aking Maria Clara', 'Please', 'Tanging Ikaw - Harana'],
    'streams': [147271614, 30528694, 19633670, 17215577, 12382421, 10491792, 
                6610848, 3542129, 2736790, 2556841, 2524881, 2235932, 1732684, 
                1331076, 913222, 413891, 402313, 312292],
    'release_date': ['2023-03-31', '2021-02-13', '2022-07-08', '2025-01-31',
                     '2023-04-28', '2020-07-04', '2021-07-31', '2024-03-08',
                     '2020-01-24', '2019-07-21', '2023-09-22', '2023-11-24',
                     '2020-12-09', '2024-07-26', '2024-11-08', '2024-05-24',
                     '2025-10-03', '2025-07-04']
})
songs['release_date'] = pd.to_datetime(songs['release_date'])

# Filter songs released within the time series range
songs_in_range = songs[
    (songs['release_date'] >= df['date'].min()) & 
    (songs['release_date'] <= df['date'].max())
].copy()

# Print summary statistics
print("=" * 70)
print("SPOTIFY DESCRIPTIVE MODEL - CUMULATIVE GROWTH")
print("=" * 70)
print(f"\nTime Period: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Total Days: {len(df)}")
print(f"\n{'Metric':<20} {'Cumulative Growth':<20}")
print("-" * 70)
print(f"{'Followers':<20} {df['followers_growth'].iloc[-1]:>18.1f}%")
print(f"{'Daily Listeners':<20} {df['listeners_growth'].iloc[-1]:>18.1f}%")
print(f"{'Daily Streams':<20} {df['streams_growth'].iloc[-1]:>18.1f}%")

print(f"\n{'TOTALS':<30}")
print("-" * 70)
print(f"  Total Followers: {df['followers'].iloc[-1]:,}")
print(f"  Total Songs: {len(songs)}")
print(f"  Total All-Time Streams: {songs['streams'].sum():,}")
print(f"  Average Streams per Song: {songs['streams'].mean():,.0f}")
print(f"  Top Song: {songs.loc[songs['streams'].idxmax(), 'song']} ({songs['streams'].max():,} streams)")

# Create individual visualizations

# Figure 1: Daily Streams with Release Dates
plt.figure(figsize=(16, 7))
plt.plot(df['date'], df['streams'], linewidth=2, color='#1DB954', label='Daily Streams', alpha=0.8)
plt.ylabel('Daily Streams', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.title('Daily Streams with Song Releases', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')

# Add release markers with improved visibility
for _, song in songs_in_range.iterrows():
    closest_idx = (df['date'] - song['release_date']).abs().idxmin()
    stream_val = df.loc[closest_idx, 'streams']
    plt.scatter(song['release_date'], stream_val, color='#FF4444', s=200, zorder=5, 
                edgecolors='white', linewidth=2.5, marker='o', alpha=0.9)
    plt.annotate(song['song'], 
                xy=(song['release_date'], stream_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4444', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.5))

plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# Figure 2: Daily Listeners with Release Dates
plt.figure(figsize=(16, 7))
plt.plot(df['date'], df['listeners'], linewidth=2, color='#1E90FF', label='Daily Listeners', alpha=0.8)
plt.ylabel('Daily Listeners', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.title('Daily Listeners with Song Releases', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')

for _, song in songs_in_range.iterrows():
    closest_idx = (df['date'] - song['release_date']).abs().idxmin()
    listener_val = df.loc[closest_idx, 'listeners']
    plt.scatter(song['release_date'], listener_val, color='#FF4444', s=200, zorder=5, 
                edgecolors='white', linewidth=2.5, marker='o', alpha=0.9)
    plt.annotate(song['song'], 
                xy=(song['release_date'], listener_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4444', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.5))

plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# Figure 3: Follower Growth with Release Dates
plt.figure(figsize=(16, 7))
plt.plot(df['date'], df['followers'], linewidth=2, color='#9B59B6', label='Total Followers', alpha=0.8)
plt.ylabel('Total Followers', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.title('Follower Growth with Song Releases', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')

for _, song in songs_in_range.iterrows():
    closest_idx = (df['date'] - song['release_date']).abs().idxmin()
    follower_val = df.loc[closest_idx, 'followers']
    plt.scatter(song['release_date'], follower_val, color='#FF4444', s=200, zorder=5, 
                edgecolors='white', linewidth=2.5, marker='o', alpha=0.9)
    plt.annotate(song['song'], 
                xy=(song['release_date'], follower_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4444', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.5))

plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# Figure 4: Streams Growth Percentage with Release Dates
plt.figure(figsize=(16, 7))
plt.plot(df['date'], df['streams_growth'], linewidth=2, color='#FF6B6B', label='Streams Growth (%)', alpha=0.8)
plt.ylabel('Growth from Baseline (%)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.title('Streams Growth Percentage with Song Releases', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)

for _, song in songs_in_range.iterrows():
    closest_idx = (df['date'] - song['release_date']).abs().idxmin()
    growth_val = df.loc[closest_idx, 'streams_growth']
    plt.scatter(song['release_date'], growth_val, color='#FF4444', s=200, zorder=5, 
                edgecolors='white', linewidth=2.5, marker='o', alpha=0.9)
    plt.annotate(song['song'], 
                xy=(song['release_date'], growth_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4444', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.5))

plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# Figure 5: Followers Growth Percentage with Release Dates
plt.figure(figsize=(16, 7))
plt.plot(df['date'], df['followers_growth'], linewidth=2, color='#FFB347', label='Followers Growth (%)', alpha=0.8)
plt.ylabel('Growth from Baseline (%)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=14, fontweight='bold')
plt.title('Followers Growth Percentage with Song Releases', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=30, ha='right')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.3)

for _, song in songs_in_range.iterrows():
    closest_idx = (df['date'] - song['release_date']).abs().idxmin()
    growth_val = df.loc[closest_idx, 'followers_growth']
    plt.scatter(song['release_date'], growth_val, color='#FF4444', s=200, zorder=5, 
                edgecolors='white', linewidth=2.5, marker='o', alpha=0.9)
    plt.annotate(song['song'], 
                xy=(song['release_date'], growth_val),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#FF4444', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#FF4444', lw=1.5))

plt.legend(loc='upper left', fontsize=12, framealpha=0.9)
plt.tight_layout()
plt.show()

# Additional Analysis: Impact of releases
print(f"\n{'RELEASE IMPACT ANALYSIS':<70}")
print("=" * 70)
print(f"{'Song':<35} {'Release Date':<15} {'30-Day Avg Growth':<20}")
print("-" * 70)

# Sort by release date chronologically
songs_in_range_sorted = songs_in_range.sort_values('release_date')

for _, song in songs_in_range_sorted.iterrows():
    release_idx = (df['date'] - song['release_date']).abs().idxmin()
    before_start = max(0, release_idx - 30)
    after_end = min(len(df), release_idx + 30)
    
    if before_start < release_idx and release_idx < after_end:
        avg_before = df.loc[before_start:release_idx-1, 'streams'].mean()
        avg_after = df.loc[release_idx:after_end, 'streams'].mean()
        growth = ((avg_after / avg_before - 1) * 100) if avg_before > 0 else 0
        print(f"{song['song']:<35} {song['release_date'].date()} {growth:>18.1f}%")

print("=" * 70)