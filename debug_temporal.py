#!/usr/bin/env python3
"""
Debug script to understand the temporal ordering issue.
"""

import numpy as np
import polars as pl

def debug_temporal_logic():
    """Debug the temporal indexing logic."""
    
    # Simple case: Alice (09:00), Bob (11:00), Charlie (13:00)
    data = pl.DataFrame({
        "user": ["Alice", "Bob", "Charlie"],
        "item": ["X", "X", "X"],
        "timestamp": ["2024-01-01 09:00:00", "2024-01-01 11:00:00", "2024-01-01 13:00:00"]
    })
    
    print("Original data:")
    print(data)
    
    # Convert timestamp to datetime
    data = data.with_columns(
        pl.col("timestamp").str.to_datetime().alias("timestamp")
    )
    
    print("\nAfter datetime conversion:")
    print(data)
    
    # Sort by timestamp ascending (earliest first)
    data_asc = data.sort("timestamp", descending=False)
    print("\nAfter ascending sort:")
    print(data_asc)
    
    # Get the users in ascending timestamp order
    users = data_asc["user"].to_list()
    timestamps = data_asc["timestamp"].to_list()
    
    print(f"\nUsers in ascending order: {users}")
    print(f"Timestamps in ascending order: {timestamps}")
    
    # Create upper triangular indices
    n = len(users)
    upper_tri = np.triu_indices(n, k=1)
    print(f"\nUpper triangular indices: {upper_tri}")
    
    print(f"\nEdges created:")
    for i, j in zip(upper_tri[0], upper_tri[1]):
        source_user = users[i]
        target_user = users[j]
        source_time = timestamps[i]
        target_time = timestamps[j]
        print(f"  {source_user} ({source_time}) → {target_user} ({target_time})")
        
    print(f"\nTemporal check:")
    for i, j in zip(upper_tri[0], upper_tri[1]):
        source_time = timestamps[i]
        target_time = timestamps[j]
        is_earlier = source_time < target_time  # Earlier in time (smaller timestamp in asc order)
        print(f"  {users[i]} → {users[j]}: source time {source_time} > target time {target_time} = {is_earlier}")

if __name__ == "__main__":
    debug_temporal_logic()