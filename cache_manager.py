#!/usr/bin/env python3
"""
Utility script to manage the VLM response cache.

Usage:
    python cache_manager.py stats    # Show cache statistics
    python cache_manager.py clear    # Clear all cache entries
    python cache_manager.py list     # List all cache entries (summary)
"""

import sys
from src.utils.cache import get_cache

def show_stats():
    """Display cache statistics."""
    cache = get_cache()
    stats = cache.stats()
    
    print("=" * 60)
    print("CACHE STATISTICS")
    print("=" * 60)
    print(f"Enabled: {cache.enabled}")
    print(f"Location: {cache.cache_dir}")
    print(f"Entries: {stats['size']}")
    print(f"Volume: {stats['volume']:,} bytes ({stats['volume'] / 1024:.2f} KB)")
    print("=" * 60)

def clear_cache():
    """Clear all cache entries."""
    cache = get_cache()
    
    before_stats = cache.stats()
    print(f"Cache contains {before_stats['size']} entries ({before_stats['volume']:,} bytes)")
    
    response = input("Are you sure you want to clear the cache? (yes/no): ")
    if response.lower() != "yes":
        print("Cancelled.")
        return
    
    cache.clear()
    
    after_stats = cache.stats()
    print(f"\n✓ Cache cleared!")
    print(f"Removed {before_stats['size']} entries")
    print(f"Freed {before_stats['volume']:,} bytes")
    print(f"Current entries: {after_stats['size']}")

def list_cache():
    """List cache entries (summary)."""
    cache = get_cache()
    
    print("=" * 60)
    print("CACHE ENTRIES")
    print("=" * 60)
    
    count = 0
    for key in cache.cache.iterkeys():
        value = cache.cache.get(key)
        value_type = type(value).__name__
        value_size = len(str(value)) if value else 0
        
        print(f"{count + 1}. Key: {key[:32]}... | Type: {value_type} | Size: {value_size} bytes")
        count += 1
        
        if count >= 20:
            remaining = len(cache.cache) - count
            if remaining > 0:
                print(f"... and {remaining} more entries")
            break
    
    print("=" * 60)
    print(f"Total: {len(cache.cache)} entries")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        show_stats()
    elif command == "clear":
        clear_cache()
    elif command == "list":
        list_cache()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()

