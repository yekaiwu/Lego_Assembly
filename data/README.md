# Data Directory

This directory stores:

1. **parts_database.db**: SQLite database caching LEGO parts from Rebrickable
   - Automatically created on first run
   - Contains: parts, colors, part_categories, part_colors, part_dimensions

2. **Cache files**: May be stored here if configured

## Initial Setup

The database will be automatically initialized when you first run the system. Optionally, you can pre-populate the color database:

```python
from src.plan_generation import PartDatabase
db = PartDatabase()
db.fetch_colors_from_api()  # Requires REBRICKABLE_API_KEY in .env
```

## Database Schema

### parts
- part_num (PRIMARY KEY)
- name
- part_cat_id
- part_material
- year_from, year_to

### colors
- id (PRIMARY KEY)
- name
- rgb
- is_trans

### part_colors
- part_num, color_id (COMPOSITE KEY)
- num_sets

### part_dimensions
- part_num (PRIMARY KEY)
- width_studs, height_studs, depth_studs

