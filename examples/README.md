# Examples

This directory is for storing example LEGO instruction manuals for testing.

## Suggested Test Materials

1. **Simple Sets**: 
   - Small car or house (10-20 steps)
   - Good for initial testing

2. **Medium Sets**:
   - 30-50 steps
   - Tests dependency graph construction

3. **Complex Sets**:
   - 100+ steps with subassemblies
   - Tests full system capabilities

## Running Examples

```bash
# Process a manual in this directory
python main.py examples/simple_car.pdf -o output/simple_car

# With fallback VLMs
python main.py examples/medium_house.pdf --use-fallback -o output/medium_house
```

## Manual Sources

You can obtain LEGO instruction manuals from:
- Official LEGO website (https://www.lego.com/service/buildinginstructions)
- Physical LEGO sets you own
- Rebrickable (https://rebrickable.com/sets/)

## Format Requirements

- **PDF**: Multi-page instruction manuals
- **Images**: PNG/JPG sequences in a directory
- **Naming**: Sequential naming recommended (step_001.png, step_002.png, ...)

## Note

This directory is ignored by git. Add your own test materials here for local testing.

