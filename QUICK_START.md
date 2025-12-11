# Quick Start Guide

Get started with the LEGO Assembly System in 5 minutes.

## 1. Install Dependencies

```bash
# Install uv (recommended - faster)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Python packages
uv sync

# OR use pip if you prefer
pip install -r requirements.txt

# Install Poppler (for PDF processing)
# macOS:
brew install poppler

# Ubuntu/Debian:
sudo apt-get install poppler-utils
```

## 2. Configure API Keys

```bash
# Copy template
cp ENV_TEMPLATE.txt .env

# Edit .env and add your API keys
nano .env
```

**Minimum required**: `DASHSCOPE_API_KEY` (for Qwen-VL)

Get your key at: https://dashscope.console.aliyun.com/

## 3. Run Your First Example

```bash
# Interactive mode - prompts for URL
python main.py

# Or provide URL directly
python main.py https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6521147.pdf

# Or use local PDF file
python main.py /path/to/your/lego_manual.pdf

# Or use test images directory
python main.py /path/to/instruction_images/ -o ./output
```

## 4. Check Output

Your output directory will contain:

- `{assembly_id}_plan.json` - Full structured plan
- `{assembly_id}_plan.txt` - Human-readable instructions
- `{assembly_id}_extracted.json` - Raw VLM extractions
- `{assembly_id}_dependencies.json` - Step dependency graph

## Example Output Structure

```json
{
  "assembly_id": "my_lego_set",
  "total_steps": 10,
  "steps": [
    {
      "step_number": 1,
      "parts": [
        {
          "part_num": "3001",
          "part_name": "Brick 2 x 4",
          "color": "red",
          "position": {"x": 0, "y": 0, "z": 0}
        }
      ]
    }
  ]
}
```

## Troubleshooting

### "No API key configured"
→ Add `DASHSCOPE_API_KEY` to your `.env` file

### "Poppler not found"
→ Install Poppler using package manager (see step 1)

### "Part not found in database"
→ This is normal for first runs. Parts will show as "unmatched" but system will continue

### API rate limits
→ Caching is enabled by default. Re-running same manual won't call API again

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Explore the [examples/](examples/) directory for sample manuals
- Check [src/](src/) for API reference and customization options

## Support

For issues or questions, review the main README.md troubleshooting section.

