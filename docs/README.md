# Documentation Generation

## Regenerate Docs

```bash
# 1. Generate RST files from GDScript sources
python docs/generate_api_rst.py addons/godot-stat-math/core docs/modules

# 2. Build HTML documentation  
./make.bat html
```

## View Results

Open `docs/_build/html/index.html` in browser.

## Requirements

- Python 3.x
- Sphinx (`pip install sphinx`) 