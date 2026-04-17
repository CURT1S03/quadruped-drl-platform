# Terrain Assets

Place custom terrain YAML configurations here.

See `example_mixed.yaml` for the format.

## Available Terrain Types

| Type | Description |
|------|-------------|
| `MeshPyramidStairsTerrainCfg` | Ascending pyramid stairs |
| `MeshInvertedPyramidStairsTerrainCfg` | Descending pyramid stairs |
| `MeshRandomGridTerrainCfg` | Random box obstacles / hurdles |
| `HfRandomUniformTerrainCfg` | Random rough ground |
| `HfPyramidSlopedTerrainCfg` | Ascending sloped terrain |
| `HfInvertedPyramidSlopedTerrainCfg` | Descending sloped terrain |

## Built-in Presets

These presets are available without a YAML file:

- `flat` — Flat plane, no obstacles
- `easy` — Gentle rough ground, mild slopes, low boxes
- `obstacle` — Mixed terrain (default): stairs, hurdles, rough ground, slopes
- `hard` — Tall stairs, high boxes, steep slopes
- `stairs` — Stairs only (ascending + descending)
- `slopes` — Slopes only (ascending + descending)
