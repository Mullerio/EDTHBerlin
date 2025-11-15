# Grid Integration Changes - Summary

## Overview
Successfully integrated frontend JSON grid data with the simulation framework. Key changes enable per-attacker targets and correct observable mask interpretation.

## Key Changes Made

### 1. Environment Class (`src/env.py`)

**Made target distribution optional:**
- `Environment.__init__()` now accepts `target=None`
- Added `attackers` parameter to add attackers at initialization
- Modified `_generate_prob_map()` to handle `None` target (returns uniform map)

**Made SectorEnv flexible:**
- `SectorEnv.__init__()` now accepts `target=None` and `attackers` list
- Allows creating environments without global target distribution

### 2. Grid Integration (`utils/grid_integration.py`)

**Fixed observable mask interpretation:**
- **CORRECTED**: Frontend JSON uses `0 = non-observable` (protected zone), `1 = observable` (open sky)
- Previous interpretation was inverted
- Updated `_apply_mask_to_env()` to correctly map: `matrix == 1` → `mask = True`

**Implemented per-attacker targets:**
- Each attacker gets its own `PointTarget` distribution
- No global target distribution needed
- Attackers are created with their individual targets and added to environment

**Fixed coordinate conversion:**
- Frontend: `[row, col]` where row=0 is TOP
- Simulation: `[x, y]` where y=0 is BOTTOM
- Conversion: `x = col`, `y = height - 1 - row`

### 3. Test Infrastructure

**Created test script** (`utils/test_integration_simple.py`):
- Loads 150x150 grid JSON
- Builds environment and attackers
- Verifies mask interpretation with sample points
- All tests passing ✓

**Updated test notebook** (`utils/test_grid_integration.ipynb`):
- Changed to load `grid-150x150(1).json`
- Updated documentation for correct mask interpretation
- Fixed colormap (now uses `RdYlGn` instead of `RdYlGn_r`)
- Uses environment's built-in visualization methods

## Test Results

**Grid Data:**
- Matrix shape: 150 × 150 cells
- Cell size: 100m
- Physical size: 15000m × 15000m
- Non-observable cells: 2,401 (protected zone)
- Observable cells: 20,099 (open sky)

**Environment:**
- Successfully created with no global target
- Observable mask correctly applied
- Mask verification: All sample points match ✓

**Attackers:**
- 4 attackers loaded successfully
- Each has individual start and target positions
- Speed-based trajectories (50 m/s default)
- Flight distances: 10.9 - 15.3 km
- Flight times: 3.7 - 5.1 minutes

## Usage Example

```python
from utils.grid_integration import load_and_build_env

# Load JSON and build environment
env, attackers, grid_data = load_and_build_env("grid-150x150(1).json")

# Attackers are already added to environment
print(f"Environment has {len(env.atk_drones)} attackers")

# Visualize
env.visualize_trajectories(show_sectors=True)

# Analyze trajectories
for attacker in attackers:
    analysis = env.analyze_trajectory(attacker.trajectory)
    print(f"Detection probability: {analysis['cumulative_detection_prob']:.2%}")
```

## JSON Format Requirements

```json
{
  "matrix": [[0, 1, 1, ...], ...],  // 0 = non-observable, 1 = observable
  "cell_size_m": 100.0,
  "centroid": {"row": 75, "col": 75, "meters": [7500, 7500]},
  "attackers": [
    {
      "name": "Attacker 1",
      "start": {
        "grid": {"row": 28, "col": 14},
        "meters": {"x": 1400, "y": 2800}
      },
      "target": {
        "grid": {"row": 126, "col": 132},
        "meters": {"x": 13200, "y": 12600}
      }
    }
  ]
}
```

## Benefits

1. **Per-Attacker Targets**: Each drone can have a different destination
2. **Correct Mask**: Non-observable zones properly protect specific areas
3. **Flexible Environment**: Can create environments with or without target distributions
4. **Clean Integration**: Frontend JSON directly maps to simulation objects
5. **Automatic Coordinate Conversion**: Row/col → x/y handled internally

## Next Steps

1. ✅ Test with real 150x150 grid - **COMPLETE**
2. ✅ Verify mask interpretation - **COMPLETE**
3. ✅ Test attacker trajectories - **COMPLETE**
4. 🔄 Add detectors to non-observable regions
5. 🔄 Run full simulations with detection analysis
6. 🔄 Test with Jupyter notebook visualization

## Files Modified

- `src/env.py` - Made target optional, added attackers init parameter
- `utils/grid_integration.py` - Fixed mask interpretation, per-attacker targets
- `utils/test_grid_integration.ipynb` - Updated for 150x150 grid
- `utils/test_integration_simple.py` - Created verification script (NEW)
