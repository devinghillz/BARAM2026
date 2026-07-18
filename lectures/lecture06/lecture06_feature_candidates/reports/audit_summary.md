# Lecture 06 Feature Candidate Audit

| block | train shape | test shape | test missing rows | test missing cells |
|---|---:|---:|---:|---:|
| `time_features` | [26304, 13] | [8760, 13] | 0 | 0 |
| `wind_grid_features` | [26304, 573] | [8760, 573] | 3 | 368 |
| `grid_statistics` | [26304, 372] | [8760, 372] | 3 | 240 |
| `physical_grid_features` | [26304, 404] | [8760, 404] | 3 | 336 |
| `center_nearest` | [78912, 119] | [26280, 119] | 9 | 210 |
| `model_difference_center_nearest` | [78912, 11] | [26280, 11] | 9 | 27 |
| `turbine_nearest` | [78912, 119] | [26280, 119] | 9 | 210 |
| `model_difference_turbine_nearest` | [78912, 11] | [26280, 11] | 9 | 27 |
| `idw_p1` | [78912, 119] | [26280, 119] | 9 | 210 |
| `model_difference_idw_p1` | [78912, 11] | [26280, 11] | 9 | 27 |
| `idw_p2` | [78912, 119] | [26280, 119] | 9 | 210 |
| `model_difference_idw_p2` | [78912, 11] | [26280, 11] | 9 | 27 |

- Weight sum min: 1.000000000000
- Weight sum max: 1.000000000000
- Registry rows: 1,850
- Registry label_used any: False
- Registry fit_required any: False
