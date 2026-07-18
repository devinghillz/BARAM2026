# Lecture 05 Master Data Package Audit

## Shapes

| table | rows | columns |
|---|---:|---:|
| `ldaps_train_raw_wide` | 26,304 | 482 |
| `ldaps_test_raw_wide` | 8,760 | 482 |
| `gfs_train_raw_wide` | 26,304 | 317 |
| `gfs_test_raw_wide` | 8,760 | 317 |
| `weather_train_raw_wide` | 26,304 | 797 |
| `weather_test_raw_wide` | 8,760 | 797 |
| `master_train_with_labels` | 26,304 | 800 |
| `label_availability` | 26,304 | 4 |

## Label Missingness

| target | missing | valid |
|---|---:|---:|
| `kpx_group_1` | 104 | 26,200 |
| `kpx_group_2` | 103 | 26,201 |
| `kpx_group_3` | 8,766 | 17,538 |

- All targets available rows: 17,533
- Raw weather feature columns: 795
- Weather train missing cells: 0
- Weather test missing cells: 752

## Saved Files

- `train_forecast_index`: `lectures/lecture05/lecture05_master_data_package/metadata/train_forecast_index.csv`
- `test_forecast_index`: `lectures/lecture05/lecture05_master_data_package/metadata/test_forecast_index.csv`
- `ldaps_grid_metadata`: `lectures/lecture05/lecture05_master_data_package/metadata/ldaps_grid_metadata.csv`
- `gfs_grid_metadata`: `lectures/lecture05/lecture05_master_data_package/metadata/gfs_grid_metadata.csv`
- `turbine_metadata`: `lectures/lecture05/lecture05_master_data_package/metadata/turbine_metadata.csv`
- `weather_test_raw_wide`: `lectures/lecture05/lecture05_master_data_package/master/weather_test_raw_wide.csv`
- `label_availability`: `lectures/lecture05/lecture05_master_data_package/master/label_availability.csv`
- `master_train_with_labels`: `lectures/lecture05/lecture05_master_data_package/master/master_train_with_labels.csv`
