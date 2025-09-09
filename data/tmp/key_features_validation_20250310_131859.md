# Key Prediction Features Validation Report

Date: 2025-03-10 13:18:59

This report validates that the post-computation values from a fighter's second-to-last fight match the pre-computation values in their most recent fight.

## Summary

- Total fighters analyzed: 20
- Total value comparisons: 140
- Overall match rate: 92.14%

## Match Rate by Feature

| Feature | Match Rate |
|---------|------------|
| precomp_elo | 50.00% |
| precomp_sapm | 100.00% |
| precomp_sigstr_pm | 100.00% |
| precomp_strdef | 100.00% |
| precomp_subavg | 100.00% |
| precomp_tdavg | 95.00% |
| precomp_tddef | 100.00% |

## Mismatches (11 total)

### Cezar Ferreira

- Most recent fight: 2019-07-13 00:00:00
- Second most recent fight: 2017-11-11 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_tdavg | 3.0 | postcomp_tdavg | 2.0 |
| precomp_elo | 1520.4711837756424 | postcomp_elo | 1505.8526895931714 |

### Francisco Trinaldo

- Most recent fight: 2022-10-01 00:00:00
- Second most recent fight: 2022-05-07 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1630.1200634249117 | postcomp_elo | 1617.4503263057231 |

### Jake Ellenberger

- Most recent fight: 2017-04-22 00:00:00
- Second most recent fight: 2016-12-03 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1471.20192530494 | postcomp_elo | 1546.0201691664945 |

### Ovince Saint Preux

- Most recent fight: 2021-06-26 00:00:00
- Second most recent fight: 2020-05-13 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1634.9865511578976 | postcomp_elo | 1619.2208908859805 |

### Trevor Smith

- Most recent fight: 2018-12-15 00:00:00
- Second most recent fight: 2018-05-27 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1501.4823911085182 | postcomp_elo | 1542.082285522583 |

### Kyle Daukaus

- Most recent fight: 2022-06-18 00:00:00
- Second most recent fight: 2022-02-19 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1561.279567047211 | postcomp_elo | 1526.1637126081598 |

### Matt Brown

- Most recent fight: 2023-05-13 00:00:00
- Second most recent fight: 2022-03-26 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1549.52436586138 | postcomp_elo | 1524.8566276044132 |

### Viviane Araujo

- Most recent fight: 2024-11-16 00:00:00
- Second most recent fight: 2024-02-03 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1521.0662617624655 | postcomp_elo | 1540.0712589497057 |

### Manon Fiorot

- Most recent fight: 2022-10-22 00:00:00
- Second most recent fight: 2022-03-26 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1588.739319882335 | postcomp_elo | 1568.4174281899882 |

### Kennedy Nzechukwu

- Most recent fight: 2024-12-07 00:00:00
- Second most recent fight: 2024-10-26 00:00:00

| Pre Feature | Pre Value | Post Feature | Post Value |
|------------|-----------|-------------|------------|
| precomp_elo | 1605.9886096779476 | postcomp_elo | 1565.2429073575315 |


## Conclusion

Most pre_comp values from the most recent fight match the post_comp values from the second-to-last fight, but there are some discrepancies that should be investigated.

While we can generally use post_comp values for predictions, the specific features with lower match rates should be examined closely.