# Elo System Comparison — April 3, 2026

## Evaluation Setup
- Fights: 3,287 (2014-2024, both fighters >= 2 prior UFC fights)
- Weighted by recency (λ=0.13, recent fights count more)

## Individual Feature Tests

| System | W.Corr | W.Acc | W.LL | W.AUC |
|---|---|---|---|---|
| Baseline (opp_quality_k only) | 0.1461 | 55.6% | 0.6830 | 0.5805 |
| + sliding_k | **0.1642** | **56.2%** | **0.6814** | **0.5891** |
| + upset momentum | 0.1464 | 55.6% | 0.6829 | 0.5807 |
| + champ_mult | 0.1461 | 55.6% | 0.6830 | 0.5805 |
| + margin of victory | 0.1284 | 55.9% | 0.6887 | 0.5719 |
| + streak regression | 0.1332 | 55.6% | 0.6849 | 0.5755 |
| + loss asymmetry | 0.0852 | 53.5% | 0.6963 | 0.5464 |
| + debut seeding | 0.0430 | 52.2% | 0.7517 | 0.5244 |

## Best Combination

| System | W.Corr | W.Acc | W.LL | W.AUC |
|---|---|---|---|---|
| Baseline | 0.1461 | 55.6% | 0.6830 | 0.5805 |
| **opp_quality + sliding_k + upset + champ** | **0.1647** | **56.2%** | **0.6812** | **0.5895** |

## Best Active Rankings (opp_quality + sliding_k + upset + champ)

1. Islam Makhachev — 2072
2. Alex Pereira — 2064
3. Ilia Topuria — 2043
4. Tom Aspinall — 1996
5. Khamzat Chimaev — 1964
6. Max Holloway — 1951
7. Jon Jones — 1945
8. Magomed Ankalaev — 1935
9. Michael Morales — 1927
10. Ciryl Gane — 1916

## Feature Descriptions

- **opp_quality_k**: K *= clamp(opp_elo / 1500, 0.6, 1.5) — beating elite opponents moves Elo more
- **sliding_k**: K starts 1.5x for first 5 fights, decreases to 0.8x after 15 fights — new fighters converge faster
- **upset momentum**: when winner's expected prob was <30%, K *= 1 + (0.30 - expected) — big upsets move Elo more
- **champ_mult**: 5-round fights get 1.2x K — title fights carry more weight
