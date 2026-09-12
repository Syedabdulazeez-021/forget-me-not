| Method | Seed | Avg accuracy | Avg forgetting | BWT | Wall (s) |
| --- | --- | --- | --- | --- | --- |
| Naive fine-tuning (lower bound) | 42 | 0.1805 | 0.9557 | -0.9557 | 91 |
| LwF | 42 | 0.1801 | 0.9349 | -0.9349 | 93 |
| EWC | 42 | 0.1604 | 0.8837 | -0.8837 | 154 |
| Experience replay (buffer=200) | 42 | 0.2273 | 0.8805 | -0.8805 | 83 |
| Drift-freeze (ours) [bottom, k=1] | 42 | 0.1802 | 0.9549 | -0.9549 | 79 |
| Drift-freeze (ours) [bottom, k=2] | 42 | 0.1798 | 0.9470 | -0.9470 | 77 |
| Drift-freeze (ours) [bottom, k=3] | 42 | 0.1736 | 0.9284 | -0.9284 | 66 |
| Drift-freeze (ours) [paramrandom, k=1] | 42 | 0.1773 | 0.9349 | -0.9349 | 98 |
| Drift-freeze (ours) [paramrandom, k=2] | 42 | 0.1672 | 0.9069 | -0.9069 | 93 |
| Drift-freeze (ours) [paramrandom, k=3] | 42 | 0.1556 | 0.8626 | -0.8626 | 69 |
| Drift-freeze (ours) [random, k=1] | 42 | 0.1801 | 0.9526 | -0.9526 | 82 |
| Drift-freeze (ours) [random, k=2] | 42 | 0.1782 | 0.9456 | -0.9456 | 80 |
| Drift-freeze (ours) [random, k=3] | 42 | 0.1651 | 0.9287 | -0.9287 | 71 |
| Drift-freeze (ours) [top, k=0] | 42 | 0.1805 | 0.9557 | -0.9557 | 81 |
| Drift-freeze (ours) [top, k=1] | 42 | 0.1726 | 0.9484 | -0.9484 | 84 |
| Drift-freeze (ours) [top, k=2] | 42 | 0.1695 | 0.9427 | -0.9427 | 76 |
| Drift-freeze (ours) [top, k=3] | 42 | 0.1679 | 0.9255 | -0.9255 | 75 |
| Drift-freeze (ours) [explicit, k=2] | 42 | 0.1801 | 0.9526 | -0.9526 | 82 |
| Drift-freeze (ours) [explicit, k=2] | 42 | 0.1800 | 0.9379 | -0.9379 | 85 |
| Drift-freeze (ours) [explicit, k=2] | 42 | 0.1802 | 0.9513 | -0.9513 | 81 |
| Drift-freeze (ours) [explicit, k=2] | 42 | 0.1726 | 0.9484 | -0.9484 | 86 |
| Drift-freeze (ours) [explicit, k=2] | 42 | 0.1802 | 0.9549 | -0.9549 | 79 |
| Cumulative joint (upper bound) | 42 | 0.8161 | 0.0226 | -0.0149 | 244 |
