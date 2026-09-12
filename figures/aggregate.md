| Method | Seeds | Avg accuracy | Avg forgetting | BWT |
| --- | --- | --- | --- | --- |
| Naive fine-tuning (lower bound) | 1 | 0.1805 | 0.9557 | -0.9557 |
| LwF | 1 | 0.1801 | 0.9349 | -0.9349 |
| EWC (lambda=5000.0) | 1 | 0.1604 | 0.8837 | -0.8837 |
| Experience replay (buffer=200) | 1 | 0.2273 | 0.8805 | -0.8805 |
| Drift-freeze (ours) (select=bottom) (k=1) | 1 | 0.1802 | 0.9549 | -0.9549 |
| Drift-freeze (ours) (select=bottom) (k=2) | 1 | 0.1798 | 0.9470 | -0.9470 |
| Drift-freeze (ours) (select=bottom) (k=3) | 1 | 0.1736 | 0.9284 | -0.9284 |
| Drift-freeze (ours) (select=paramrandom) (k=1) | 1 | 0.1773 | 0.9349 | -0.9349 |
| Drift-freeze (ours) (select=paramrandom) (k=2) | 1 | 0.1672 | 0.9069 | -0.9069 |
| Drift-freeze (ours) (select=paramrandom) (k=3) | 1 | 0.1556 | 0.8626 | -0.8626 |
| Drift-freeze (ours) (select=random) (k=1) | 1 | 0.1801 | 0.9526 | -0.9526 |
| Drift-freeze (ours) (select=random) (k=2) | 1 | 0.1782 | 0.9456 | -0.9456 |
| Drift-freeze (ours) (select=random) (k=3) | 1 | 0.1651 | 0.9287 | -0.9287 |
| Drift-freeze (ours) (select=top) (k=0) | 1 | 0.1805 | 0.9557 | -0.9557 |
| Drift-freeze (ours) (select=top) (k=1) | 1 | 0.1726 | 0.9484 | -0.9484 |
| Drift-freeze (ours) (select=top) (k=2) | 1 | 0.1695 | 0.9427 | -0.9427 |
| Drift-freeze (ours) (select=top) (k=3) | 1 | 0.1679 | 0.9255 | -0.9255 |
| Drift-freeze (ours) (select=explicit) (k=2) | 5 | 0.1786 ± 0.0030 | 0.9490 ± 0.0059 | -0.9490 ± 0.0059 |
| Cumulative joint (upper bound) | 1 | 0.8161 | 0.0226 | -0.0149 |

**Mandatory lower bound (naive): 0.1805 average accuracy**
**Mandatory upper bound (joint): 0.8161**

| Method | Avg accuracy | Delta vs baseline | Beats baseline? |
| --- | --- | --- | --- |
| Drift-freeze (ours) (select=bottom) (k=1) | 0.1802 | -0.0003 | NO |
| Drift-freeze (ours) (select=bottom) (k=2) | 0.1798 | -0.0007 | NO |
| Drift-freeze (ours) (select=bottom) (k=3) | 0.1736 | -0.0069 | NO |
| Drift-freeze (ours) (select=paramrandom) (k=1) | 0.1773 | -0.0032 | NO |
| Drift-freeze (ours) (select=paramrandom) (k=2) | 0.1672 | -0.0133 | NO |
| Drift-freeze (ours) (select=paramrandom) (k=3) | 0.1556 | -0.0249 | NO |
| Drift-freeze (ours) (select=random) (k=1) | 0.1801 | -0.0004 | NO |
| Drift-freeze (ours) (select=random) (k=2) | 0.1782 | -0.0023 | NO |
| Drift-freeze (ours) (select=random) (k=3) | 0.1651 | -0.0154 | NO |
| Drift-freeze (ours) (select=top) (k=0) | 0.1805 | +0.0000 | NO |
| Drift-freeze (ours) (select=top) (k=1) | 0.1726 | -0.0079 | NO |
| Drift-freeze (ours) (select=top) (k=2) | 0.1695 | -0.0110 | NO |
| Drift-freeze (ours) (select=top) (k=3) | 0.1679 | -0.0126 | NO |
| EWC (lambda=5000.0) | 0.1604 | -0.0201 | NO |
| LwF | 0.1801 | -0.0004 | NO |
| Experience replay (buffer=200) | 0.2273 | +0.0468 | YES |
| Drift-freeze (ours) (select=explicit) (k=2) | 0.1786 | -0.0019 | NO |

