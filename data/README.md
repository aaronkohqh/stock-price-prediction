# data/

`raw/` holds cached price data pulled by `src/data.py` from Yahoo Finance.
It is gitignored on purpose: the data is fully reproducible from code, so the
repo commits the fetcher, not the CSVs.

To populate locally:

```python
from src.data import fetch_prices, to_log_returns
prices = fetch_prices("NVDA")
returns = to_log_returns(prices)
```
