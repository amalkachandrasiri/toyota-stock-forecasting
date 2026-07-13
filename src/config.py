import logging
from pathlib import Path
from typing import Final

# =========================
# Configuration
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / 'data' / 'raw_data' / 'toyota_dataset.csv'

DATA_SMALL_PATH: Final[Path] = BASE_DIR / 'data' / 'processed_data' / 'dataset_small.csv'
DATA_LARGE_PATH: Final[Path] = BASE_DIR / 'data' / 'processed_data' / 'dataset_large.csv'

# report paths 
CLOSE_PRICING_TRENDS : Final[Path] = BASE_DIR / 'reports' / 'close_pricing_trends.png'
TRADING_VOLUME       : Final[Path] = BASE_DIR / 'reports' / 'trading_volume.png'
DAILY_RETURNS        : Final[Path] = BASE_DIR / 'reports' / 'daily_returns.png'
DIST_DAILY_RETURNS   : Final[Path] = BASE_DIR / 'reports' / 'dist_daily_returns.png'
OUTLIERS             : Final[Path] = BASE_DIR / 'reports' / 'outliers.png'
MOVING_AVG_30_DAYS   : Final[Path] = BASE_DIR / 'reports' / 'moving_avg_30_days.png'
TS_DECOMPOSITION     : Final[Path] = BASE_DIR / 'reports' / 'ts_decomposition.png'
