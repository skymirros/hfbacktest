import numpy as np

from hftbacktest.data.utils import binancefutures

data = binancefutures.convert(
    'data\\binance_future\ethusdt_20251011.gz',
    output_filename='data\\binance_future\ethusdt_20251011.npz',
    combined_stream=True
)