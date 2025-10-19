#%%
from numba import int64, float64
from numba.experimental import jitclass
# 定义类型规格
spec = [
    ('n', int64),
    ('_sum', float64),
    ('_comp', float64)
]
@jitclass(spec)
class StreamingMean:
    """高精度流式平均值（Neumaier补偿求和，避免浮点误差累积）"""
    def __init__(self):
        self.n = 0
        self._sum = 0.0
        self._comp = 0.0  # 误差补偿

    def push(self, x: float):
        self.n += 1
        t = self._sum + x
        # Neumaier compensation
        if abs(self._sum) >= abs(x):
            self._comp += (self._sum - t) + x
        else:
            self._comp += (x - t) + self._sum
        self._sum = t

    @property
    def mean(self) -> float:
        return 0.0 if self.n == 0 else (self._sum + self._comp) / self.n
#%%
from hftbacktest import BUY, SELL, GTX, LIMIT, BUY_EVENT
from numba.typed import Dict
from numba import uint64,njit
import numpy as np

@njit
def measure_trading_intensity(order_arrival_depth, out):
    max_tick = 0
    for depth in order_arrival_depth:
        if not np.isfinite(depth):
            continue

        # Sets the tick index to 0 for the nearest possible best price
        # as the order arrival depth in ticks is measured from the mid-price
        tick = round(depth / .5) - 1

        # In a fast-moving market, buy trades can occur below the mid-price (and vice versa for sell trades)
        # since the mid-price is measured in a previous time-step;
        # however, to simplify the problem, we will exclude those cases.
        if tick < 0 or tick >= len(out):
            continue

        # All of our possible quotes within the order arrival depth,
        # excluding those at the same price, are considered executed.
        out[:tick] += 1

        max_tick = max(max_tick, tick)
    return out[:max_tick]
@njit
def measure_trading_intensity_and_volatility(hbt):
    tick_size = hbt.depth(0).tick_size
    arrival_depth = np.full(10_000_000, np.nan, np.float64)
    mid_price_chg = np.full(10_000_000, np.nan, np.float64)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    # Checks every 100 milliseconds.
    while hbt.elapse(100_000_000) == 0:
        #--------------------------------------------------------
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(0):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(0)

        depth = hbt.depth(0)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        t += 1
        if t >= len(arrival_depth) or t >= len(mid_price_chg):
            raise Exception
    return arrival_depth[:t], mid_price_chg[:t]

@njit
def linear_regression(x, y):
    sx = np.sum(x)
    sy = np.sum(y)
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)
    w = len(x)
    slope = (w * sxy - sx * sy) / (w * sx2 - sx**2)
    intercept = (sy - slope * sx) / w
    return slope, intercept

@njit
def compute_coeff(xi, gamma, delta, A, k):
    inv_k = np.divide(1, k)
    c1 = 1 / (xi * delta) * np.log(1 + xi * delta * inv_k)
    c2 = np.sqrt(np.divide(gamma, 2 * A * delta * k) * ((1 + xi * delta * inv_k) ** (k / (xi * delta) + 1)))
    return c1, c2

@njit
def gridtrading_glft_mm(hbt, recorder, gamma = 0.05, delta = 1 ,adj1 = 1 ,adj2 = 0.05):
    asset_no = 0
    tick_size = hbt.depth(asset_no).tick_size

    arrival_depth = np.full(10_000_000, np.nan, np.float64)
    mid_price_chg = np.full(10_000_000, np.nan, np.float64)

    t = 0
    prev_mid_price_tick = np.nan
    mid_price_tick = np.nan

    tmp = np.zeros(500, np.float64)
    ticks = np.arange(len(tmp)) + 0.5

    A = np.nan
    k = np.nan
    volatility = np.nan
    # gamma = 0.05
    # delta = 1
    # adj1 = 1
    # adj2 = 0.05

    order_qty = 1
    max_position = 200
    grid_num = 20

    avg = StreamingMean()

    # Checks every 100 milliseconds.
    while hbt.elapse(1_000_000_000) == 0:
        #--------------------------------------------------------
        # Records market order's arrival depth from the mid-price.
        if not np.isnan(mid_price_tick):
            depth = -np.inf
            for last_trade in hbt.last_trades(asset_no):
                trade_price_tick = last_trade.px / tick_size

                if last_trade.ev & BUY_EVENT == BUY_EVENT:
                    depth = np.nanmax([trade_price_tick - mid_price_tick, depth])
                else:
                    depth = np.nanmax([mid_price_tick - trade_price_tick, depth])
            arrival_depth[t] = depth

        hbt.clear_last_trades(asset_no)
        hbt.clear_inactive_orders(asset_no)

        depth = hbt.depth(asset_no)
        position = hbt.position(asset_no)
        avg.push(position)

        orders = hbt.orders(asset_no)

        best_bid_tick = depth.best_bid_tick
        best_ask_tick = depth.best_ask_tick

        prev_mid_price_tick = mid_price_tick
        mid_price_tick = (best_bid_tick + best_ask_tick) / 2.0

        # Records the mid-price change for volatility calculation.
        mid_price_chg[t] = mid_price_tick - prev_mid_price_tick

        #--------------------------------------------------------
        # Calibrates A, k and calculates the market volatility.

        # Updates A, k, and the volatility every 5-sec.
        if t % 10 == 0:
            # Window size is 10-minute.
            if t >= 600 - 1:
                # Calibrates A, k
                tmp[:] = 0
                lambda_ = measure_trading_intensity(arrival_depth[t + 1 - 6_000:t + 1], tmp)
                if len(lambda_) > 2:
                    lambda_ = lambda_[:25] / 600
                    x = ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    A = np.exp(logA)
                    k = -k_

                # Updates the volatility.
                volatility = np.nanstd(mid_price_chg[t + 1 - 6_000:t + 1]) * np.sqrt(10)

        #--------------------------------------------------------
        # Computes bid price and ask price.

        c1, c2 = compute_coeff(gamma, gamma, delta, A, k)

        half_spread_tick = (c1 + delta / 2 * c2 * volatility) * adj1
        skew = c2 * volatility * adj2

        reservation_price_tick = mid_price_tick - skew * position

        bid_price_tick = np.minimum(np.round(reservation_price_tick - half_spread_tick), best_bid_tick)
        ask_price_tick = np.maximum(np.round(reservation_price_tick + half_spread_tick), best_ask_tick)

        bid_price = bid_price_tick * tick_size
        ask_price = ask_price_tick * tick_size

        grid_interval = max(np.round(half_spread_tick) * tick_size, tick_size)

        bid_price = np.floor(bid_price / grid_interval) * grid_interval
        ask_price = np.ceil(ask_price / grid_interval) * grid_interval

        #--------------------------------------------------------
        # Updates quotes.

        # Creates a new grid for buy orders.
        new_bid_orders = Dict.empty(np.uint64, np.float64)
        if position < max_position and np.isfinite(bid_price):
            for i in range(grid_num):
                bid_price_tick = round(bid_price / tick_size)

                # order price in tick is used as order id.
                new_bid_orders[uint64(bid_price_tick)] = bid_price

                bid_price -= grid_interval

        # Creates a new grid for sell orders.
        new_ask_orders = Dict.empty(np.uint64, np.float64)
        if position > -max_position and np.isfinite(ask_price):
            for i in range(grid_num):
                ask_price_tick = round(ask_price / tick_size)

                # order price in tick is used as order id.
                new_ask_orders[uint64(ask_price_tick)] = ask_price

                ask_price += grid_interval

        order_values = orders.values();
        while order_values.has_next():
            order = order_values.get()
            # Cancels if a working order is not in the new grid.
            if order.cancellable:
                if (
                    (order.side == BUY and order.order_id not in new_bid_orders)
                    or (order.side == SELL and order.order_id not in new_ask_orders)
                ):
                    hbt.cancel(asset_no, order.order_id, False)

        for order_id, order_price in new_bid_orders.items():
            # Posts a new buy order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_buy_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        for order_id, order_price in new_ask_orders.items():
            # Posts a new sell order if there is no working order at the price on the new grid.
            if order_id not in orders:
                hbt.submit_sell_order(asset_no, order_id, order_price, order_qty, GTX, LIMIT, False)

        #--------------------------------------------------------
        # Records variables and stats for analysis.

        t += 1

        if t >= len(arrival_depth) or t >= len(mid_price_chg):
            raise Exception

        # Records the current state for stat calculation.
        recorder.record(hbt)


    return avg.mean
#%%
from hftbacktest.stats import LinearAssetRecord
from hftbacktest import Recorder, BacktestAsset, ROIVectorMarketDepthBacktest

asset = (
    BacktestAsset()
        .data([
            'data/binance_spot/solusdt_20251011.npz',
            'data/binance_spot/solusdt_20251012.npz',
            'data/binance_spot/solusdt_20251013.npz',
            'data/binance_spot/solusdt_20251014.npz',
            'data/binance_spot/solusdt_20251015.npz',
        ])
        .initial_snapshot('data/binance_spot/solusdt_20251010_eod.npz')
        .linear_asset(1.0)
        .intp_order_latency([
            'data/binance_spot/solusdt_20251011_latency.npz',
            'data/binance_spot/solusdt_20251012_latency.npz',
            'data/binance_spot/solusdt_20251013_latency.npz',
            'data/binance_spot/solusdt_20251014_latency.npz',
            'data/binance_spot/solusdt_20251015_latency.npz',
        ])
        .power_prob_queue_model(2.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.5/1e4, 3/1e4)
        .tick_size(0.01)
        .lot_size(0.001)
        .roi_lb(0.0)
        .roi_ub(3000.0)
        .last_trades_capacity(10000)
)


#%%
import optuna
def objective(trail:optuna.Trial):
    gamma = trail.suggest_float('gamma', 0.01, 1.0, step=0.01)
    delta = trail.suggest_float('delta', 0.1, 10, step=0.1)
    adj1 = trail.suggest_float('adj1', 0.1, 10, step=0.1)
    adj2 = trail.suggest_float('adj2', 0.01, 1.0, step=0.01)

    hbt = ROIVectorMarketDepthBacktest([asset])
    recorder = Recorder(1, 5_000_000)
    avg = gridtrading_glft_mm(hbt, recorder.recorder, gamma, delta, adj1, adj2)

    hbt.close()
    stats = LinearAssetRecord(recorder.get(0)).stats(book_size=30_000)
    return stats.splits[0]['Return'] * 1e4, stats.splits[0]['DailyNumberOfTrades'] if  stats.splits[0]['DailyNumberOfTrades'] > abs(avg) else -stats.splits[0]['DailyNumberOfTrades'] , -abs(avg), -stats.splits[0]['MaxDrawdown'] * 1e4
#%%
study = optuna.create_study(study_name="mm-grid-glft-solusdt-No@1",
                            directions=["maximize", "maximize", "maximize", "maximize"],
                            sampler=optuna.samplers.TPESampler(seed=42),
                            storage="mysql://optuna:AyHfbtAyAiRjR4ck@47.86.7.11/optuna",
                            load_if_exists=True)
study.optimize(objective, n_trials=int(3 * 1e3), n_jobs=1)