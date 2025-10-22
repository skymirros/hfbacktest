# %%
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

# %%
from hftbacktest import BUY, SELL, GTX, LIMIT, BUY_EVENT
from numba.typed import Dict
from numba import uint64,njit
import numpy as np
from datetime import datetime, timezone


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
def gridtrading_glft_mm(hbt, recorder, gamma = 0.05, delta = 1.0 ,adj1 = 1.0 ,adj2 = 0.05, fit_window = 600, bins_for_fit = 70):
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


    order_qty = 0.5
    max_position = 200
    grid_num = 20

    avg = StreamingMean()

    # Checks every 1 second.
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
        if t % 5 == 0:
            # Window size is 10-minute.
            if t >= fit_window - 1:
                # Calibrates A, k
                tmp[:] = 0
                lambda_ = measure_trading_intensity(arrival_depth[t + 1 - fit_window:t + 1], tmp)
                if len(lambda_) > 2:
                    lambda_ = lambda_[:bins_for_fit] / fit_window
                    x = ticks[:len(lambda_)]
                    y = np.log(lambda_)
                    k_, logA = linear_regression(x, y)
                    A = np.exp(logA)
                    k = -k_

                # Updates the volatility.
                volatility = np.nanstd(mid_price_chg[t + 1 - fit_window:t + 1]) * np.sqrt(10)
        
        if abs(k) < 1e-9 or abs(A) < 1e-9:
            t += 1
            return
        

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

        # # Creates a new grid for buy orders.
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

# %%
from hftbacktest.stats import LinearAssetRecord
from hftbacktest import Recorder, BacktestAsset, ROIVectorMarketDepthBacktest

data = np.concatenate(
[np.load('C:/Users/81393/Desktop/code/hfbacktest/data/bitget_spot/{}/SPOT.SOLUSDT.npz'.format(date))['data'] for date in [20251020, 20251021, 20251022]]
)
initial_snapshot = np.load('C:/Users/81393/Desktop/code/hfbacktest/data/bitget_spot/20251019/SPOT.SOLUSDT.eod.npz')['data']
latency_data = np.concatenate(
[np.load('C:/Users/81393/Desktop/code/hfbacktest/data/bitget_spot/{}/SPOT.SOLUSDT.latency.npz'.format(date))['data'] for date in [20251020, 20251021, 20251022]]
)

asset = (
    BacktestAsset()
        .data(data)
        .initial_snapshot(initial_snapshot)
        .linear_asset(1.0)
        .intp_order_latency(latency_data)
        .power_prob_queue_model(2.0)
        .no_partial_fill_exchange()
        .trading_value_fee_model(-0.5/1e4, 3/1e4)
        .tick_size(0.01)
        .lot_size(0.0001)
        .roi_lb(0.0)
        .roi_ub(500.0)
        .last_trades_capacity(10000)
)



# %%
import optuna
def objective(trail:optuna.Trial| None = None):
    if trail:
        gamma = trail.suggest_float('gamma', 0.01, 1.0, step=0.01)
        delta = trail.suggest_float('delta', 0.1, 10, step=0.1)
        adj1 = trail.suggest_float('adj1', 0.01, 10, step=0.01)
        adj2 = trail.suggest_float('adj2', 0.01, 1.0, step=0.01)
        fit_window = trail.suggest_int('update_A_K_window', 100, 3000, step=100)
        bins_for_fit = trail.suggest_int('bins_for_fit', 40, 120, step=10)
    else:
    # 1231 [0.17404209333333825, 13984.706214662881, 44265.853372434016, 1.717052732784075] {'gamma': 0.04, 'delta': 4.1, 'adj1': 0.1, 'adj2': 0.04, 'update_A_K_window': 1700, 'bins_for_fit': 60}
    # 26922 [0.14802678166655256, 7778.96604340188, 24660.457478005865, 3.227600182861747] {'gamma': 0.04, 'delta': 9.3, 'adj1': 0.1, 'adj2': 0.15000000000000002, 'update_A_K_window': 1300, 'bins_for_fit': 50}

        gamma = 0.04
        delta = 4.1
        adj1 = 0.1
        adj2 = 0.04
        fit_window = 1700
        bins_for_fit = 60
    hbt = ROIVectorMarketDepthBacktest([asset])
    recorder = Recorder(1, 5_000_000)
    avg = gridtrading_glft_mm(hbt, recorder.recorder, gamma, delta, adj1, adj2, fit_window, bins_for_fit) 

    hbt.close()
    stats = LinearAssetRecord(recorder.get(0)).stats(book_size=30_000)

    return stats.splits[0]['Return'] * 1e2, stats.splits[0]['DailyTurnover'] * 1e2  ,stats.splits[0]['DailyNumberOfTrades'] / abs(avg or 1), stats.splits[0]['ReturnOverMDD'] if not np.isnan(stats.splits[0]['ReturnOverMDD']) else 0
    # return stats

# %%
study = optuna.create_study(study_name="mm-grid-glft-solusdt-bitget@No.7",
                            directions=["maximize", "maximize", "maximize", "maximize"],
                            sampler=optuna.samplers.NSGAIIISampler(
    population_size=128,        # ≈ 参考点数
    dividing_parameter=8,      # H
    seed=42,                   # 复现用
),
                            storage="mysql://optuna:AyHfbtAyAiRjR4ck@47.86.7.11/optuna",
                            load_if_exists=True)
study.optimize(objective, n_trials=int(3 * 1e3), n_jobs=1)


