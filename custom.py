import numpy as np

from numba import njit, uint64
from numba.typed import Dict
from hftbacktest.stats import LinearAssetRecord
from dataclasses import dataclass


from hftbacktest import (
    BacktestAsset,
    ROIVectorMarketDepthBacktest,
    GTX,
    LIMIT,
    BUY,
    SELL,
    BUY_EVENT,
    SELL_EVENT,
    Recorder
)
from hftbacktest.stats import LinearAssetRecord

import copy

@dataclass
class Config:
    # 资产与数据
    symbol: str = "SOLUSDT"
    engine: str = "AS"
    test_interval_ms: int = 1000  # 主循环节拍（≈ quoteRefreshMs）

    # 做市规模/订单管理
    baseOrderQty: float = 2
    maxOrdersPerSide: int = 2
    repriceThresholdBps: float = 1.5
    cancelThrottleMs: int = 400
    queueSizeAheadPctLimit: float = 0.35
    queueBumpTicks: int = 1

    # Alpha / 价心偏移
    obiDepth: int = 5
    obiWeight: float = 0.8
    bpLookback: int = 12
    bpWeight: float = 0.4
    micropriceWeight: float = 0.6
    alphaToPriceBps: float = 3.0

    # AS 工业化参数
    riskAversionGamma: float = 0.1
    baseHalfSpreadBps: float = 2.0
    volHalfSpreadK: float = 0.25
    inventorySkewK: float = 0.4


    # 波动率窗口（秒）
    volLookbackSec: int = 300

    # 风控-以 quote 计
    targetInventoryQuote: float = 0
    maxNetPosQuote: float = 5_00000.0

    # 其他
    makerFee: float = -0.00005   # 示例费率：maker 返佣
    takerFee: float = 0.0007     # 示例费率：taker 付费

CFG = Config()

@njit
def floor_to_tick(px: float, tick: float) -> float:
    return np.floor(px / tick) * tick

@njit
def ceil_to_tick(px: float, tick: float) -> float:
    return np.ceil(px / tick) * tick

@njit
def bps_distance(a: float, b: float) -> float:
    if b == 0.0:
        return 0.0
    return np.abs((a - b) / b) * 1e4

@njit
def microprice_from_top(bid_px: float, bid_qty: float, ask_px: float, ask_qty: float) -> float:
    denom = bid_qty + ask_qty
    if denom <= 0.0:
        return (bid_px + ask_px) * 0.5
    return (ask_px * bid_qty + bid_px * ask_qty) / denom

@njit
def minute_index_from_ns(ts_ns: np.int64) -> np.int64:
    return ts_ns // np.int64(60_000_000_000)  # 60s -> ns





from typing import Tuple


@njit
def calc_obi(depth, levels: int) -> float:
    # 使用 depth.bid_qty_at_tick / ask_qty_at_tick 逐档向外累加
    L = max(1, levels)
    bid_tick = depth.best_bid_tick
    ask_tick = depth.best_ask_tick
    b_sum = 0.0
    a_sum = 0.0
    for i in range(L):
        b_sum += depth.bid_qty_at_tick(bid_tick - i)
        a_sum += depth.ask_qty_at_tick(ask_tick + i)
    denom = b_sum + a_sum
    if denom <= 0.0:
        return 0.0
    return (b_sum - a_sum) / denom  # [-1, 1]

@njit
def barportion_avg(last_n: int, bar_o, bar_h, bar_l, bar_c, bar_cnt: int) -> float:
    if bar_cnt == 0:
        return 0.5
    N = last_n if last_n < bar_cnt else bar_cnt
    eps = 1e-12
    s = 0.0
    for i in range(N):
        idx = bar_cnt - 1 - i
        rng = bar_h[idx] - bar_l[idx]
        if rng < eps:
            s += 0.5
        else:
            s += (bar_c[idx] - bar_l[idx]) / (rng + eps)
    return s / N

@njit
def vol_scale_from_mids(mids, mids_ts, cnt: int, lookback_ns: np.int64) -> float:
    # 用时间窗裁剪，计算 mid 的相邻报酬波动率
    if cnt < 5:
        return 0.5
    cur_ts = mids_ts[cnt - 1]
    cutoff = cur_ts - lookback_ns
    # 找到起点
    start = 0
    for i in range(cnt - 1, -1, -1):
        if mids_ts[i] <= cutoff:
            start = i
            break
    if cnt - start < 5:
        start = max(0, cnt - 5)

    # 计算简单相邻收益
    prev = mids[start]
    # 收益个数 <= cnt - start - 1
    s = 0.0
    ss = 0.0
    n = 0
    for i in range(start + 1, cnt):
        r = (mids[i] - prev) / prev if prev > 0 else 0.0
        s += r
        ss += r * r
        n += 1
        prev = mids[i]
    if n <= 1:
        return 0.5
    mean = s / n
    var = (ss - n * mean * mean) / (n - 1)
    stdev = np.sqrt(max(var, 0.0))
    # 与 TS 版缩放一致，压到 [0.1, 3]
    return min(3.0, max(0.1, (stdev * 1e4) / 10.0))

# --------------------------- 价差/中心（AS / GLFT） ---------------------------

@njit
def compute_alpha(mid: float,
                  depth,
                  obiDepth: int,
                  obiW: float, bpW: float, microW: float, alphaToBps: float,
                  # bars:
                  bar_o, bar_h, bar_l, bar_c, bar_cnt: int,bpLookback:float) -> Tuple[float, float]:
    obi = calc_obi(depth, obiDepth)
    bp = barportion_avg(bpLookback, bar_o, bar_h, bar_l, bar_c, bar_cnt) - 0.5
    micro = microprice_from_top(depth.best_bid, depth.best_bid_qty,
                                depth.best_ask, depth.best_ask_qty)
    microShiftBps = ((micro - mid) / mid) * 1e4 if mid > 0 else 0.0
    centerShift = alphaToBps * (obiW * obi + bpW * bp + microW * (microShiftBps / 10.0))
    skew = obiW * obi + bpW * bp
    return centerShift, skew

@njit
def compute_quotes_AS(mid: float, depth, volScale: float,
                      centerShiftBps: float, skewAlpha: float,
                      invSkew: float,riskAversionGamma:float,baseHalfSpreadBps:float,volHalfSpreadK:float) -> Tuple[float, float, float]:
    # 中心
    center = mid * (1.0 + centerShiftBps / 1e4 + (invSkew * riskAversionGamma) / 10.0)
    halfBps = max(0.5, baseHalfSpreadBps + volHalfSpreadK * volScale)
    skewBps = 0.4 * halfBps * skewAlpha
    bid = center * (1.0 - (halfBps - skewBps) / 1e4)
    ask = center * (1.0 + (halfBps + skewBps) / 1e4)
    return bid, ask, halfBps

@njit
def compute_quotes_GLFT(mid: float, depth, volScale: float, invSkew: float,cfg:Config) -> Tuple[float, float, float]:
    mp = microprice_from_top(depth.best_bid, depth.best_bid_qty,
                             depth.best_ask, depth.best_ask_qty)
    fair = cfg.glft_kFair * mp + (1.0 - cfg.glft_kFair) * mid
    center = fair * (1.0 + (cfg.glft_riskK * invSkew) / 10.0)
    halfBps = max(0.5, cfg.glft_baseHalfSpreadBps + cfg.glft_volHalfSpreadK * volScale)
    bid = center * (1.0 - halfBps / 1e4)
    ask = center * (1.0 + halfBps / 1e4)
    return bid, ask, halfBps

def strategy(hbt,stat,cfg:Config):
    asset_no = 0
    tick = hbt.depth(asset_no).tick_size
    lot  = hbt.depth(asset_no).lot_size

    # 时间/节拍（ns）
    step_ns = np.int64(cfg.test_interval_ms) * 1_000_000
    cancel_throttle_ns = np.int64(cfg.cancelThrottleMs) * 1_000_000
    vol_lookback_ns = np.int64(cfg.volLookbackSec) * 1_000_000_000

    # 条形图缓存（最多 2k 根）
    BAR_MAX = 2000
    bar_o = np.zeros(BAR_MAX, np.float64)
    bar_h = np.zeros(BAR_MAX, np.float64)
    bar_l = np.zeros(BAR_MAX, np.float64)
    bar_c = np.zeros(BAR_MAX, np.float64)
    bar_cnt = 0
    last_minute = np.int64(-1)

    # mid 滚动（最多 8192 点）
    MID_MAX = 8192
    mids = np.zeros(MID_MAX, np.float64)
    mids_ts = np.zeros(MID_MAX, np.int64)
    mid_cnt = 0

    def push_mid(ts_ns: np.int64, m: float):
        nonlocal mid_cnt
        if mid_cnt < MID_MAX:
            mids[mid_cnt] = m
            mids_ts[mid_cnt] = ts_ns
            mid_cnt += 1
        else:
            # 简单滑窗：全部左移 1
            for i in range(1, MID_MAX):
                mids[i-1] = mids[i]
                mids_ts[i-1] = mids_ts[i]
            mids[MID_MAX-1] = m
            mids_ts[MID_MAX-1] = ts_ns

    def on_trade_price(ts_ns: np.int64, px: float):
        nonlocal bar_cnt, last_minute
        m_idx = minute_index_from_ns(ts_ns)
        if m_idx != last_minute:
            # 开新 bar
            if bar_cnt < BAR_MAX:
                bar_o[bar_cnt] = px
                bar_h[bar_cnt] = px
                bar_l[bar_cnt] = px
                bar_c[bar_cnt] = px
                bar_cnt += 1
            else:
                # 左移一格
                for i in range(1, BAR_MAX):
                    bar_o[i-1] = bar_o[i]
                    bar_h[i-1] = bar_h[i]
                    bar_l[i-1] = bar_l[i]
                    bar_c[i-1] = bar_c[i]
                bar_o[BAR_MAX-1] = px
                bar_h[BAR_MAX-1] = px
                bar_l[BAR_MAX-1] = px
                bar_c[BAR_MAX-1] = px
            last_minute = m_idx
        else:
            i = bar_cnt - 1
            if px > bar_h[i]: bar_h[i] = px
            if px < bar_l[i]: bar_l[i] = px
            bar_c[i] = px

    # 撤单节流/对账
    last_reconcile_ts = np.int64(0)
    last_cancel_ts = np.int64(-1)

    next_oid = np.int64(1)

    # 事件主循环：每 step_ns 检查一次
    while hbt.elapse(step_ns) == 0:
        hbt.clear_inactive_orders(asset_no)
        ts = hbt.current_timestamp

        depth = hbt.depth(asset_no)
        best_bid = depth.best_bid
        best_ask = depth.best_ask
        if not np.isfinite(best_bid) or not np.isfinite(best_ask) or best_bid <= 0.0 or best_ask <= 0.0:
            continue

        mid = 0.5 * (best_bid + best_ask)
        push_mid(ts, mid)

        # 用 market trades 更新 1m bars（若无成交，以 mid 兜底）
        last_trades = hbt.last_trades(asset_no)
        seen_trade = False
        for t in last_trades:
            on_trade_price(t.exch_ts, t.px)
            seen_trade = True
        if not seen_trade:
            on_trade_price(ts, mid)
        hbt.clear_last_trades(asset_no)

        # 风控：净仓位（quote 计）= position * mid
        pos = hbt.position(asset_no)
        net_quote = pos * mid
        if np.abs(net_quote) > cfg.maxNetPosQuote:
            # 超限本轮不报价
            continue

        # 节流：与 TS 相同（quoteRefreshMs * 0.8）
        if ts - last_reconcile_ts < np.int64(cfg.test_interval_ms * 0.8) * 1_000_000:
            continue
        last_reconcile_ts = ts

        # 波动尺度
        vol_scale = vol_scale_from_mids(mids, mids_ts, mid_cnt, vol_lookback_ns)

        # Alpha 与中心
        centerShiftBps, skewAlpha = compute_alpha(
            mid, depth, cfg.obiDepth,
            cfg.obiWeight, cfg.bpWeight, cfg.micropriceWeight, cfg.alphaToPriceBps,
            bar_o, bar_h, bar_l, bar_c, bar_cnt, cfg.bpLookback
        )

        # 库存 skew（用 quote 口径），与 TS 同构的 tanh 饱和
        dev = net_quote - cfg.targetInventoryQuote
        invSkew = -cfg.inventorySkewK * np.tanh(dev / max(1.0, cfg.maxNetPosQuote))

        # 计算目标价
        if cfg.engine == "AS":
            bid_raw, ask_raw, halfBps = compute_quotes_AS(mid, depth, vol_scale, centerShiftBps, skewAlpha, invSkew,riskAversionGamma=cfg.riskAversionGamma,baseHalfSpreadBps=cfg.baseHalfSpreadBps,volHalfSpreadK=cfg.volHalfSpreadK)
        else:
            bid_raw, ask_raw, halfBps = compute_quotes_GLFT(mid, depth, vol_scale, invSkew,cfg=cfg)

        bid = floor_to_tick(bid_raw, tick)
        ask = ceil_to_tick(ask_raw, tick)

        # 生成分层报价计划
        max_layers = max(1, cfg.maxOrdersPerSide)
        plans_side = np.empty(max_layers * 2, np.int8)
        plans_px   = np.empty(max_layers * 2, np.float64)
        plans_qty  = np.empty(max_layers * 2, np.float64)

        base_qty = np.ceil(cfg.baseOrderQty / lot) * lot
        # 第一层
        plans_side[0] = BUY;  plans_px[0] = bid; plans_qty[0] = base_qty
        plans_side[1] = SELL; plans_px[1] = ask; plans_qty[1] = base_qty
        idx = 2
        for i in range(2, max_layers + 1):
            expand = (i * halfBps) / 1e4
            b2 = floor_to_tick(bid * (1.0 - expand), tick)
            a2 = ceil_to_tick(ask * (1.0 + expand), tick)
            plans_side[idx] = BUY;  plans_px[idx] = b2; plans_qty[idx] = base_qty; idx += 1
            plans_side[idx] = SELL; plans_px[idx] = a2; plans_qty[idx] = base_qty; idx += 1

        # 现有挂单 -> 需要撤销的（保持 “一轮最多撤 1 单”）
        orders = hbt.orders(asset_no)
        need_cancel_id = np.int64(0)
        order_values= orders.values()
        while order_values.has_next():
            o = order_values.get()
            # 如果订单价不在计划集里，或需要 bump，则考虑撤销
            # 1) 不在计划价集合
            in_plan = False
            for j in range(idx):
                if plans_side[j] == o.side:
                    # 对齐到 tick 比较
                    if np.abs(o.price - plans_px[j]) < 1e-12:
                        in_plan = True
                        break

            if not in_plan:
                # 与该 side 的第一目标价比较偏离
                best_target = 0.0
                for j in range(idx):
                    if plans_side[j] == o.side:
                        best_target = plans_px[j]
                        break
                if best_target != 0.0:
                    if bps_distance(o.price, best_target) > cfg.repriceThresholdBps:
                        need_cancel_id = o.order_id
                        break
                else:
                    need_cancel_id = o.order_id
                    break
            else:
                # 2) 队列过拥挤 -> bump 1 tick（仅入撤单清单，不直接改价）
                # 拟合拥挤度：价位总量 / 对手最优量
                my_px = o.price
                my_tick = int(round(my_px / tick))
                if o.side == BUY:
                    top_total = depth.bid_qty_at_tick(my_tick)
                    opp_top = depth.best_ask_qty
                else:
                    top_total = depth.ask_qty_at_tick(my_tick)
                    opp_top = depth.best_bid_qty
                opp_top = max(opp_top, lot)
                ratio = min(1.0, top_total / opp_top)
                if ratio > cfg.queueSizeAheadPctLimit:
                    # 估算 bump 后是否仍在容忍阈内
                    bump_px = my_px + (tick * cfg.queueBumpTicks if o.side == BUY else -tick * cfg.queueBumpTicks)
                    # 与该 side 的第一目标价比较
                    desired_first = 0.0
                    for j in range(idx):
                        if plans_side[j] == o.side:
                            desired_first = plans_px[j]
                            break
                    if desired_first != 0.0 and bps_distance(bump_px, desired_first) <= cfg.repriceThresholdBps * 3.0:
                        need_cancel_id = o.order_id
                        break

        # 撤单节流：如需撤且距离上次撤单已超过阈值 -> 撤第一个并本轮不补
        if need_cancel_id != 0 and (ts - last_cancel_ts) > cancel_throttle_ns:
            res = hbt.cancel(asset_no, need_cancel_id, True)
            last_cancel_ts = ts
            # 与 TS 一致：本轮直接 return（下一轮再补）
            # 等效为：继续下一 tick
            continue

        # 下单补齐（仅对缺失的计划价位补单；Post Only GTX + LIMIT）
        for j in range(idx):
            ps, px, qty = plans_side[j], plans_px[j], plans_qty[j]
            # 是否已存在
            exists = False
            order_values = orders.values()
            while order_values.has_next():
                o = order_values.get()
                if o.side == ps and np.abs(o.price - px) < 1e-12:
                    exists = True
                    break
            if exists:
                continue
            if ps == BUY:
                hbt.submit_buy_order(asset_no, next_oid, px, qty, GTX, LIMIT, True)
            else:
                hbt.submit_sell_order(asset_no, next_oid, px, qty, GTX, LIMIT, True)
            next_oid += 1
        stat.record(hbt)


data = np.concatenate(
[np.load('data\\binance_spot\\solfdusd_{}.npz'.format(date))['data'] for date in [20251011, 20251012, 20251013]]
)
initial_snapshot = np.load('data\\binance_spot\\solfdusd_20251010_eod.npz')['data']
latency_data = np.concatenate(
[np.load('data\\binance_spot\\solfdusd_{}_latency.npz'.format(date))['data'] for date in [20251011, 20251012, 20251013]]
)

def test(cfg):
    roi_lb = 50
    roi_ub = 500


    asset = (
        BacktestAsset()
            .data(data)
            .initial_snapshot(initial_snapshot)
            .linear_asset(1.0)
            .intp_order_latency(latency_data)
            .power_prob_queue_model(2)
            .no_partial_fill_exchange()
            .trading_value_fee_model(0.0000, 0.0003)
            .tick_size(0.01)
            .lot_size(0.001)
            .roi_lb(roi_lb)
            .roi_ub(roi_ub)
    )

    hbt = ROIVectorMarketDepthBacktest([asset])

    recorder = Recorder(1, 30_000_000)

    strategy(hbt,recorder.recorder,cfg)


    hbt.close()
    stats = LinearAssetRecord(recorder.get(0)).stats(book_size=10_000_000)
    return stats





from dataclasses import asdict
# pip install optuna numpy pandas
import optuna, numpy as np, pandas as pd
from optuna.trial import TrialState
def objective(trial: optuna.Trial = None):
    # 创建配置副本以避免修改原始配置
    cfg = copy.deepcopy(CFG)

    # 定义要优化的参数范围
    # AS 工业化参数
    cfg.baseHalfSpreadBps = trial.suggest_float("baseHalfSpreadBps", 0.1, 2.0)
    cfg.riskAversionGamma = trial.suggest_float("riskAversionGamma", 0.01, 1.0)
    cfg.volHalfSpreadK = trial.suggest_float("volHalfSpreadK", 0.01, 10)
    cfg.inventorySkewK = trial.suggest_float("inventorySkewK", 0.01, 10)



    # Alpha / 价心偏移
    cfg.obiWeight = trial.suggest_float("obiWeight", 0.01, 5.0)
    cfg.bpLookback = trial.suggest_int("bpLookback", 0.01, 10)
    cfg.bpWeight = trial.suggest_float("bpWeight", 0.001, 5.0)
    cfg.micropriceWeight = trial.suggest_float("micropriceWeight", 0.01, 1.0)
    cfg.alphaToPriceBps = trial.suggest_float("alphaToPriceBps", 0.01, 10.0)

    # 做市规模/订单管理
    cfg.repriceThresholdBps = trial.suggest_float("repriceThresholdBps", 0.01, 10.0)
    cfg.queueSizeAheadPctLimit = trial.suggest_float("queueSizeAheadPctLimit", 0.01, 10)

    # 设置用户属性以便后续分析
    for key, value in asdict(cfg).items():
        trial.set_user_attr(key, value)

    stats = test(cfg)
    return stats.splits[0]['Return'] * 1e4, stats.splits[0]['DailyNumberOfTrades'], -stats.splits[0]['MaxDrawdown'] * 1e4


# 优化
study = optuna.load_study(study_name="mm-custom-30m",storage= "mysql://optuna:AyHfbtAyAiRjR4ck@47.86.7.11/optuna")
study.optimize(objective, n_trials=int(3 * 1e3), n_jobs=1)



