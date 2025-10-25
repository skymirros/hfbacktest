# -*- coding: utf-8 -*-
"""
纯 Python 版 TradingIntensityIndicator
--------------------------------------
用途：在线估计 AS 模型里到达强度 λ(δ) = α * exp(-κ δ) 的 (α, κ)。
做法：把滑动窗口内的真实成交按“与中价的绝对价差”分桶聚合，然后用 exp 衰减曲线拟合出 (α, κ)。

依赖：
- numpy
- scipy.optimize.curve_fit

说明：
- 为了兼容原始 Cython 版，保留了 c_* 方法名（但都是普通 Python 函数）。
- 仍然以 Decimal 存储 _alpha/_kappa；current_value 返回 (alpha, kappa)。
- 事件订阅（监听 TradeEvent）在无 Hummingbot 环境时需要你自己对接（见下面注释）。
"""

import warnings
from decimal import Decimal
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning


class TradesForwarder:
    """
    成交事件转发器：在 HB 中继承自 EventListener 并在 c_call 中转发。
    纯 Python 里简单实现为可调用对象，order_book 侧需要调用 forwarder(event)。
    """
    def __init__(self, indicator: "TradingIntensityIndicator"):
        self._indicator = indicator

    # 兼容写法：既可作为回调函数（__call__），也提供 c_call 名称
    def __call__(self, arg: Any):
        self.c_call(arg)

    def c_call(self, arg: Any):
        self._indicator.c_register_trade(arg)


class TradingIntensityIndicator:
    """
    交易强度指示器：
    - 收集滑窗内的成交，按“成交前最近一次中价”计算 price_level=|trade.price - mid|
    - 把 (price_level, amount) 聚合成（距离 → 总量）关系
    - 用 y = a * exp(-b * x) 拟合得到 (alpha=a, kappa=b)
    """

    def __init__(self,
                 order_book: Any,
                 price_delegate: Any,
                 sampling_length: int = 30):
        """
        :param order_book: 需要能订阅成交事件的对象
            - 在 Hummingbot 里：order_book.c_add_listener(OrderBookEvent.TradeEvent, callback)
            - 在你自己的环境里：请实现 add_listener("trade", callback) 或等价功能
        :param price_delegate: 需要能提供中价的方法
            - 在 HB 里：price_delegate.get_price_by_type(PriceType.MidPrice)
            - 你也可以传入具备 get_mid() 方法的对象，并在 c_calculate 里做兼容判断
        :param sampling_length: 滑动窗口的离散时间片个数（见 c_calculate 内部逻辑）
        """
        self._alpha: Decimal = Decimal("0")
        self._kappa: Decimal = Decimal("0")

        # 时间片 → [{price_level, amount}, ...]
        self._trade_samples: Dict[Any, List[Dict[str, float]]] = {}

        # 暂存“刚收到但尚未和中价序列对齐”的成交列表（原代码中的 _current_trade_sample）
        self._current_trade_sample: List[Any] = []

        # 事件转发器
        self._trades_forwarder = TradesForwarder(self)

        # 订单簿：尝试兼容不同的监听接口名
        self._order_book = order_book
        try:
            # Hummingbot 接口
            from hummingbot.core.event.events import OrderBookEvent as _OBE
            self._order_book.c_add_listener(_OBE.TradeEvent, self._trades_forwarder)
        except Exception:
            # 你自己的接口：请确保会在成交时调用 self._trades_forwarder(event)
            # 例如：order_book.add_listener("trade", self._trades_forwarder)
            pass

        # 中价委托：原版使用 get_price_by_type(PriceType.MidPrice)
        self._price_delegate = price_delegate

        # 滑动窗口长度（按“离散键”数量）
        self._sampling_length: int = int(sampling_length)
        self._samples_length: int = 0  # 用来判断“是否变化”的基准
        self._last_quotes: List[Dict[str, float]] = []  # 降序保存 [{'timestamp', 'price'}, ...]

        # 静音 scipy 优化的告警（与原版一致）
        warnings.simplefilter("ignore", OptimizeWarning)

    # ------------------ 只读属性 ------------------ #
    @property
    def current_value(self) -> Tuple[Decimal, Decimal]:
        """返回 (alpha, kappa)"""
        return self._alpha, self._kappa

    @property
    def is_sampling_buffer_full(self) -> bool:
        """滑动窗口是否已装满 sampling_length 个时间片"""
        return len(self._trade_samples.keys()) == self._sampling_length

    @property
    def is_sampling_buffer_changed(self) -> bool:
        """
        滑动窗口是否发生变化（键数量是否改变）
        - 上层可以用它来决定是否触发一次取值/计算
        """
        is_changed = self._samples_length != len(self._trade_samples.keys())
        self._samples_length = len(self._trade_samples.keys())
        return is_changed

    # ------------------ 读写属性 ------------------ #
    @property
    def sampling_length(self) -> int:
        return self._sampling_length

    @sampling_length.setter
    def sampling_length(self, new_len: int):
        self._sampling_length = int(new_len)

    # 仅用于单元测试的辅助接口（保持原名）
    @property
    def last_quotes(self) -> list:
        return self._last_quotes

    @last_quotes.setter
    def last_quotes(self, value):
        self._last_quotes = value

    # ------------------ 计算入口 ------------------ #
    def calculate(self, timestamp: float):
        """与原版一致的包装：外层可调用 calculate(ts)"""
        self.c_calculate(timestamp)

    # ------------------ 核心：对齐中价并入桶 ------------------ #
    def c_calculate(self, timestamp: float):
        """
        1) 读取当前中价并头插入 _last_quotes（保持降序：最新在前）
        2) 遍历 _current_trade_sample，每笔成交找到“成交前最近一次中价”的 quote
        3) 计算 price_level = |trade.price - quote.price|，并按 (quote.timestamp + 1) 作为键入桶
        4) 裁剪 _last_quotes 和 _trade_samples，维持滑窗长度
        5) 若滑窗已满，则拟合 (alpha, kappa)
        """
        price = self._get_mid_price()
        # 降序保存最近的中价快照
        self._last_quotes = [{'timestamp': timestamp, 'price': price}] + self._last_quotes

        latest_processed_quote_idx: Optional[int] = None

        # 遍历所有“尚未对齐”的成交
        for trade in self._current_trade_sample:
            # 找到“成交前最近一次中价快照”
            for i, quote in enumerate(self._last_quotes):
                if quote["timestamp"] < self._get_trade_timestamp(trade):
                    if latest_processed_quote_idx is None or i < latest_processed_quote_idx:
                        latest_processed_quote_idx = i

                    price_level = abs(self._get_trade_price(trade) - float(quote["price"]))
                    amount = self._get_trade_amount(trade)

                    # 把该成交归到“中价后的下一个离散时间片”（原代码是 +1）
                    key = quote["timestamp"] + 1
                    if key not in self._trade_samples:
                        self._trade_samples[key] = []
                    self._trade_samples[key].append({"price_level": price_level, "amount": amount})
                    break

        # 当前批次的成交都处理完
        self._current_trade_sample = []

        # 截断 _last_quotes：只保留“处理到的最远索引 + 1”之前的部分，避免无限增长
        if latest_processed_quote_idx is not None:
            self._last_quotes = self._last_quotes[: latest_processed_quote_idx + 1]

        # 控制窗口长度：只保留最近 sampling_length 个键
        if len(self._trade_samples.keys()) > self._sampling_length:
            timestamps = sorted(self._trade_samples.keys())
            timestamps = timestamps[-self._sampling_length:]
            self._trade_samples = {ts: self._trade_samples[ts] for ts in timestamps}

        # 若窗口已满，进行参数拟合
        if self.is_sampling_buffer_full:
            self.c_estimate_intensity()

    # ------------------ 接收成交（事件回调） ------------------ #
    def register_trade(self, trade: Any):
        """与原版一致的包装：外层可直接调用 register_trade(trade)"""
        self.c_register_trade(trade)

    def c_register_trade(self, trade: Any):
        """把成交事件先暂存，等到下一次 c_calculate 再对齐入桶"""
        self._current_trade_sample.append(trade)

    # ------------------ 参数拟合：exp 回归得到 (α, κ) ------------------ #
    def c_estimate_intensity(self):
        """
        把滑窗内所有时间片里的样本按 price_level 聚合：
            trades_consolidated[price_level] = 总 amount
        然后对 (price_level, lambda) 做 y = a * exp(-b * x) 拟合，得到 a=alpha, b=kappa。
        """
        trades_consolidated: Dict[float, float] = {}
        price_levels: List[float] = []

        # 1) 聚合（距离 → 总量）
        for ts, tick_list in self._trade_samples.items():
            for trade in tick_list:
                lvl = float(trade['price_level'])
                if lvl not in trades_consolidated:
                    trades_consolidated[lvl] = 0.0
                    price_levels.append(lvl)
                trades_consolidated[lvl] += float(trade['amount'])

        # 2) 距离降序排序（和原实现一致）
        price_levels = sorted(price_levels, reverse=True)

        # 3) 取出对应的“强度近似值”（这里用总量作为强度 proxy）
        lambdas = [trades_consolidated[lvl] for lvl in price_levels]

        # 4) 为了能取 log / 拟合，0 值替换成极小正数
        lambdas_adj = [1e-10 if x == 0 else x for x in lambdas]

        # 5) 指数曲线拟合：y = a * exp(-b * x)，约束 a>=0, b>=0；初值使用上次结果以增强稳定性
        try:
            params, _ = curve_fit(
                lambda t, a, b: a * np.exp(-b * t),
                xdata=np.asarray(price_levels, dtype=np.float64),
                ydata=np.asarray(lambdas_adj, dtype=np.float64),
                p0=(float(self._alpha), float(self._kappa)),
                method='dogbox',
                bounds=([0.0, 0.0], [np.inf, np.inf]),
            )
            # 存回 Decimal（与原版一致）
            self._alpha = Decimal(str(params[0]))
            self._kappa = Decimal(str(params[1]))
        except (RuntimeError, ValueError):
            # 拟合失败就保持上次的 (alpha, kappa)
            pass

    # ------------------ 辅助：兼容不同环境的 mid/成交字段读取 ------------------ #
    def _get_mid_price(self) -> float:
        """
        读取中价：
        - HB：price_delegate.get_price_by_type(PriceType.MidPrice)
        - 其他：若对象实现了 get_mid()，则用之；否则抛错
        """
        # 尝试 HB 的接口
        try:
            from hummingbot.core.data_type.common import PriceType as _PT  # 可选导入
            return float(self._price_delegate.get_price_by_type(_PT.MidPrice))
        except Exception:
            # 你的自定义接口
            if hasattr(self._price_delegate, "get_mid"):
                return float(self._price_delegate.get_mid())
            raise RuntimeError("price_delegate 需要实现 get_price_by_type(PriceType.MidPrice) 或 get_mid()")

    @staticmethod
    def _get_trade_timestamp(trade: Any) -> float:
        """读取成交时间戳：支持 trade.timestamp 或 trade['timestamp'] 两种风格"""
        if hasattr(trade, "timestamp"):
            return float(trade.timestamp)
        if isinstance(trade, dict) and "timestamp" in trade:
            return float(trade["timestamp"])
        raise KeyError("trade 缺少 timestamp 字段")

    @staticmethod
    def _get_trade_price(trade: Any) -> float:
        """读取成交价格：支持 trade.price 或 trade['price']"""
        if hasattr(trade, "price"):
            return float(trade.price)
        if isinstance(trade, dict) and "price" in trade:
            return float(trade["price"])
        raise KeyError("trade 缺少 price 字段")

    @staticmethod
    def _get_trade_amount(trade: Any) -> float:
        """读取成交数量：支持 trade.amount 或 trade['amount']"""
        if hasattr(trade, "amount"):
            return float(trade.amount)
        if isinstance(trade, dict) and "amount" in trade:
            return float(trade["amount"])
        raise KeyError("trade 缺少 amount 字段")
