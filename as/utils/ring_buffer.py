# -*- coding: utf-8 -*-
"""
纯 Python 版 RingBuffer（等价于原 Cython 实现）
------------------------------------------------
- 依赖：numpy、logging
- 语义保持与 Cython 版一致：
  * 固定长度的环形缓冲区，按写入顺序覆盖最旧值
  * 仅当缓冲区“写满”时，mean/variance/std_dev 才返回有效数值，否则返回 NaN
  * get_as_numpy_array() 返回“从最旧到最新”的有序视图（复制出来的数组）
"""

import logging
from typing import Optional
import numpy as np

pmm_logger: Optional[logging.Logger] = None


class RingBuffer:
    """
    环形缓冲区（固定容量 FIFO）：
      - _buffer: numpy.float64 数组，存放数据
      - _delimiter: 下一个写入的位置（指针）
      - _is_full: 是否已经写满过一次（到达容量后置 True）
    """

    # —— 与原类保持一致的类方法 —— #
    @classmethod
    def logger(cls) -> logging.Logger:
        """
        获取模块级 logger；与原 Cython 版保持一致的入口。
        """
        global pmm_logger
        if pmm_logger is None:
            pmm_logger = logging.getLogger(__name__)
        return pmm_logger

    # —— 纯 Python 的构造 —— #
    def __init__(self, length: int):
        """
        构造函数（替代 Cython 的 __cinit__）：
        :param length: 缓冲区长度（容量）
        """
        if length <= 0:
            raise ValueError("length must be positive")
        self._length: int = int(length)
        self._buffer: np.ndarray = np.zeros(self._length, dtype=np.float64)
        self._delimiter: int = 0  # 下一个写入位置
        self._is_full: bool = False

    # —— 与原 c_add_value/c_increment_delimiter/c_is_empty 等价的内部方法 —— #
    def _increment_delimiter(self) -> None:
        """
        写指针向前推进一格；若回绕到 0，说明已覆盖一圈，标记为 full。
        （等价于 Cython: c_increment_delimiter）
        """
        self._delimiter = (self._delimiter + 1) % self._length
        if (not self._is_full) and (self._delimiter == 0):
            self._is_full = True

    def _is_empty(self) -> bool:
        """
        缓冲区是否“尚未写入任何元素”（未满且指针仍为 0）
        （等价于 Cython: c_is_empty）
        """
        return (not self._is_full) and (self._delimiter == 0)

    # —— 对外 API（与原 Python 包装层保持一致的命名） —— #
    def add_value(self, val: float) -> None:
        """
        写入一个值；覆盖当前指针位置，然后推进指针。
        （等价于 Cython: c_add_value + 公共 add_value 组合）
        """
        self._buffer[self._delimiter] = float(val)
        self._increment_delimiter()

    def get_last_value(self) -> float:
        """
        获取“最近写入的值”。如果为空，返回 np.nan。
        （等价于 Cython: c_get_last_value + 公共 get_last_value）
        """
        if self._is_empty():
            return float(np.nan)
        # Python 的负索引等价于 Cython 中 delimiter-1 的回绕语义
        return float(self._buffer[self._delimiter - 1])

    def get_as_numpy_array(self) -> np.ndarray:
        """
        以“时间顺序”（最旧→最新）返回当前窗口内的数据的拷贝。
        未满：返回 [0 : _delimiter)
        已满：返回 [_delimiter : end) + [0 : _delimiter)
        （等价于 Cython: c_get_as_numpy_array + 公共 get_as_numpy_array）
        """
        if not self._is_full:
            # 未写满：直接切出 [0, delimiter) 的有效段
            return self._buffer[: self._delimiter].copy()
        # 已写满：按回绕顺序拼接
        idx = (np.arange(self._delimiter, self._delimiter + self._length, dtype=np.int64) % self._length)
        return self._buffer[idx].copy()

    # —— 统计属性：保持“仅在写满后才返回有效值”的语义 —— #
    @property
    def is_full(self) -> bool:
        """是否已经写满过一圈（与原 c_is_full 语义一致）"""
        return self._is_full

    @property
    def mean_value(self) -> float:
        """
        均值：仅在写满后计算，否则返回 NaN
        （等价于 Cython: c_mean_value）
        """
        if not self._is_full:
            return float(np.nan)
        return float(np.mean(self.get_as_numpy_array()))

    @property
    def std_dev(self) -> float:
        """
        标准差：仅在写满后计算，否则返回 NaN
        （等价于 Cython: c_std_dev；注意这里为总体标准差，ddof=0）
        """
        if not self._is_full:
            return float(np.nan)
        return float(np.std(self.get_as_numpy_array()))

    @property
    def variance(self) -> float:
        """
        方差：仅在写满后计算，否则返回 NaN
        （等价于 Cython: c_variance；总体方差，ddof=0）
        """
        if not self._is_full:
            return float(np.nan)
        return float(np.var(self.get_as_numpy_array()))

    # —— 长度属性（与原 length getter/setter 语义一致） —— #
    @property
    def length(self) -> int:
        """缓冲区容量（与原 length 属性一致）"""
        return self._length

    @length.setter
    def length(self, value: int) -> None:
        """
        重设容量：按当前顺序拿到数据 → 清空并重建 → 仅回灌“最新的 value 个”数据。
        （等价于原实现：先取 get_as_numpy_array()，再重建并追加）
        """
        value = int(value)
        if value <= 0:
            raise ValueError("length must be positive")

        data = self.get_as_numpy_array()  # 先保存已有数据（最旧→最新）
        self._length = value
        self._buffer = np.zeros(self._length, dtype=np.float64)
        self._delimiter = 0
        self._is_full = False

        # 仅把“最近的 value 个”重新灌入（与原逻辑一致）
        for v in data[-value:]:
            self.add_value(float(v))
