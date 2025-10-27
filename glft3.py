"""
1. 价差计算：使用 δ_bid + δ_ask 作为实际价差，理论价差作为参考
2. 库存归一化：显式处理 Δq，确保量纲一致性
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import math
from typing import Optional, Dict


@dataclass
class GLFTAsymptoticParams:
    """GLFT 渐近近似参数（包含库存步长）"""
    sigma_t: float   # 波动率 [tick/√sec]
    gamma_t: float   # 风险厌恶 [1/tick]
    q_step: float = 1.0  # 库存网格步长（最小交易单位）[股]
    
    def __post_init__(self):
        if self.sigma_t <= 0:
            raise ValueError("sigma_t 必须 > 0")
        if self.gamma_t <= 0:
            raise ValueError("gamma_t 必须 > 0")
        if self.q_step <= 0:
            raise ValueError("q_step 必须 > 0")


class GLFTAsymptoticMarketMaker:
    """
    GLFT 渐近闭式近似做市商（最终修正版本）
    """
    
    def __init__(self, params: GLFTAsymptoticParams):
        self.p = params
    
    def _normalize_inventory(self, q: float) -> float:
        """库存归一化：i = q / Δq"""
        return q / self.p.q_step
    
    def _compute_common_risk_factor(self, A: float, k_t: float) -> float:
        """计算风险因子"""
        sigma2 = self.p.sigma_t ** 2
        gamma = self.p.gamma_t
        k = max(k_t, 1e-12)
        A = max(A, 1e-12)
        
        term = (1 + gamma/k) ** (1 + k/gamma)
        return math.sqrt((sigma2 * gamma) / (2 * k * A) * term)
    
    def _compute_base_spread_component(self, k_t: float) -> float:
        """计算基准价差分量"""
        gamma = self.p.gamma_t
        k = max(k_t, 1e-12)
        return (1.0 / gamma) * math.log(1.0 + gamma / k)
    
    def compute_optimal_quotes(self, 
                             q: float,
                             A_bid: float, k_bid_t: float,
                             A_ask: float, k_ask_t: float) -> Dict[str, float]:
        """
        计算最优报价（采纳建议修正）
        """
        # 库存归一化（关键修正）
        i = self._normalize_inventory(q)
        
        # 计算基准分量和风险因子
        C_bid = self._compute_base_spread_component(k_bid_t)
        C_ask = self._compute_base_spread_component(k_ask_t)
        risk_factor_bid = self._compute_common_risk_factor(A_bid, k_bid_t)
        risk_factor_ask = self._compute_common_risk_factor(A_ask, k_ask_t)
        
        # 应用归一化库存公式
        delta_bid = C_bid + (2 * i + 1) / 2 * risk_factor_bid
        delta_ask = C_ask - (2 * i - 1) / 2 * risk_factor_ask
        
        # 价差计算修正：使用实际报价之和
        actual_spread = delta_bid + delta_ask
        
        
        return {
            # 核心输出
            "delta_bid": delta_bid,
            "delta_ask": delta_ask, 
            "actual_spread": actual_spread,      # 主要使用的价差
            # 参考信息
            "normalized_inventory": i,
            # 诊断信息
            "C_bid": C_bid, "C_ask": C_ask,
            "risk_factor_bid": risk_factor_bid, "risk_factor_ask": risk_factor_ask,
        }



def demonstrate_spread_calculation():
    """演示价差计算修正"""
    print("\n=== 价差计算修正验证 ===\n")
    
    params = GLFTAsymptoticParams(sigma_t=4, gamma_t=0.001, q_step=0.1)
    mm = GLFTAsymptoticMarketMaker(params)
    
    # 非对称参数案例
    result = mm.compute_optimal_quotes(
        q=0.001,
        A_bid=0.0235, k_bid_t=1.219,  # bid侧流动性差
        A_ask=0.0235, k_ask_t=1.219   # ask侧流动性好  
    )
    
    print("非对称参数下的价差计算：")
    print(f"Bid侧: A=0.5, k=0.2 → δ_bid = {result['delta_bid']:.4f}")
    print(f"Ask侧: A=1.5, k=0.4 → δ_ask = {result['delta_ask']:.4f}")
    print(f"实际价差 (δ_bid + δ_ask): {result['actual_spread']:.4f}")


if __name__ == "__main__":
    # demonstrate_correction_necessity()
    demonstrate_spread_calculation()