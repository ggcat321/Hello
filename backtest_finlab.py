# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 18:46:20 2022

@author: crazy
"""

import numpy as np
import pandas as pd
from FinMind import strategies
from FinMind.data import DataLoader
from FinMind.strategies.base import Strategy
from ta.momentum import StochasticOscillator


data_loader = DataLoader()
# data_loader.login(user_id, password) # 可選
obj = strategies.BackTest(
     stock_id="0056",
     start_date="2018-01-01",
     end_date="2019-01-01",
     trader_fund=500000.0,
     fee=0.001425,
     data_loader=data_loader,
)
obj.stock_price

class Kd(Strategy):
     """
     summary:
          日KD 80 20
          日K線 <= 20 進場
          日K線 >= 80 出場
     """
     kdays = 9
     kd_upper = 80
     kd_lower = 20
     def create_trade_sign(self, stock_price: pd.DataFrame) -> pd.DataFrame:
          stock_price = stock_price.sort_values("date")
          kd = StochasticOscillator(
               high=stock_price["max"],
               low=stock_price["min"],
               close=stock_price["close"],
               n=self.kdays,
          )
          rsv_ = kd.stoch().fillna(50)
          _k = np.zeros(stock_price.shape[0])
          _d = np.zeros(stock_price.shape[0])
          for i, r in enumerate(rsv_):
               if i == 0:
                    _k[i] = 50
                    _d[i] = 50
               else:
                    _k[i] = _k[i - 1] * 2 / 3 + r / 3
                    _d[i] = _d[i - 1] * 2 / 3 + _k[i] / 3
          stock_price["K"] = _k
          stock_price["D"] = _d
          stock_price.index = range(len(stock_price))
          stock_price["signal"] = 0
          stock_price.loc[stock_price["K"] <= self.kd_lower, "signal"] = 1
          stock_price.loc[stock_price["K"] >= self.kd_upper, "signal"] = -1
          return stock_price
      
obj.add_strategy(Kd)
obj.simulate()
obj.final_stats
obj.trade_detail
obj.plot()
