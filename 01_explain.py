
# ===== JoinQuant 多因子选股（可直接回测）=====
# 因子：动量(20/5日)、低波动(20日反向)、流动性(Amihud反向)、价值(1/PE或1/PB)、规模(对数流通市值, 取反)、换手率(近20日)
# 处理：截面MAD去极值 + ZScore标准化 + 规模中性化（简化）+ 线性加权合成
# 选股：每周一调仓，股票池=中证500；Top 20% 等权；含交易成本；T+1生效
# 回测引擎：JoinQuant（网页版）

import numpy as np                      # 数值运算
import pandas as pd                     # 表格数据处理
from jqdata import *                    # 聚宽数据/交易API
from statsmodels.api import OLS, add_constant  # 规模中性化时做线性回归

# ========== 参数 ==========
INDEX_POOL = '000905.XSHG'    # 股票池：中证500指数组成股
REBAL_FREQ = 5                # 调仓频率：每5个交易日计算一次信号（周频）
TOP_Q = 0.20                  # 选股比例：截面得分前20%
COST = 0.001                  # 单边交易费率：0.1%（回测中计入成本）
WEIGHTS = {                   # 各因子的线性合成权重
    'MOM20': 0.30,            # 20日动量
    'MOM5' : 0.10,            # 5日动量（短因子）
    'LOWVOL20': 0.25,         # 20日波动率的相反数（越低越好）
    'ILLIQ20' : 0.10,         # Amihud非流动性相反数（越流动越好）
    'VALUE'   : 0.15,         # 价值：1/PE或1/PB
    'SIZE'    : 0.05,         # 规模：对数流通市值取负（小盘溢价）
    'TURN20'  : 0.05,         # 20日换手（作为活跃度/价格发现 proxy）
}
MIN_STOCKS = 30               # 最少持仓数，避免过度集中

def initialize(context):
    set_benchmark('000300.XSHG')   # 基准指数：沪深300
    set_slippage(FixedSlippage(0)) # 滑点设为0（成本用费率覆盖）
    # 手续费设置：开/平仓费率相同，最低5元（聚宽订单最小佣金）
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=COST,
                             close_commission=COST, close_today_commission=COST, min_commission=5),
                   type='stock')
    run_daily(daily, time='open')   # 每日开盘运行：计算信号（仅在周频日真正计算）
    run_daily(rebalance, time='open')  # 每日开盘检查是否需要按昨日信号调仓（T+1）
    g.day_count = 0                 # 全局计数器：跟踪交易日
    g.target_weights = {}           # 全局目标权重：由 daily 计算、由 rebalance 执行

def daily(context):
    # —— 按周频生成信号（T日生成，T+1执行）——
    g.day_count += 1
    if g.day_count % REBAL_FREQ != 1:   # 只在“调仓前一日”生成信号
        return

    date = context.current_dt.date()    # 当前日期（datetime.date）
    pool = get_index_stocks(INDEX_POOL, date)  # 取指数成分股列表
    # 过滤特殊状态
    pool = [s for s in pool if not is_st_stock(s, date)]  # 剔除ST/风险警示股
    if len(pool) < MIN_STOCKS:          # 股票池过小则跳过
        g.target_weights = {}
        return

    # --- 行情数据：拉取近 ~6 个月K线用于滚动计算（余量避免空窗）
    start = get_trade_days(end_date=date, count=130)[0]   # 回溯130个交易日
    df = get_price(pool, start_date=start, end_date=date, frequency='daily',
                   fields=['open','high','low','close','volume','money'], panel=False)
    if df.empty:
        g.target_weights = {}
        return
    df = df.sort_values(['code','time'])       # 先按代码、时间排序
    df['ret'] = df.groupby('code')['close'].pct_change()  # 日收益率

    # --- 因子构造 ---
    # 动量：近20/5日收盘涨幅
    mom20 = df.groupby('code')['close'].pct_change(20)
    mom5  = df.groupby('code')['close'].pct_change(5)

    # 波动率：20日收益率标准差（越小越好，因此取负作为“低波动”）
    vol20 = df.groupby('code')['ret'].rolling(20).std().reset_index(level=0, drop=True)
    lowvol20 = -vol20

    # Amihud 非流动性：E(|ret|/金额)；越小越流动 → 取负得到“流动性越好得分越高”
    illiq = (df['ret'].abs() / (df['money'].replace(0, np.nan))).replace([np.inf, -np.inf], np.nan)
    illiq20 = -df.assign(illiq=illiq).groupby('code')['illiq'].rolling(20).mean().reset_index(level=0, drop=True)

    # 换手 proxy：20日平均成交量 / 20日平均成交额（粗略衡量活跃度，单位相对）
    turn20 = df.groupby('code')['volume'].rolling(20).mean().reset_index(level=0, drop=True) / \
             (df.groupby('code')['money'].rolling(20).mean().reset_index(level=0, drop=True) / 100.0)

    # 截面取“最新一日”的各指标
    last = df.groupby('code').tail(1).set_index('code')
    fac = pd.DataFrame({
        'MOM20': mom20.groupby(df['code']).tail(1).values,
        'MOM5' : mom5.groupby(df['code']).tail(1).values,
        'LOWVOL20': lowvol20.groupby(df['code']).tail(1).values,
        'ILLIQ20' : illiq20.groupby(df['code']).tail(1).values,
        'TURN20'  : turn20.groupby(df['code']).tail(1).values,
    }, index=last.index)

    # --- 基本面因子：价值/规模（以聚宽 valuation 表为准）
    q = query(valuation.code, valuation.pb, valuation.pe_ratio, valuation.circulating_market_cap) \
        .filter(valuation.code.in_(list(fac.index)))
    val = get_fundamentals(q, date=date).set_index('code')
    if not val.empty:
        fac['VALUE'] = 1.0 / val[['pe_ratio','pb']].replace({0: np.nan}).mean(axis=1)  # 1/PE或1/PB的平均
        fac['SIZE']  = -np.log(val['circulating_market_cap'].replace(0, np.nan))       # 规模越小越好 → 取负

    # --- 截面处理：去极值 & 标准化 & 规模中性化 ---
    def winsorize_mad(x, n=5.0):
        # MAD去极值（对异常值鲁棒）：clip 到 [中位数 ± n*MAD]
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med)) + 1e-12
        up, lo = med + n*1.4826*mad, med - n*1.4826*mad
        return np.clip(x, lo, up)

    for c in fac.columns:
        fac[c] = winsorize_mad(fac[c].values)                      # 1) 去极值
        fac[c] = (fac[c] - fac[c].mean()) / (fac[c].std() + 1e-12) # 2) ZScore 标准化

    # 3) 规模中性化（简化版）：对除 SIZE 以外的因子用 SIZE 做一元回归，取残差
    if 'SIZE' in fac.columns:
        base = add_constant(fac[['SIZE']].fillna(0).values)        # X = [1, SIZE]
        for c in fac.columns:
            if c == 'SIZE':
                continue
            y = fac[c].fillna(0).values
            try:
                resid = y - OLS(y, base).fit().predict(base)       # 残差 = 真实值 - 拟合值
                fac[c] = resid
            except:
                pass                                               # 回归失败时保持原值（避免报错中断）

    # --- 合成得分（线性加权） ---
    score = pd.Series(0.0, index=fac.index)
    for k, w in WEIGHTS.items():
        if k in fac.columns:
            score = score.add(w * fac[k].fillna(0), fill_value=0)  # 缺失按0处理以保证横截面可比
    fac['score'] = score

    # --- 选股：Top 20% 等权，至少 MIN_STOCKS 只 ---
    fac = fac.dropna(subset=['score']).sort_values('score', ascending=False)
    n = max(MIN_STOCKS, int(len(fac) * TOP_Q))
    picks = list(fac.head(n).index)
    if len(picks) == 0:
        g.target_weights = {}
        return
    w = 1.0 / len(picks)                      # 等权配置
    g.target_weights = {s: w for s in picks}  # 保存到全局，等到 T+1 执行

def rebalance(context):
    # —— T+1 调仓执行：根据昨日生成的 g.target_weights 调整仓位 ——
    if not g.target_weights:
        return
    # 1) 卖出不在目标持仓里的股票（完全清仓）
    for s in list(context.portfolio.positions.keys()):
        if s not in g.target_weights:
            order_target(s, 0)
    # 2) 买入/调仓使之达到目标权重
    for s, w in g.target_weights.items():
        order_target_percent(s, w)
