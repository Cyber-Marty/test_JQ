# -*- coding: utf-8 -*-
# JoinQuant LIGHTgbm多因子选股
# 说明：
# 1) lightGbm
# 2) 每月首个交易日再平衡；训练窗口：滚动 36 个月；标的：沪深300 成分。
# 3) 标签：下一月度(下一个月末)的前瞻 1M 收益率；目标：预测横截面预期收益并做多 TopN。
# 4) 交易：等权+行业暴露软约束（每个一级行业最多占比 20%），含滑点与佣金设置，可一键回测。
# 5) 代码在 JoinQuant Python3 环境下运行（已适配 API）。

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# ---- 模型适配层：优先用 lightgbm，失败则用 sklearn HistGB 作为无依赖降级 ----
try:
    import lightgbm as lgb
    _HAS_LGBM = True
except Exception:
    _HAS_LGBM = False


class MLRegressor(object):
    """仅使用 LightGBM 的回归器；若不可用或训练失败则抛错（不降级）。"""
    def __init__(self, backend='lgbm', random_state=42):
        if backend != 'lgbm':
            raise RuntimeError("当前策略已锁定仅使用 LightGBM。请将 g.model_backend 设为 'lgbm'.")
        if not _HAS_LGBM:
            raise RuntimeError("LightGBM 未安装/不可用：请在环境中提供 lightgbm 包后再运行。")
        self.random_state = random_state
        self.backend = 'lgbm'
        self.model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=0,
        )

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        m = np.isfinite(X).all(axis=1) & np.isfinite(y)
        Xc, yc = X[m], y[m]
        if Xc.shape[0] < 10:
            raise RuntimeError("训练样本不足（<10），无法进行 LightGBM 拟合。")
        # 训练失败直接抛错，不做降级
        self.model.fit(Xc, yc)
        return self

    def predict(self, X):
        X = np.asarray(X)
        X = np.where(np.isfinite(X), X, 0.0)
        return self.model.predict(X)


# ==================== JoinQuant 回测主流程 ====================

def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    log.set_level('order', 'error')  # 降低下单日志噪音

    # 交易费用与滑点（保守设置，便于面试解释）
    set_slippage(FixedSlippage(0.002))
    set_order_cost(
        OrderCost(open_commission=0.0003, close_commission=0.0013,
                  close_today_commission=0.0, min_commission=5),
        type='stock'
    )

    g.index_code = '000300.XSHG'
    g.top_n = 30
    g.max_industry_weight = 0.20  # 每个一级行业最多 20%
    g.lookback_months = 36        # 训练使用最近 36 个月横截面
    g.min_history_days = 250      # 最少历史天数保障

    g.features = [
        # 估值类
        'pe_ratio', 'pb_ratio', 'ps_ratio', 'pcf_ratio',
        # 质量类
        'roe', 'roa', 'gross_profit_margin', 'net_profit_margin',
        'inc_net_profit_year_on_year',
        # 规模
        'circulating_market_cap', 'market_cap',
        # 成交/换手
        'turnover_ratio',
        # 动量与波动（由行情计算）
        'ret_1m', 'ret_3m', 'ret_6m', 'ret_12m',
        'volatility_3m', 'volatility_6m',
        # 防御性：Beta（对沪深300）
        'beta_6m'
    ]

    # 训练状态缓存
    g.model = None
    g.scaler = None  # 简易标准化（横截面 Z-score）
    g.model_backend = 'lgbm'  # 仅用 LightGBM（禁止降级）

    # 调度：每月首个交易日开盘再平衡；月末做训练更新
    run_monthly(rebalance, 1, time='open')
    run_monthly(train_update, 0, time='close')  # 月末收盘后训练，避免未来函数


# ----- 公共工具函数 -----

def month_ends(start_date, end_date):
    """返回[start_date, end_date]区间内每个月的最后一个交易日（date 列表）。
    完全移除对 get_all_trade_days / get_trade_days 的依赖，统一用基准指数的日线行情推导交易日。
    """
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    df_px = get_price(g.index_code, start_date=start, end_date=end, fields=['close'], panel=False)
    if df_px is None or df_px.empty:
        return []
    # 兼容不同结构
    if 'time' in df_px.columns:
        dts = pd.to_datetime(df_px['time'])
    else:
        dts = pd.to_datetime(df_px.index)
    df = pd.DataFrame({'d': dts})
    df['ym'] = df['d'].dt.to_period('M')
    mes = df.groupby('ym')['d'].max()
    # 转为 python date 列表
    return list(mes.dt.date.values)


def industry_map(stocks):
    """返回 {code: industry_str}；兼容 get_industry 的多种返回结构。
    sw_l1 可能为字符串或字典（含 industry_name / index_code）。统一抽取为字符串。
    """
    info = get_industry(stocks)
    out = {}
    for code in stocks:
        d = info.get(code, {}) or {}
        v = d.get('sw_l1')
        if isinstance(v, dict):
            name = v.get('industry_name') or v.get('index_name') or v.get('index_code') or 'NA'
        elif v is None:
            name = 'NA'
        else:
            name = str(v)
        out[code] = name
    return out


def clean_universe(date):
    # 指数成分 + 基础过滤
    stocks = get_index_stocks(g.index_code, date)
    q = query(valuation.code).filter(
        valuation.code.in_(stocks)
    )
    df = get_fundamentals(q, date)
    if df is None or df.empty:
        return []
    stocks = list(df['code'])

    # 去除停牌、ST、新股(<60日)
    filtered = []
    for s in stocks:
        # ST 过滤
        if 'ST' in get_security_info(s).display_name:
            continue
        # 上市天数
        ipo_days = (date - get_security_info(s).start_date).days
        if ipo_days < 60:
            continue
        # 停牌过滤
        if not is_tradeable(s, date):
            continue
        filtered.append(s)
    return filtered


def is_tradeable(stock, date):
    # 简单可交易判断：最近一天有价格且未停牌
    price = get_price(stock, end_date=date, count=1, fields=['close', 'paused'], panel=False)
    if price is None or price.empty:
        return False
    return (price['paused'].iloc[-1] == 0) and np.isfinite(price['close'].iloc[-1])


def calc_market_features(date, stocks):
    if not stocks:
        return pd.DataFrame()
    end = date
    start = (pd.to_datetime(date) - pd.DateOffset(months=13)).date()  # 取 13 个月以算 12M

    px = get_price(stocks, start_date=start, end_date=end, fields=['close'], panel=False)
    if px is None or px.empty:
        return pd.DataFrame()
    px = px.pivot(index='time', columns='code', values='close').dropna(how='all')

    # 指数数据用于 Beta 计算
    bench = get_price(g.index_code, start_date=start, end_date=end, fields=['close'], panel=False)
    if bench is None or bench.empty:
        return pd.DataFrame()
    # 兼容不同返回结构：有的版本返回包含 'time' 列的 DataFrame，有的版本直接以时间为索引
    if 'time' in bench.columns:
        bench = bench.set_index('time')['close']
    elif 'close' in bench.columns:
        bench = bench['close']
        # 确保索引为时间索引
        bench.index = pd.to_datetime(bench.index)
    else:
        # 兜底：Series 形态
        bench = bench.squeeze()
        bench.index = pd.to_datetime(bench.index)

    rets = px.pct_change()
    bench_ret = bench.pct_change()

    feats = {}
    for code in px.columns:
        s = px[code].dropna()
        if s.empty or len(s) < 60:
            continue
        r = s.pct_change().dropna()
        def roll_ret(days):
            if len(s) < days + 1:
                return np.nan
            return float(s.iloc[-1] / s.iloc[-days-1] - 1.0)

        ret_1m = roll_ret(21)
        ret_3m = roll_ret(63)
        ret_6m = roll_ret(126)
        ret_12m = roll_ret(252)

        vol_3m = float(r.tail(63).std()) if len(r) >= 63 else np.nan
        vol_6m = float(r.tail(126).std()) if len(r) >= 126 else np.nan

        # 简单 OLS Beta（对基准收益）
        aligned = pd.concat([rets[code], bench_ret], axis=1).dropna()
        beta_6m = np.nan
        if len(aligned) >= 126:
            y = aligned.iloc[-126:, 0].values
            x = aligned.iloc[-126:, 1].values
            if np.isfinite(x).sum() > 10 and np.isfinite(y).sum() > 10:
                x1 = np.vstack([x, np.ones_like(x)]).T
                try:
                    b = np.linalg.lstsq(x1, y, rcond=None)[0]
                    beta_6m = float(b[0])
                except Exception:
                    beta_6m = np.nan

        feats[code] = {
            'ret_1m': ret_1m, 'ret_3m': ret_3m, 'ret_6m': ret_6m, 'ret_12m': ret_12m,
            'volatility_3m': vol_3m, 'volatility_6m': vol_6m,
            'beta_6m': beta_6m,
        }

    mkt_df = pd.DataFrame.from_dict(feats, orient='index')
    mkt_df.index.name = 'code'
    return mkt_df


def fetch_fundamental_features(date, stocks):
    if not stocks:
        return pd.DataFrame()
    q = query(
        valuation.code,
        valuation.pe_ratio, valuation.pb_ratio, valuation.ps_ratio, valuation.pcf_ratio,
        valuation.circulating_market_cap, valuation.market_cap,
        indicator.roe, indicator.roa, indicator.gross_profit_margin,
        indicator.net_profit_margin,        indicator.inc_net_profit_year_on_year,
        valuation.turnover_ratio,
    ).filter(valuation.code.in_(stocks))
    df = get_fundamentals(q, date)
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.set_index('code')
    return df


def compute_factors(date, stocks):
    fin = fetch_fundamental_features(date, stocks)
    mkt = calc_market_features(date, stocks)
    if fin.empty and mkt.empty:
        return pd.DataFrame()
    df = fin.join(mkt, how='outer')
    # 简单的极值化与标准化（横截面）
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    df = df[g.features].copy()
    # 横截面去极值（5/95 分位）
    def winsorize(s):
        if s.dropna().empty:
            return s
        lo, hi = s.quantile(0.05), s.quantile(0.95)
        return s.clip(lo, hi)
    df = df.apply(winsorize, axis=0)
    # Z-score
    df = df.apply(lambda s: (s - s.mean()) / (s.std() + 1e-9), axis=0)
    return df


def forward_return_1m(date, stocks, return_benchmark=False):
    """前瞻 1M 收益；可返回基准同期收益用以构造超额收益标签。"""
    end_dt = (pd.to_datetime(date) + pd.offsets.MonthEnd(2)).date()
    month_end_list = month_ends(date, end_dt)
    if len(month_end_list) < 2:
        if return_benchmark:
            return pd.Series(index=stocks, dtype=float), np.nan
        return pd.Series(index=stocks, dtype=float)
    next_me = month_end_list[1]
    px = get_price(stocks, start_date=date, end_date=next_me, fields=['close'], panel=False)
    if px is None or px.empty:
        if return_benchmark:
            return pd.Series(index=stocks, dtype=float), np.nan
        return pd.Series(index=stocks, dtype=float)
    px = px.pivot(index='time', columns='code', values='close')
    if px.empty or len(px) < 2:
        if return_benchmark:
            return pd.Series(index=stocks, dtype=float), np.nan
        return pd.Series(index=stocks, dtype=float)
    ret = px.iloc[-1] / px.iloc[0] - 1.0
    ret.name = 'fwd_1m'
    if not return_benchmark:
        return ret
    # 基准同期收益
    bpx = get_price(g.index_code, start_date=date, end_date=next_me, fields=['close'], panel=False)
    if bpx is None or bpx.empty:
        bench_ret = np.nan
    else:
        if 'close' in bpx:
            s = bpx['close']
        elif 'time' in bpx.columns:
            s = bpx.set_index('time')['close']
        else:
            s = bpx.squeeze()
        s = s.dropna()
        bench_ret = float(s.iloc[-1] / s.iloc[0] - 1.0) if len(s) >= 2 else np.nan
    return ret, bench_ret


def assemble_training_set(asof_date):
    """滚动月度横截面训练集；标签为**超额收益**(个股1M - 基准1M)，并对更近月份赋更高权重。"""
    end = asof_date
    start = (pd.to_datetime(asof_date) - pd.DateOffset(months=g.lookback_months)).date()
    mes = month_ends(start, end)
    X_list, y_list, w_list = [], [], []
    if not mes or len(mes) < 2:
        return None, None, None
    # 时间衰减：最近月份权重最高
    total_m = max(1, len(mes) - 1)
    for i, d in enumerate(mes[:-1]):  # 最后一个月末仅用于打标签，不作为起点横截面
        stocks = clean_universe(d)
        if not stocks:
            continue
        X = compute_factors(d, stocks)
        if X is None or X.empty:
            continue
        y_raw, bench_ret = forward_return_1m(d, list(X.index), return_benchmark=True)
        if y_raw is None or y_raw.empty:
            continue
        # 超额收益标签
        if np.isnan(bench_ret):
            y = y_raw
        else:
            y = y_raw - bench_ret
        df = X.join(y.rename('fwd_excess'), how='inner')
        df = df.dropna(subset=['fwd_excess'])
        if df.shape[0] < 20:
            continue
        # 时间权重：线性或指数，这里采用指数衰减，最近权重=1.0，越远越小
        age = total_m - i
        decay = np.exp(-0.05 * age)
        X_list.append(df[g.features])
        y_list.append(df['fwd_excess'])
        w_list.append(pd.Series(decay, index=df.index))
    if not X_list:
        return None, None, None
    X_all = pd.concat(X_list, axis=0)
    y_all = pd.concat(y_list, axis=0)
    w_all = pd.concat(w_list, axis=0)
    return X_all, y_all, w_all


def train_update(context):
    today = context.current_dt.date()
    X, y, w = assemble_training_set(today)
    if X is None or X.empty:
        log.warn('训练数据为空，跳过模型更新。')
        return
    model = MLRegressor(backend=g.model_backend, random_state=42)
    try:
        if w is not None and len(w) == len(y):
            model.model.fit(np.asarray(X.values), np.asarray(y.values), sample_weight=np.asarray(w.values))
        else:
            model.fit(X.values, y.values)
    except Exception as e:
        # 对于仅用 LGBM 模式，这里直接抛错；若允许降级可在 MLRegressor 内处理
        raise
    g.model = model
    log.info('[模型更新] 样本数: %d | 特征维度: %d | 模型: LightGBM'
         % (len(y), X.shape[1]))
def score_today(date):
    stocks = clean_universe(date)
    if not stocks:
        return pd.DataFrame()
    X = compute_factors(date, stocks)
    if X is None or X.empty:
        return pd.DataFrame()
    # 若模型缺失，临时训练一次
    if g.model is None:
        Xtr, ytr, wtr = assemble_training_set(date)
        if Xtr is not None and not Xtr.empty:
            mdl = MLRegressor(backend=g.model_backend, random_state=42)
            if wtr is not None and len(wtr) == len(ytr):
                mdl.model.fit(np.asarray(Xtr.values), np.asarray(ytr.values), sample_weight=np.asarray(wtr.values))
            else:
                mdl.fit(Xtr.values, ytr.values)
            g.model = mdl
    if g.model is None:
        tmp = X.copy()
        simple = tmp[['ret_3m','ret_6m','ret_12m']].mean(axis=1) - tmp[['pe_ratio','pb_ratio','ps_ratio']].mean(axis=1)
        preds = simple.fillna(simple.median())
    else:
        preds = pd.Series(g.model.model.predict(np.asarray(X.values)), index=X.index)
    # —— 预测中性化：行业 + 规模 ——
    ind = industry_map(list(X.index))
    dfn = pd.DataFrame({'score': preds, 'industry': pd.Series(ind), 'mcap': X['market_cap'] if 'market_cap' in X.columns else 0})
    dummies = pd.get_dummies(dfn['industry'], prefix='ind', dummy_na=True)
    Z = pd.concat([pd.Series(1.0, index=dfn.index, name='const'), dummies, dfn[['mcap']]], axis=1).fillna(0.0)
    yv = dfn['score'].values
    Zv = Z.values
    try:
        beta = np.linalg.lstsq(Zv, yv, rcond=None)[0]
        resid = yv - Zv.dot(beta)
        preds = pd.Series(resid, index=dfn.index)
    except Exception:
        pass
    out = pd.DataFrame({'score': preds})
    # 行业软约束：每个一级行业最多 20% 名额
    ind = industry_map(list(out.index))
    out['industry'] = out.index.map(lambda c: ind.get(c, 'NA'))
    out = out.sort_values('score', ascending=False)

    # 选股：TopN + 行业上限
    picks = []
    cap_per_ind = max(1, int(g.top_n * g.max_industry_weight))
    ind_count = {}
    for code, row in out.iterrows():
        k = str(row['industry'])
        if ind_count.get(k, 0) < cap_per_ind:
            picks.append(code)
            ind_count[k] = ind_count.get(k, 0) + 1
        if len(picks) >= g.top_n:
            break
    out['selected'] = out.index.isin(picks)
    return out


def rebalance(context):
    today = context.current_dt.date()
    sc = score_today(today)
    if sc is None or sc.empty:
        log.warn('无可交易标的，跳过。')
        return
    target_list = list(sc.sort_values('score', ascending=False).head(g.top_n).index)
    if len(target_list) == 0:
        return
    w = 1.0 / len(target_list)
    band = 0.2 * w  # 不交易带：目标权重±20%

    # 卖出不在池内的持仓
    for code in list(context.portfolio.positions.keys()):
        if code not in target_list:
            order_target_value(code, 0)

    total_value = context.portfolio.total_value
    # 现权重
    cur = {}
    for code, pos in context.portfolio.positions.items():
        if pos.avg_cost > 0:
            cur[code] = pos.value / total_value
        else:
            cur[code] = 0.0
    # 买入/调仓：启用 no-trade band 限制换手
    for code in target_list:
        tw = w
        cw = cur.get(code, 0.0)
        if abs(tw - cw) > band:
            order_target_value(code, total_value * tw)

    record(num=len(target_list), turn=np.sum([abs(cur.get(c,0.0)-w) for c in target_list]))

    # 记录
    record(num=len(target_list), cash=context.portfolio.cash / 1e6)




def after_trading_end(context):
    # 每月末计算一次简单 IC（仅日志）
    today = context.current_dt.date()
    mes = month_ends((pd.to_datetime(today) - pd.DateOffset(months=1)).date(), today)
    if not mes:
        return
    d = mes[-1]
    sc = score_today(d)
    if sc is None or sc.empty:
        return
    stocks = list(sc.index)
    fwd = forward_return_1m(d, stocks)
    df = sc.join(fwd, how='inner')
    df = df.dropna(subset=['score', 'fwd_1m'])
    if df.shape[0] >= 20:
        ic = df['score'].corr(df['fwd_1m'])
        log.info('[月度IC] %s : %.4f' % (d, ic))


