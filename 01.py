from jqdata import *
import pandas as pd
import numpy as np

# ==================== 初始化 ====================
def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    run_monthly(rebalance, 1)     # 每月调仓
    context.weight_log = []   # 记录每期（每月）因子权重


    # 换手抑制
    context.target_n   = 30
    context.hold_buffer = 45

    # IC-IR 自适应
    context.ic_window = 12  # 最近12期（12个月）IC滚动窗口
    context.ic_hist = {'EP': [], 'BP': [], 'ROE': [], 'GPM': [], 'MOM': []}
    context.prev_snapshot = None   # 上期因子快照（做完中性化后的截面）
    context.prev_date = None       # 上次调仓日期（用于计算下一期收益）

# ==================== 工具函数 ====================
def _winsorize(s, p1=0.01, p99=0.99):
    s = s.replace([np.inf, -np.inf], np.nan)
    lo, hi = s.quantile(p1), s.quantile(p99)
    return s.clip(lo, hi)

def _zscore(s):
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.notnull().sum() == 0: return pd.Series(0.0, index=s.index)
    std = s.std()
    if std == 0 or np.isnan(std): return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std

def _spearman_ic(f, r):
    # Spearman = Pearson(rank(x), rank(y))
    df = pd.concat([f, r], axis=1, join='inner').dropna()
    if df.shape[0] < 5: return 0.0
    x = df.iloc[:,0].rank()
    y = df.iloc[:,1].rank()
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0 or np.isnan(sx) or np.isnan(sy): return 0.0
    return np.corrcoef(x, y)[0,1]

def _neutralize_to_size(df, col, size_col='log_mktcap'):
    # 截面回归 col ~ 1 + log_mktcap，取残差
    s = df[[col, size_col]].dropna()
    out_col = col + '_sz'
    df[out_col] = 0.0
    if s.empty: return df
    X = np.column_stack([np.ones(len(s)), s[size_col].values])
    y = s[col].values
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        resid = y - X.dot(beta)
        df.loc[s.index, out_col] = resid
    except:
        pass
    return df
    
def _record_weights(context, weights):
    """记录并打印本期因子权重，便于在日志中导出/搜索"""
    # 规范顺序
    names = ['EP','BP','ROE','GPM','MOM']
    rec = {'date': context.current_dt.date()}
    for k in names:
        rec['w_' + k] = float(weights.get(k, 0.0))
    context.weight_log.append(rec)

    # 打印两种格式：人读的 & CSV行，便于你在日志里筛选/复制
    log.info("WEIGHTS %s | EP=%.3f BP=%.3f ROE=%.3f GPM=%.3f MOM=%.3f" % (
        rec['date'], rec['w_EP'], rec['w_BP'], rec['w_ROE'], rec['w_GPM'], rec['w_MOM']
    ))
    print("WEIGHTS_CSV,%s,%.6f,%.6f,%.6f,%.6f,%.6f" % (
        rec['date'], rec['w_EP'], rec['w_BP'], rec['w_ROE'], rec['w_GPM'], rec['w_MOM']
    ))

def _get_sw1_map(codes):
    """
    返回 Series(index=codes, value=SW1行业名)。获取失败的标记为 'UNK'。
    说明：逐只调用 get_industry（每月一次，性能可接受）。
    """
    inds = []
    for c in codes:
        try:
            info = get_industry(c)
            name = info.get('sw_l1', {}).get('industry_name', 'UNK')
            inds.append(name if name else 'UNK')
        except:
            inds.append('UNK')
    return pd.Series(inds, index=codes, name='sw1')

def _industry_neutralize(df, cols, industry_col='sw1'):
    """
    行业中性化：按 SW1 行业做“减行业均值”，再做一次截面标准化。
    返回 *_ind 列（industry-neutral）。
    """
    if industry_col not in df.columns:
        for c in cols: df[c + '_ind'] = df[c]
        return df

    grouped = df.groupby(industry_col)
    for c in cols:
        demean = df[c] - grouped[c].transform('mean')
        df[c + '_ind'] = demean
        df[c + '_ind'] = _zscore(df[c + '_ind'])
    return df

def _forward_return_since(context, date_from, codes):
    """计算自 date_from 到当前回测时点的区间收益（下一期收益）。"""
    if not codes:
        return pd.Series(dtype=float)
    end_date = context.current_dt.date()
    n = len(get_trade_days(start_date=date_from, end_date=end_date))
    if n < 2:
        return pd.Series(dtype=float)
    px = history(n, '1d', 'close', security_list=codes, df=True).dropna(axis=1)
    if px.shape[1] == 0:
        return pd.Series(dtype=float)
    return (px.iloc[-1] / px.iloc[0] - 1.0)


def _compute_ic_and_update(context):
    """用上期快照与本期实际区间收益计算IC，并维护 context.ic_hist。"""
    if context.prev_snapshot is None or context.prev_date is None:
        return
    prev_codes = list(context.prev_snapshot.index)
    fwd = _forward_return_since(context, context.prev_date, prev_codes)

    if fwd.empty: return
    snap = context.prev_snapshot.join(fwd.rename('RET'), how='inner').dropna()
    if snap.empty: return

    # 逐因子 Spearman IC（使用行业&市值中性化后的最终列）
    for name, col in [('EP','EP_fin'), ('BP','BP_fin'), ('ROE','ROE_fin'),
                      ('GPM','GPM_fin'), ('MOM','MOM_fin')]:
        ic = _spearman_ic(snap[col], snap['RET'])
        arr = context.ic_hist[name]
        arr.append(ic)
        if len(arr) > context.ic_window:
            context.ic_hist[name] = arr[-context.ic_window:]

def _derive_weights(context):
    """
    根据最近滚动IC构建 IC-IR 权重：
    - IC-IR = mean(IC) / std(IC)
    - 仅保留正值，负值记为 0（因子失效期降权）
    - 若全部为 0 或样本不足/异常 → 回退等权
    """
    factor_names = ['EP','BP','ROE','GPM','MOM']
    scores = {}

    # 计算每个因子的 IC-IR
    for name in factor_names:
        arr = np.array(context.ic_hist[name], dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size < 3:
            scores[name] = 0.0
            continue
        mu = float(np.nanmean(arr))
        sd = float(np.nanstd(arr))
        ic_ir = (mu / sd) if sd > 0 else 0.0
        scores[name] = max(ic_ir, 0.0)   # 只保留正值

    # —— 关键稳健化：显式转成 float 列表，再求和 —— #
    vals = [float(v) for v in scores.values()]
    total = float(np.nansum(vals))  # 避免任何 sum 污染/类型问题

    # 冷启动 / 全零 / 非数 → 等权
    if (not np.isfinite(total)) or total <= 0:
        w_eq = 1.0 / len(factor_names)
        return {name: w_eq for name in factor_names}

    # 归一化（再次防御非数）
    weights = {}
    for name in factor_names:
        v = float(scores.get(name, 0.0))
        w = v / total if total > 0 else 0.0
        if not np.isfinite(w):
            w = 0.0
        weights[name] = w

    # 归一化后如果全为0，再回退等权
    if sum(weights.values()) == 0:
        w_eq = 1.0 / len(factor_names)
        return {name: w_eq for name in factor_names}

    return weights


# ==================== 因子计算（含市值&行业中性化） ====================
def _calc_current_factors():
    """
    返回因子表（index=code），包含：
    - 原始：EP/BP/ROE/GPM/MOM、log_mktcap、sw1
    - 市值中性化：*_sz
    - 行业中性化后最终用于打分与IC：*_fin
    """
    pool = get_index_stocks('000906.XSHG')   # 中证800
    if not pool: return pd.DataFrame()

    q = query(
        valuation.code, valuation.pe_ratio, valuation.pb_ratio, valuation.market_cap,
        indicator.roe, indicator.gross_profit_margin
    ).filter(valuation.code.in_(pool))
    df = get_fundamentals(q)
    if df.empty: return df

    # 交易性过滤：ST/停牌
    cur = get_current_data()
    df = df[~df['code'].map(lambda x: cur[x].is_st)]
    df = df[df['code'].map(lambda x: not cur[x].paused)]
    if df.empty: return df
    df = df.set_index('code')

    # 流动性过滤：近60日日均成交额 > 3,000万
    codes = df.index.tolist()
    money = history(60, '1d', 'money', security_list=codes, df=True).dropna(axis=1)
    if money.empty: return pd.DataFrame()
    df = df.join(money.mean().rename('avg_amt'), how='inner')
    df = df[df['avg_amt'] > 3e7]
    if df.empty: return df

    # 动量：12-1 = (t-21)/(t-252)-1
    px = history(252, '1d', 'close', security_list=df.index.tolist(), df=True).dropna(axis=1)
    if px.shape[1] == 0: return pd.DataFrame()
    mom = (px.iloc[-21] / px.iloc[0] - 1.0).rename('MOM')
    df = df.join(mom, how='inner')
    if df.empty: return df

    # 构造基础因子（越大越好）
    df['EP']  = 1.0 / df['pe_ratio'].replace(0, np.nan)
    df['BP']  = 1.0 / df['pb_ratio'].replace(0, np.nan)
    df['ROE'] = df['roe']
    df['GPM'] = df['gross_profit_margin']
    df['log_mktcap'] = np.log(df['market_cap'].replace(0, np.nan))

    # 去极值 & 标准化
    for c in ['EP','BP','ROE','GPM','MOM']:
        df[c] = _zscore(_winsorize(df[c]))

    # 市值中性化（得到 *_sz）
    for c in ['EP','BP','ROE','GPM','MOM']:
        df = _neutralize_to_size(df, c, 'log_mktcap')
        df[c + '_sz'] = _zscore(df[c + '_sz'])

    # 行业标识（申万一级）
    df['sw1'] = _get_sw1_map(df.index.tolist())

    # 行业中性化（对 *_sz 做减行业均值 + 再标准化，得到 *_ind）
    df = _industry_neutralize(df, [c + '_sz' for c in ['EP','BP','ROE','GPM','MOM']], industry_col='sw1')

    # 最终用于打分与IC的列（统一命名 *_fin）
    for base in ['EP','BP','ROE','GPM','MOM']:
        df[base + '_fin'] = df[base + '_sz_ind'].copy() if (base + '_sz_ind') in df.columns else df[base + '_sz'].copy()
        df[base + '_fin'] = _zscore(df[base + '_fin'])  # 最后一轮标准化以确保可比

    return df

# ==================== 主流程 ====================
def rebalance(context):
    # 1) 用上期快照 + 本期收益更新历史IC（用于本期权重）
    _compute_ic_and_update(context)

    # 2) 计算本期因子（含市值&行业中性化）
    df = _calc_current_factors()
    if df.empty: return

    # 3) 生成动态权重（IC-IR）
    w = _derive_weights(context)
    
    # 记录权重（用于面试展示 & 后续可视化）
    _record_weights(context, w)


    # 4) 合成得分（用 * _fin 列）
    df['score'] = ( w['EP']  * df['EP_fin']
                  + w['BP']  * df['BP_fin']
                  + w['ROE'] * df['ROE_fin']
                  + w['GPM'] * df['GPM_fin']
                  + w['MOM'] * df['MOM_fin'] )

    # 5) 排序 + 换手抑制（缓冲带）
    ranked = df.sort_values('score', ascending=False)
    topN = ranked.index[:context.target_n].tolist()
    buffer_set = set(ranked.index[:context.hold_buffer].tolist())
    current = set(context.portfolio.positions.keys())
    keep = [s for s in current if s in buffer_set]
    need = context.target_n - len(keep)
    add  = [s for s in ranked.index if s not in keep][:need]
    target = keep + add

    # 6) 调仓：先清仓非目标，再按等权市值建仓
    for s in list(current):
        if s not in target:
            order_target_value(s, 0)

    if len(target) == 0:
        # 仍需保存快照（避免下一期 IC 丢失）
        context.prev_snapshot = ranked[['EP_fin','BP_fin','ROE_fin','GPM_fin','MOM_fin']].copy()
        context.prev_date = context.current_dt.date()
        return

    total_value = context.portfolio.total_value
    tgt_val = total_value / len(target)
    for s in target:
        order_target_value(s, tgt_val)

    # 7) 保存本期快照（用于下期计算IC）
    context.prev_snapshot = ranked[['EP_fin','BP_fin','ROE_fin','GPM_fin','MOM_fin']].copy()
    context.prev_date = context.current_dt.date()