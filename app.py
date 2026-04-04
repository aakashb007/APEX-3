import streamlit as st
import pandas as pd
import numpy as np
import requests, time, csv, os, json
from datetime import datetime, timezone, timedelta

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="APEXAI // G/L Scanner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
GL_PERF_FILE   = "gl_performance.csv"
JOURNAL_FILE   = "trade_journal.csv"
SETTINGS_FILE  = "apex_settings.json"

DEFAULT_SETTINGS = {
    "gl_enabled":        True,
    "gl_interval":       1,
    "gl_top_n":          20,
    "gl_min_gain_pct":   3.0,
    "gl_min_loss_pct":   1.5,
    "gl_pullback_min":   1.5,
    "gl_pullback_max":   8.0,
    "gl_min_rr":         1.5,
    "gl_vol_expansion":  1.5,
    "gl_rsi_ob":         75,
    "gl_rsi_os":         30,
    "gl_alert_pullback": True,
    "gl_alert_breakout": True,
    "gl_alert_bounce":   True,
    "gl_alert_breakdown":True,
    "gl_alert_pregainer":True,
    "gl_alert_preloser": True,
    "discord_webhook":   "https://discord.com/api/webhooks/1476606856599179265/74wKbIJEXNJ9h10Ab0Q9Vp7ZmeJ52XY18CP3lKxg3eR1BbpZSdX65IT8hbZjpEIXSqEg",
    "groq_key":          "",
    "symbol_blacklist":  "PAXG,WBTC,WETH,STETH,FDUSD,USDTUSDT",
}

# ─── SESSION STATE ───────────────────────────────────────────────────────────
_SS = {
    'gl_results': [], 'gl_last_scan': '—', 'gl_last_ts': 0,
    'gl_history': [], 'gl_alerted': set(),
    'gl_active_done': False, 'gl_pre_pending': False,
    '_binance_ok': None,
}
for k, v in _SS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── SETTINGS LOAD/SAVE ──────────────────────────────────────────────────────
@st.cache_data(ttl=10, show_spinner=False)
def load_settings():
    s = DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                s.update(json.load(f))
        except:
            pass
    return s

def save_settings(s):
    with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(s, f, indent=2)
    load_settings.clear()


# ─── EXCHANGE / OHLCV ────────────────────────────────────────────────────────
def fetch_ohlcv_smart(symbol, timeframe, limit):
    import ccxt as _fx
    _bnb_sym = symbol.replace(':USDT', '').replace(':USD', '')

    if st.session_state.get('_binance_ok', None) != False:
        try:
            _ex = _fx.binanceusdm({'enableRateLimit': True, 'timeout': 3000})
            raw = _ex.fetch_ohlcv(_bnb_sym, timeframe, limit=limit)
            if raw and len(raw) > 10:
                st.session_state['_binance_ok'] = True
                return pd.DataFrame(raw, columns=['ts','open','high','low','close','volume']).astype(float)
        except:
            st.session_state['_binance_ok'] = False

    try:
        _ex = _fx.gateio({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        raw = _ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        if raw and len(raw) > 10:
            return pd.DataFrame(raw, columns=['ts','open','high','low','close','volume']).astype(float)
    except:
        pass

    try:
        _ex = _fx.mexc({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        raw = _ex.fetch_ohlcv(symbol, timeframe, limit=limit)
        if raw and len(raw) > 10:
            return pd.DataFrame(raw, columns=['ts','open','high','low','close','volume']).astype(float)
    except:
        pass

    return pd.DataFrame()


def _is_crypto_symbol(sym):
    _EXCLUDE = {
        'XAU','XAG','XAUT','GOLD','SILVER','UKOIL','USOIL','BRENT','WTI',
        'NATGAS','COPPER','PLATINUM','PALLADIUM','WHEAT','CORN','SOYBEAN',
        'EUR','GBP','JPY','AUD','CAD','CHF','NZD','CNY','CNH',
        'KRW','SGD','HKD','MXN','RUB','TRY','ZAR','SEK','NOK',
        'SPX','SPY','NDX','QQQ','DJI','NIFTY','FTSE','DAX','NIKK',
        'HSI','CSI','VIX','INDEX',
        'HALF','USDTUSDT','USDCUSDT','USDTUSD','FDUSD',
    }
    _EXCLUDE_PREFIXES = ('XAU','XAG','GOLD','SILVER','OIL','EUR','GBP','JPY')
    try:
        base = sym.split('/')[0].upper().strip()
        if base in _EXCLUDE: return False
        if any(base.startswith(p) for p in _EXCLUDE_PREFIXES): return False
        return True
    except:
        return True


# ─── TIME HELPERS ────────────────────────────────────────────────────────────
def _dual_time():
    now = datetime.now(timezone.utc)
    utc_str = now.strftime('%Y-%m-%d %H:%M:%S UTC')
    pkt = now + timedelta(hours=5)
    pkt_str = pkt.strftime('%Y-%m-%d %H:%M:%S PKT')
    return utc_str, pkt_str


# ─── TECHNICAL INDICATORS ────────────────────────────────────────────────────
def _dst_ema(s, p):
    return s.ewm(span=p, adjust=False).mean()

def _dst_rsi(s, p=14):
    d    = s.diff()
    gain = d.clip(lower=0).ewm(com=p-1, min_periods=p).mean()
    loss = (-d.clip(upper=0)).ewm(com=p-1, min_periods=p).mean()
    return 100 - (100 / (1 + gain / loss))

def _dst_atr(h, l, c, p=14):
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, min_periods=p).mean()

def _dst_supertrend(h, l, c, mult=3.0, period=10):
    atr_v     = _dst_atr(h, l, c, period)
    hl2       = (h + l) / 2
    upper_raw = hl2 + mult * atr_v
    lower_raw = hl2 - mult * atr_v
    upper     = upper_raw.copy()
    lower     = lower_raw.copy()
    direction = pd.Series(1.0, index=c.index)
    for i in range(1, len(c)):
        lower.iloc[i] = lower_raw.iloc[i] if (lower_raw.iloc[i] > lower.iloc[i-1] or c.iloc[i-1] < lower.iloc[i-1]) else lower.iloc[i-1]
        upper.iloc[i] = upper_raw.iloc[i] if (upper_raw.iloc[i] < upper.iloc[i-1] or c.iloc[i-1] > upper.iloc[i-1]) else upper.iloc[i-1]
        if   c.iloc[i] > upper.iloc[i-1]: direction.iloc[i] =  1
        elif c.iloc[i] < lower.iloc[i-1]: direction.iloc[i] = -1
        else:                              direction.iloc[i] = direction.iloc[i-1]
    st_line = pd.Series(np.where(direction == 1, lower, upper), index=c.index)
    return st_line, direction


# ─── GL CORE CALCULATIONS ────────────────────────────────────────────────────
def _gl_pct_change(df_4h):
    try:
        if len(df_4h) < 2: return 0.0
        open_4h   = float(df_4h['close'].iloc[-4]) if len(df_4h) >= 4 else float(df_4h['close'].iloc[0])
        close_now = float(df_4h['close'].iloc[-1])
        return (close_now - open_4h) / open_4h * 100
    except:
        return 0.0

def _gl_pct_change_1h(df_5m):
    try:
        if len(df_5m) < 12: return 0.0
        return (float(df_5m['close'].iloc[-1]) - float(df_5m['close'].iloc[-12])) / float(df_5m['close'].iloc[-12]) * 100
    except:
        return 0.0

def _gl_dynamic_tpsl(df_5m, df_4h, close_now, direction='LONG'):
    try:
        h5 = df_5m['high']; l5 = df_5m['low']
        h4 = df_4h['high']; l4 = df_4h['low']
        swing_lows  = [float(l5.iloc[i]) for i in range(-30, -1) if l5.iloc[i] < l5.iloc[i-1] and l5.iloc[i] < l5.iloc[i+1]]
        swing_highs = [float(h5.iloc[i]) for i in range(-30, -1) if h5.iloc[i] > h5.iloc[i-1] and h5.iloc[i] > h5.iloc[i+1]]

        if direction == 'LONG':
            lows_below  = [x for x in swing_lows  if x < close_now * 0.999]
            highs_above = [x for x in swing_highs if x > close_now * 1.001]
            sl  = max(lows_below)  if lows_below  else close_now * 0.98
            tp3 = float(h4.iloc[-8:].max())
        else:
            highs_above = [x for x in swing_highs if x > close_now * 1.001]
            lows_below  = [x for x in swing_lows  if x < close_now * 0.999]
            sl  = min(highs_above) if highs_above else close_now * 1.02
            tp3 = float(l4.iloc[-8:].min())

        risk = abs(close_now - sl)
        if direction == 'LONG':
            tp1_r = close_now + risk * 1.0
            tp2_r = close_now + risk * 2.0
            tp3_r = max(tp3, close_now + risk * 3.0)
        else:
            tp1_r = close_now - risk * 1.0
            tp2_r = close_now - risk * 2.0
            tp3_r = min(tp3, close_now - risk * 3.0)
        rr = round(abs(tp2_r - close_now) / max(risk, 0.000001), 2)
        return {'sl': round(sl,8), 'tp': round(tp2_r,8),
                'tp1': round(tp1_r,8), 'tp2': round(tp2_r,8), 'tp3': round(tp3_r,8), 'rr': rr}
    except:
        return None


# ─── SIGNAL DETECTORS ────────────────────────────────────────────────────────
def _gl_check_gainer_pullback(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        chg_1h = _gl_pct_change_1h(df_5m)
        if chg_4h < float(s.get('gl_min_gain_pct', 3.0)): return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now    = float(c.iloc[-1])
        recent_high  = float(h.iloc[-24:].max())
        pullback_pct = (recent_high - close_now) / recent_high * 100
        pb_min = float(s.get('gl_pullback_min', 1.5))
        pb_max = float(s.get('gl_pullback_max', 8.0))
        if not (pb_min <= pullback_pct <= pb_max): return None
        rsi_now = float(_dst_rsi(c).iloc[-1])
        if not (45 < rsi_now < float(s.get('gl_rsi_ob', 75))): return None
        vol_ma  = v.rolling(20).mean()
        vol_now = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        vol_ratio = vol_now / volma
        if vol_ratio < float(s.get('gl_vol_expansion', 1.3)): return None
        if not (float(v.iloc[-3:].mean()) < float(v.iloc[-8:-3].mean())): return None
        _, st_dir = _dst_supertrend(h, l, c, 3.0, 10)
        if float(st_dir.iloc[-1]) != 1: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'LONG')
        if not tpsl or tpsl['rr'] < float(s.get('gl_min_rr', 1.5)): return None
        return {
            'type': 'GAINER_PULLBACK', 'direction': 'LONG', 'symbol': symbol,
            'close': close_now, 'sl': tpsl['sl'], 'tp': tpsl['tp'],
            'tp1': tpsl['tp1'], 'tp2': tpsl['tp2'], 'tp3': tpsl['tp3'],
            'rr': tpsl['rr'], 'rsi': round(rsi_now, 1),
            'chg_4h': round(chg_4h, 2), 'chg_1h': round(chg_1h, 2),
            'pullback_pct': round(pullback_pct, 2), 'recent_high': round(recent_high, 8),
            'vol_ratio': round(vol_ratio, 2),
            'emoji': '🟢', 'label': 'GAINER PULLBACK',
            'reasons': [f'4H up {round(chg_4h,1)}%', f'Pulled back {round(pullback_pct,1)}%',
                        f'RSI {round(rsi_now,1)}', f'Vol {round(vol_ratio,1)}x', 'ST BULLISH', 'Volume declining on pullback']
        }
    except:
        return None

def _gl_check_gainer_breakout(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        if chg_4h < float(s.get('gl_min_gain_pct', 3.0)): return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now   = float(c.iloc[-1])
        recent_high = float(h.iloc[-24:].max())
        if not (close_now >= recent_high * 0.998): return None
        rsi_now = float(_dst_rsi(c).iloc[-1])
        if not (55 < rsi_now < 80): return None
        vol_ma  = v.rolling(20).mean()
        vol_now = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        if not (vol_now > volma * float(s.get('gl_vol_expansion', 1.5))): return None
        _, st_dir = _dst_supertrend(h, l, c, 3.0, 10)
        if float(st_dir.iloc[-1]) != 1: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'LONG')
        if not tpsl or tpsl['rr'] < float(s.get('gl_min_rr', 1.5)): return None
        return {
            'type': 'GAINER_BREAKOUT', 'direction': 'LONG', 'symbol': symbol,
            'close': close_now, 'sl': tpsl['sl'], 'tp': tpsl['tp'],
            'tp1': tpsl['tp1'], 'tp2': tpsl['tp2'], 'tp3': tpsl['tp3'],
            'rr': tpsl['rr'], 'rsi': round(rsi_now, 1),
            'chg_4h': round(chg_4h, 2), 'chg_1h': round(_gl_pct_change_1h(df_5m), 2),
            'vol_ratio': round(vol_now / volma, 2),
            'emoji': '🚀', 'label': 'GAINER BREAKOUT',
            'reasons': [f'4H up {round(chg_4h,1)}%', 'Breaking to new high',
                        f'RSI {round(rsi_now,1)}', f'Vol {round(vol_now/volma,1)}x', 'ST BULLISH']
        }
    except:
        return None

def _gl_check_loser_bounce(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        if chg_4h > -float(s.get('gl_min_loss_pct', 3.0)): return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now  = float(c.iloc[-1])
        recent_low = float(l.iloc[-24:].min())
        bounce_pct = (close_now - recent_low) / recent_low * 100
        if bounce_pct < 0.5: return None
        rsi_now = float(_dst_rsi(c).iloc[-1])
        if not (15 < rsi_now < float(s.get('gl_rsi_os', 40)) + 10): return None
        vol_ma  = v.rolling(20).mean()
        vol_now = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        if vol_now < volma * float(s.get('gl_vol_expansion', 1.3)): return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'LONG')
        if not tpsl or tpsl['rr'] < float(s.get('gl_min_rr', 1.5)): return None
        return {
            'type': 'LOSER_BOUNCE', 'direction': 'LONG', 'symbol': symbol,
            'close': close_now, 'sl': tpsl['sl'], 'tp': tpsl['tp'],
            'tp1': tpsl['tp1'], 'tp2': tpsl['tp2'], 'tp3': tpsl['tp3'],
            'rr': tpsl['rr'], 'rsi': round(rsi_now, 1),
            'chg_4h': round(chg_4h, 2), 'chg_1h': round(_gl_pct_change_1h(df_5m), 2),
            'vol_ratio': round(vol_now / volma, 2),
            'emoji': '🔄', 'label': 'LOSER BOUNCE',
            'reasons': [f'4H down {round(chg_4h,1)}%', f'Bouncing from low +{round(bounce_pct,1)}%',
                        f'RSI {round(rsi_now,1)} oversold', f'Vol {round(vol_now/volma,1)}x']
        }
    except:
        return None

def _gl_check_loser_breakdown(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        if chg_4h > -float(s.get('gl_min_loss_pct', 3.0)): return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now  = float(c.iloc[-1])
        recent_low = float(l.iloc[-24:].min())
        if not (close_now <= recent_low * 1.002): return None
        rsi_now = float(_dst_rsi(c).iloc[-1])
        if not (10 < rsi_now < 40): return None
        vol_ma  = v.rolling(20).mean()
        vol_now = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        if not (vol_now > volma * float(s.get('gl_vol_expansion', 1.5))): return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'SHORT')
        if not tpsl or tpsl['rr'] < float(s.get('gl_min_rr', 1.5)): return None
        return {
            'type': 'LOSER_BREAKDOWN', 'direction': 'SHORT', 'symbol': symbol,
            'close': close_now, 'sl': tpsl['sl'], 'tp': tpsl['tp'],
            'tp1': tpsl['tp1'], 'tp2': tpsl['tp2'], 'tp3': tpsl['tp3'],
            'rr': tpsl['rr'], 'rsi': round(rsi_now, 1),
            'chg_4h': round(chg_4h, 2), 'chg_1h': round(_gl_pct_change_1h(df_5m), 2),
            'vol_ratio': round(vol_now / volma, 2),
            'emoji': '📉', 'label': 'LOSER BREAKDOWN',
            'reasons': [f'4H down {round(chg_4h,1)}%', 'Breaking to new low',
                        f'RSI {round(rsi_now,1)}', f'Vol {round(vol_now/volma,1)}x']
        }
    except:
        return None

def _gl_check_pregainer(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now = float(c.iloc[-1])
        rsi_now   = float(_dst_rsi(c).iloc[-1])
        vol_ma    = v.rolling(20).mean()
        vol_now   = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        vol_ratio = vol_now / volma
        if vol_ratio < 1.3: return None
        if not (35 < rsi_now < 65): return None
        recent_high = float(h.iloc[-24:].max())
        near_breakout = (recent_high - close_now) / recent_high * 100 < 2.0
        if not near_breakout: return None
        _, st_dir = _dst_supertrend(h, l, c, 3.0, 10)
        if float(st_dir.iloc[-1]) != 1: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'LONG')
        return {
            'type': 'PRE_GAINER', 'direction': 'WATCH', 'symbol': symbol,
            'close': close_now,
            'sl': tpsl['sl'] if tpsl else 0, 'tp': tpsl['tp'] if tpsl else 0,
            'tp1': tpsl['tp1'] if tpsl else 0, 'tp2': tpsl['tp2'] if tpsl else 0,
            'tp3': tpsl['tp3'] if tpsl else 0,
            'rr': tpsl['rr'] if tpsl else 0,
            'rsi': round(rsi_now, 1), 'chg_4h': round(chg_4h, 2),
            'chg_1h': round(_gl_pct_change_1h(df_5m), 2), 'vol_ratio': round(vol_ratio, 2),
            'emoji': '👀', 'label': 'PRE-GAINER WATCH',
            'reasons': [f'Flat 4H ({round(chg_4h,1)}%)', f'RSI {round(rsi_now,1)} neutral',
                        f'Vol {round(vol_ratio,1)}x spike', 'Near breakout level', 'ST BULLISH']
        }
    except:
        return None

def _gl_check_preloser(df_5m, df_4h, s, symbol):
    try:
        chg_4h = _gl_pct_change(df_4h)
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now = float(c.iloc[-1])
        rsi_now   = float(_dst_rsi(c).iloc[-1])
        vol_ma    = v.rolling(20).mean()
        vol_now   = float(v.iloc[-1]); volma = float(vol_ma.iloc[-1])
        if volma <= 0: return None
        vol_ratio = vol_now / volma
        if vol_ratio < 1.3: return None
        if not (35 < rsi_now < 65): return None
        recent_low = float(l.iloc[-24:].min())
        near_breakdown = (close_now - recent_low) / close_now * 100 < 2.0
        if not near_breakdown: return None
        _, st_dir = _dst_supertrend(h, l, c, 3.0, 10)
        if float(st_dir.iloc[-1]) != -1: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'SHORT')
        return {
            'type': 'PRE_LOSER', 'direction': 'WATCH', 'symbol': symbol,
            'close': close_now,
            'sl': tpsl['sl'] if tpsl else 0, 'tp': tpsl['tp'] if tpsl else 0,
            'tp1': tpsl['tp1'] if tpsl else 0, 'tp2': tpsl['tp2'] if tpsl else 0,
            'tp3': tpsl['tp3'] if tpsl else 0,
            'rr': tpsl['rr'] if tpsl else 0,
            'rsi': round(rsi_now, 1), 'chg_4h': round(chg_4h, 2),
            'chg_1h': round(_gl_pct_change_1h(df_5m), 2), 'vol_ratio': round(vol_ratio, 2),
            'emoji': '⚠️', 'label': 'PRE-LOSER WATCH',
            'reasons': [f'Flat 4H ({round(chg_4h,1)}%)', f'RSI {round(rsi_now,1)} neutral',
                        f'Vol {round(vol_ratio,1)}x spike', 'Near support break', 'ST BEARISH']
        }
    except:
        return None

def _gl_check_pregainer_star(df_5m, df_4h, s, symbol):
    try:
        base = _gl_check_pregainer(df_5m, df_4h, s, symbol)
        if not base: return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now = float(c.iloc[-1])
        # Extra conditions for ⭐ star
        rsi_now   = float(_dst_rsi(c).iloc[-1])
        vol_now   = float(v.iloc[-1])
        volma     = float(v.rolling(20).mean().iloc[-1])
        vol_ratio = vol_now / volma if volma > 0 else 0
        # 4H confirmed bullish via chg
        chg_4h = _gl_pct_change(df_4h)
        if not (0.5 <= chg_4h <= 3.0): return None
        if vol_ratio < 1.8: return None
        if not (40 < rsi_now < 60): return None
        recent_low = float(l.iloc[-24:].min())
        if (close_now - recent_low) / recent_low * 100 < 1.0: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'LONG')
        return {
            'type': 'PRE_GAINER_STAR', 'direction': 'WATCH', 'symbol': symbol,
            'close': close_now,
            'sl': tpsl['sl'] if tpsl else 0, 'tp': tpsl['tp'] if tpsl else 0,
            'tp1': tpsl['tp1'] if tpsl else 0, 'tp2': tpsl['tp2'] if tpsl else 0,
            'tp3': tpsl['tp3'] if tpsl else 0,
            'rr': tpsl['rr'] if tpsl else 0,
            'rsi': round(rsi_now, 1), 'chg_4h': round(chg_4h, 2),
            'chg_1h': round(_gl_pct_change_1h(df_5m), 2), 'vol_ratio': round(vol_ratio, 2),
            'emoji': '⭐', 'label': '⭐ PRE-GAINER STAR',
            'reasons': base['reasons'] + [f'Vol {round(vol_ratio,1)}x HIGH', 'Tight RSI range', '4H up slightly']
        }
    except:
        return None

def _gl_check_preloser_star(df_5m, df_4h, s, symbol):
    try:
        base = _gl_check_preloser(df_5m, df_4h, s, symbol)
        if not base: return None
        c = df_5m['close']; h = df_5m['high']; l = df_5m['low']; v = df_5m['volume']
        close_now = float(c.iloc[-1])
        rsi_now   = float(_dst_rsi(c).iloc[-1])
        vol_now   = float(v.iloc[-1])
        volma     = float(v.rolling(20).mean().iloc[-1])
        vol_ratio = vol_now / volma if volma > 0 else 0
        chg_4h    = _gl_pct_change(df_4h)
        if not (-3.0 <= chg_4h <= -0.5): return None
        if vol_ratio < 1.8: return None
        if not (40 < rsi_now < 60): return None
        recent_high = float(h.iloc[-24:].max())
        if (recent_high - close_now) / recent_high * 100 < 1.0: return None
        tpsl = _gl_dynamic_tpsl(df_5m, df_4h, close_now, 'SHORT')
        return {
            'type': 'PRE_LOSER_STAR', 'direction': 'WATCH', 'symbol': symbol,
            'close': close_now,
            'sl': tpsl['sl'] if tpsl else 0, 'tp': tpsl['tp'] if tpsl else 0,
            'tp1': tpsl['tp1'] if tpsl else 0, 'tp2': tpsl['tp2'] if tpsl else 0,
            'tp3': tpsl['tp3'] if tpsl else 0,
            'rr': tpsl['rr'] if tpsl else 0,
            'rsi': round(rsi_now, 1), 'chg_4h': round(chg_4h, 2),
            'chg_1h': round(_gl_pct_change_1h(df_5m), 2), 'vol_ratio': round(vol_ratio, 2),
            'emoji': '⭐', 'label': '⭐ PRE-LOSER STAR',
            'reasons': base['reasons'] + [f'Vol {round(vol_ratio,1)}x HIGH', 'Tight RSI range', '4H down slightly']
        }
    except:
        return None


# ─── MAIN SCAN FUNCTION ──────────────────────────────────────────────────────
def run_gl_scan(s, status_placeholder=None, instant_webhook=None, instant_alerted=None):
    """
    Fetches live tickers from GATE, ranks by 24H % change,
    picks top N gainers + top N losers, then runs 6 signal checks.
    """
    if not s.get('gl_enabled', True): return []
    results = []
    top_n   = int(s.get('gl_top_n', 20))

    try:
        import ccxt as ccxt_mod
        ex = ccxt_mod.gate({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        tickers = ex.fetch_tickers()
    except Exception as e:
        print(f"[GL] Ticker fetch failed: {e}")
        return []

    blacklist = {b.strip().upper() for b in s.get('symbol_blacklist', '').split(',') if b.strip()}

    ranked = []
    for sym, t in tickers.items():
        if not sym.endswith('/USDT:USDT'): continue
        if not _is_crypto_symbol(sym): continue
        base = sym.split('/')[0].upper()
        if base in blacklist: continue
        pct = t.get('percentage') or t.get('change') or 0
        vol = t.get('quoteVolume') or 0
        if vol < 500000: continue
        try:
            ranked.append({'symbol': sym, 'pct': float(pct), 'vol': float(vol)})
        except:
            continue

    if not ranked:
        return []

    gainers    = sorted([r for r in ranked if r['pct'] > 0],    key=lambda x: x['pct'],  reverse=True)[:top_n]
    losers     = sorted([r for r in ranked if r['pct'] < 0],    key=lambda x: x['pct'])[:top_n]
    flat_coins = sorted([r for r in ranked if -2.0 <= r['pct'] <= 2.0], key=lambda x: x['vol'], reverse=True)[:100]
    active_list = gainers + losers
    total_coins = len(active_list) + len(flat_coins)

    if status_placeholder:
        status_placeholder.markdown(
            f'<div style="font-family:monospace;font-size:.65rem;color:#0284c7;">'
            f'🔍 Found {len(gainers)}G {len(losers)}L {len(flat_coins)} flat — {total_coins} total</div>',
            unsafe_allow_html=True)

    active_checks = []
    if s.get('gl_alert_pullback', True):  active_checks.append(_gl_check_gainer_pullback)
    if s.get('gl_alert_breakout', True):  active_checks.append(_gl_check_gainer_breakout)
    if s.get('gl_alert_bounce', True):    active_checks.append(_gl_check_loser_bounce)
    if s.get('gl_alert_breakdown', True): active_checks.append(_gl_check_loser_breakdown)

    pre_checks = []
    if s.get('gl_alert_pregainer', True): pre_checks.append(_gl_check_pregainer)
    if s.get('gl_alert_preloser',  True): pre_checks.append(_gl_check_preloser)
    pre_checks.append(_gl_check_pregainer_star)
    pre_checks.append(_gl_check_preloser_star)

    seen_syms = set()
    checked   = 0

    # Pass 1 — active signals on gainers/losers
    for item in active_list:
        if not active_checks: break
        try:
            symbol = item['symbol']
            if not symbol: continue
            checked += 1
            if status_placeholder and checked % 3 == 0:
                status_placeholder.markdown(
                    f'<div style="font-family:monospace;font-size:.65rem;color:#0284c7;">'
                    f'⚡ {checked}/{total_coins} — {symbol.replace("/USDT:USDT","")} | signals: {len(results)}</div>',
                    unsafe_allow_html=True)

            df_4h = fetch_ohlcv_smart(symbol, '4h', 50)
            df_5m = fetch_ohlcv_smart(symbol, '5m', 250)
            if df_4h.empty or len(df_4h) < 10: continue
            if df_5m.empty or len(df_5m) < 100: continue
            df_4h = df_4h.astype(float, errors='ignore')
            df_5m = df_5m.astype(float, errors='ignore')
            chg_4h = _gl_pct_change(df_4h)
            chg_1h = _gl_pct_change_1h(df_5m)
            utc_str, pkt_str = _dual_time()

            for check_fn in active_checks:
                sig = check_fn(df_5m, df_4h, s, symbol)
                if sig:
                    key = f"{symbol}_{sig['type']}"
                    if key in seen_syms: continue
                    seen_syms.add(key)
                    sig['exchange']      = 'GATE'
                    sig['scan_time']     = utc_str
                    sig['scan_time_pkt'] = pkt_str
                    sig['chg_4h']        = round(chg_4h, 2)
                    sig['chg_1h']        = round(chg_1h, 2)
                    results.append(sig)
                    if instant_webhook:
                        try:
                            _sid = f"{symbol}_{sig['type']}_{datetime.now().strftime('%Y%m%d%H')}"
                            if instant_alerted is None or _sid not in instant_alerted:
                                send_gl_discord_alert(instant_webhook, sig)
                                if instant_alerted is not None: instant_alerted.add(_sid)
                        except: pass
        except:
            continue

    results.sort(key=lambda x: abs(x.get('chg_4h', 0)), reverse=True)

    # Pass 2 — pre-signals on flat coins
    if pre_checks:
        if status_placeholder:
            status_placeholder.markdown(
                '<div style="font-family:monospace;font-size:.65rem;color:#7c3aed;">👀 Scanning flat coins for Pre-Gainer/Pre-Loser...</div>',
                unsafe_allow_html=True)
        for item in flat_coins:
            try:
                symbol = item['symbol']
                if not symbol: continue
                df_4h = fetch_ohlcv_smart(symbol, '4h', 50)
                df_5m = fetch_ohlcv_smart(symbol, '5m', 250)
                if df_4h.empty or len(df_4h) < 10: continue
                if df_5m.empty or len(df_5m) < 100: continue
                df_4h = df_4h.astype(float, errors='ignore')
                df_5m = df_5m.astype(float, errors='ignore')
                chg_4h = _gl_pct_change(df_4h)
                chg_1h = _gl_pct_change_1h(df_5m)
                utc_str, pkt_str = _dual_time()
                for check_fn in pre_checks:
                    sig = check_fn(df_5m, df_4h, s, symbol)
                    if sig:
                        key = f"{symbol}_{sig['type']}"
                        if key in seen_syms: continue
                        seen_syms.add(key)
                        sig['exchange']      = 'GATE'
                        sig['scan_time']     = utc_str
                        sig['scan_time_pkt'] = pkt_str
                        sig['chg_4h']        = round(chg_4h, 2)
                        sig['chg_1h']        = round(chg_1h, 2)
                        results.append(sig)
                        if instant_webhook:
                            try:
                                _sid = f"{symbol}_{sig['type']}_{datetime.now().strftime('%Y%m%d%H')}"
                                if instant_alerted is None or _sid not in instant_alerted:
                                    send_gl_discord_alert(instant_webhook, sig)
                                    if instant_alerted is not None: instant_alerted.add(_sid)
                            except: pass
            except:
                continue

    results.sort(key=lambda x: abs(x.get('chg_4h', 0)), reverse=True)
    return results


# ─── DISCORD ALERT ───────────────────────────────────────────────────────────
def send_gl_discord_alert(webhook_url, sig):
    try:
        type_colors = {
            'GAINER_PULLBACK': 0x059669, 'GAINER_BREAKOUT': 0x10b981,
            'LOSER_BOUNCE':    0x0284c7, 'LOSER_BREAKDOWN': 0xdc2626,
            'PRE_GAINER':      0xf59e0b, 'PRE_LOSER':       0xf97316,
            'PRE_GAINER_STAR': 0xfbbf24, 'PRE_LOSER_STAR':  0xef4444,
        }
        color     = type_colors.get(sig['type'], 0x6366f1)
        dir_emoji = "📗" if sig['direction'] == 'LONG' else "📕" if sig['direction'] == 'SHORT' else "👁"
        tp_sl_line = ""
        if sig.get('tp') and sig.get('sl'):
            tp_sl_line = f"\n🎯 **TP:** `${sig['tp']:.6f}` | 🛑 **SL:** `${sig['sl']:.6f}` | **R:R:** {sig['rr']}"
        embed = {
            'title': f"{sig['emoji']} {sig['label']} | {sig['symbol']} ({sig.get('exchange','GATE')})",
            'color': color,
            'description': (
                f"{dir_emoji} **{sig['direction']}** | 4H: **{sig['chg_4h']:+.2f}%** | 1H: **{sig['chg_1h']:+.2f}%**\n"
                f"**Entry:** `${sig['close']:.6f}`{tp_sl_line}\n"
                f"📊 **RSI:** {sig['rsi']} | **VOL:** {sig['vol_ratio']}× avg"
            ),
            'footer': {'text': f'APEXAI G/L • {sig.get("scan_time","–")} | {sig.get("scan_time_pkt","–")}'}
        }
        requests.post(webhook_url, json={'embeds': [embed]}, timeout=5)
    except:
        pass


# ─── PERFORMANCE TRACKER ─────────────────────────────────────────────────────
def ensure_gl_performance():
    _hourly_cols = []
    for _h in range(1, 25):
        _hourly_cols += [f'h{_h}_price', f'h{_h}_pct']
    headers = ['signal_id','timestamp_utc','timestamp_pkt','symbol','type','direction',
               'entry','tp','sl','rr','chg_4h','chg_1h','rsi','vol_ratio',
               'outcome','outcome_time','pnl_pct','bars_to_outcome',
               'max_move_pct','max_move_price','max_move_time',
               'final_price','final_pct','final_report'] + _hourly_cols
    if not os.path.exists(GL_PERF_FILE):
        with open(GL_PERF_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)
    else:
        try:
            _df = pd.read_csv(GL_PERF_FILE); changed = False
            for col in headers:
                if col not in _df.columns:
                    _df[col] = ''; changed = True
            if changed: _df.to_csv(GL_PERF_FILE, index=False)
        except:
            pass

def log_gl_signal(sig):
    try:
        ensure_gl_performance()
        signal_id = f"{sig['symbol']}_{sig['type']}_{sig.get('scan_time','')}"
        if os.path.exists(GL_PERF_FILE):
            df = pd.read_csv(GL_PERF_FILE)
            if signal_id in df['signal_id'].values: return
        with open(GL_PERF_FILE, 'a', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow([
                signal_id, sig.get('scan_time',''), sig.get('scan_time_pkt',''),
                sig.get('symbol',''), sig.get('type',''), sig.get('direction',''),
                sig.get('close',''), sig.get('tp',''), sig.get('sl',''), sig.get('rr',''),
                sig.get('chg_4h',''), sig.get('chg_1h',''), sig.get('rsi',''), sig.get('vol_ratio',''),
                'PENDING','','','','','',''
            ])
    except Exception as e:
        print(f"[GL PERF] Log error: {e}")

def check_gl_outcomes():
    try:
        if not os.path.exists(GL_PERF_FILE): return
        df = pd.read_csv(GL_PERF_FILE)
        pending = df[df['outcome'] == 'PENDING']
        if pending.empty: return
        import ccxt as _ccxt
        ex = _ccxt.gate({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        for idx, row in pending.iterrows():
            try:
                sym    = row['symbol']
                entry  = float(row['entry'])
                tp     = float(row['tp'])  if str(row['tp'])  not in ['', 'nan'] else None
                sl     = float(row['sl'])  if str(row['sl'])  not in ['', 'nan'] else None
                if not tp or not sl: continue
                try:
                    sig_time  = pd.to_datetime(row['timestamp_utc'])
                    age_hours = (pd.Timestamp.utcnow() - sig_time.replace(tzinfo=None)).total_seconds() / 3600
                    if age_hours > 24:
                        df.at[idx, 'outcome']      = 'EXPIRED'
                        df.at[idx, 'outcome_time'] = _dual_time()[0]
                        continue
                except:
                    pass
                ticker  = ex.fetch_ticker(sym)
                current = float(ticker['last'])
                try:
                    _prev_max = float(row['max_move_pct']) if str(row.get('max_move_pct','')) not in ['','nan'] else 0.0
                    direction = str(row.get('direction',''))
                    if direction == 'LONG' or str(row.get('type','')) in ['PRE_GAINER','GAINER_PULLBACK','GAINER_BREAKOUT','LOSER_BOUNCE']:
                        _move_pct = (current - entry) / entry * 100
                    else:
                        _move_pct = (entry - current) / entry * 100
                    if abs(_move_pct) > abs(_prev_max):
                        df.at[idx, 'max_move_pct']   = round(_move_pct, 3)
                        df.at[idx, 'max_move_price']  = round(current, 6)
                        df.at[idx, 'max_move_time']   = _dual_time()[0]
                except: pass
                direction = str(row['direction'])
                if direction == 'LONG':
                    if current >= tp:
                        df.at[idx, 'outcome']      = 'WIN';  df.at[idx, 'pnl_pct'] = round((tp - entry)/entry*100, 2)
                        df.at[idx, 'outcome_time'] = _dual_time()[0]
                    elif current <= sl:
                        df.at[idx, 'outcome']      = 'LOSS'; df.at[idx, 'pnl_pct'] = round((sl - entry)/entry*100, 2)
                        df.at[idx, 'outcome_time'] = _dual_time()[0]
                elif direction == 'SHORT':
                    if current <= tp:
                        df.at[idx, 'outcome']      = 'WIN';  df.at[idx, 'pnl_pct'] = round((entry - tp)/entry*100, 2)
                        df.at[idx, 'outcome_time'] = _dual_time()[0]
                    elif current >= sl:
                        df.at[idx, 'outcome']      = 'LOSS'; df.at[idx, 'pnl_pct'] = round((entry - sl)/entry*100, 2)
                        df.at[idx, 'outcome_time'] = _dual_time()[0]
            except: continue
        df.to_csv(GL_PERF_FILE, index=False)
    except Exception as e:
        print(f"[GL PERF] Outcome check error: {e}")

def get_gl_stats():
    try:
        if not os.path.exists(GL_PERF_FILE): return {}
        df = pd.read_csv(GL_PERF_FILE)
        stats = {}
        for sig_type in ['GAINER_PULLBACK','GAINER_BREAKOUT','LOSER_BOUNCE','LOSER_BREAKDOWN',
                         'PRE_GAINER','PRE_LOSER','PRE_GAINER_STAR','PRE_LOSER_STAR']:
            subset  = df[df['type'] == sig_type]
            wins    = len(subset[subset['outcome'] == 'WIN'])
            losses  = len(subset[subset['outcome'] == 'LOSS'])
            pending = len(subset[subset['outcome'] == 'PENDING'])
            expired = len(subset[subset['outcome'] == 'EXPIRED'])
            total   = wins + losses
            wr      = round(wins/total*100) if total > 0 else 0
            avg_pnl = round(subset[subset['outcome'].isin(['WIN','LOSS'])]['pnl_pct'].astype(float).mean(), 2) if total > 0 else 0
            stats[sig_type] = {'wins':wins,'losses':losses,'pending':pending,'expired':expired,'total':total,'wr':wr,'avg_pnl':avg_pnl}
        return stats
    except:
        return {}


# ─── AI ANALYSIS (Groq) ──────────────────────────────────────────────────────
def _get_groq_key(s):
    try:
        k = st.secrets.get("GROQ_API_KEY", "")
        if k and k.strip(): return k.strip()
    except: pass
    return (s.get("groq_key") or "").strip()

def _ai_analyse_gl(signals, s):
    import json as _jj
    gk = _get_groq_key(s)
    if not gk or not signals: return {}
    try:
        blocks = []
        for sig in signals[:8]:
            sym = sig.get('symbol','').replace('/USDT:USDT','').replace('/USDT','').upper()
            blocks.append(
                f"COIN:{sym} Signal:{sig.get('type','')} DIR:{sig.get('direction','')}\n"
                f"4H:{sig.get('chg_4h',0):+.2f}% 1H:{sig.get('chg_1h',0):+.2f}%\n"
                f"RSI:{sig.get('rsi',0)} VOL:{sig.get('vol_ratio',1):.1f}x\n"
                f"Entry:${sig.get('close',0):.4f} TP:${sig.get('tp',0):.4f} SL:${sig.get('sl',0):.4f} RR:{sig.get('rr',0)}\n"
                f"Why: {' | '.join(sig.get('reasons',[])[:4])}"
            )
        payload = '\n\n'.join(blocks)
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {gk}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.3-70b-versatile",
                "max_tokens": 500,
                "messages": [{
                    "role": "user",
                    "content": (
                        f"You are a professional crypto futures trader. Analyse these G/L signals and respond ONLY with a valid JSON object.\n\n{payload}\n\n"
                        "For each COIN key return: {\"trade_confidence\":0-100, \"rank\":1-8, \"avoid\":bool, \"avoid_reason\":\"\", \"reason\":\"\", \"key_edge\":\"\", \"key_risk\":\"\"}"
                    )
                }]
            }, timeout=15)
        raw = r.json()['choices'][0]['message']['content']
        clean = raw.replace('```json','').replace('```','').strip()
        return _jj.loads(clean)
    except:
        return {}

def _render_ai_badge(ai_data):
    if not ai_data: return ''
    _conf  = ai_data.get('trade_confidence', 0)
    _rank  = ai_data.get('rank', 99)
    _reason= ai_data.get('reason', '')
    _avoid = ai_data.get('avoid', False)
    _ar    = ai_data.get('avoid_reason', '')
    _edge  = ai_data.get('key_edge', '')
    _risk  = ai_data.get('key_risk', '')
    _cc    = '#059669' if _conf >= 75 else ('#d97706' if _conf >= 55 else '#dc2626')
    _cb    = '#f0fdf4' if _conf >= 75 else ('#fffbeb' if _conf >= 55 else '#fef2f2')
    _eh    = f'<div style="font-family:monospace;font-size:.55rem;color:#166534;margin-top:2px;">⚡ Edge: {_edge}</div>' if _edge else ''
    _rh    = f'<div style="font-family:monospace;font-size:.55rem;color:#dc2626;margin-top:1px;">⚠ Risk: {_risk}</div>' if _risk else ''
    if _rank == 1 and not _avoid:
        return (f'<div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1.5px solid #059669;border-radius:7px;padding:7px 11px;margin:4px 0;">'
                f'<div style="display:flex;align-items:center;justify-content:space-between;">'
                f'<span style="font-family:monospace;font-size:.62rem;font-weight:900;color:#059669;">🎯 BEST SETUP — AI: {_conf}/100</span>'
                f'<span style="font-family:monospace;font-size:.55rem;padding:1px 6px;border-radius:3px;background:#059669;color:white;">RANK #1</span>'
                f'</div><div style="font-family:monospace;font-size:.57rem;color:#166534;margin-top:2px;">{_reason}</div>{_eh}{_rh}</div>')
    elif _avoid:
        return (f'<div style="background:#fef2f2;border:1px solid #dc2626;border-radius:6px;padding:6px 11px;margin:4px 0;">'
                f'<span style="font-family:monospace;font-size:.6rem;font-weight:700;color:#dc2626;">⛔ AI: AVOID — {_ar}</span>{_rh}</div>')
    else:
        return (f'<div style="background:{_cb};border:1px solid {_cc}44;border-radius:6px;padding:6px 11px;margin:4px 0;">'
                f'<span style="font-family:monospace;font-size:.6rem;font-weight:700;color:{_cc};">🤖 AI: {_conf}/100 · Rank #{_rank}</span>'
                f'<div style="font-family:monospace;font-size:.57rem;color:#475569;margin-top:1px;">{_reason}</div>{_eh}{_rh}</div>')


# ─── JOURNAL HELPERS ─────────────────────────────────────────────────────────
def ensure_journal():
    headers = ["ts","symbol","exchange","type","pump_score","class","price","tp","sl","triggers","status","entry_touched","tp1","tp2","tp3","scan_time"]
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE, 'w', newline='', encoding='utf-8') as f:
            csv.writer(f).writerow(headers)

def save_to_journal(signals):
    ensure_journal()
    _jt = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    with open(JOURNAL_FILE, 'a', newline='', encoding='utf-8') as jf:
        w = csv.writer(jf)
        for sig in signals:
            w.writerow([
                _jt, sig.get('symbol',''), sig.get('exchange','GATE'),
                sig.get('label','GL'), '', sig.get('type',''),
                sig.get('close',''), sig.get('tp',''), sig.get('sl',''),
                f"4H:{sig.get('chg_4h','')}% RSI:{sig.get('rsi','')} VOL:{sig.get('vol_ratio','')}x RR:{sig.get('rr','')}",
                'ACTIVE','0','0','0','0', sig.get('scan_time','')
            ])


# ═══════════════════════════════════════════════════════════════════════════
# ─── CSS — DARK THEME + MOBILE OPTIMIZED ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
html, body, .stApp { background:#0b0f1a !important; color:#e2e8f0 !important; }
.block-container { padding:0.5rem 0.75rem 2rem !important; max-width:100% !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg,#0f172a 0%,#1e293b 100%) !important;
  border-right:1px solid #334155 !important;
}
section[data-testid="stSidebar"] * { color:#cbd5e1 !important; }
section[data-testid="stSidebar"] .stSlider > div { color:#94a3b8 !important; }
section[data-testid="stSidebar"] label { color:#94a3b8 !important; font-size:.75rem !important; }
section[data-testid="stSidebar"] .stExpander { border:1px solid #1e293b !important; background:#0f172a !important; }
section[data-testid="stSidebar"] .stExpander summary p { color:#38bdf8 !important; font-weight:700 !important; }

/* ── Buttons ── */
.stButton > button {
  background:linear-gradient(135deg,#0ea5e9,#0284c7) !important;
  color:#fff !important; border:none !important; border-radius:8px !important;
  font-family:monospace !important; font-weight:700 !important; font-size:.8rem !important;
  padding:8px 16px !important; transition:all .2s !important;
}
.stButton > button:hover { background:linear-gradient(135deg,#38bdf8,#0ea5e9) !important; transform:translateY(-1px) !important; box-shadow:0 4px 12px #0ea5e940 !important; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#10b981,#059669) !important; }
.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#34d399,#10b981) !important; box-shadow:0 4px 12px #10b98140 !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
  background:#1e293b !important; border:1px solid #334155 !important;
  border-radius:10px !important; padding:12px 14px !important;
}
div[data-testid="stMetricLabel"] { color:#94a3b8 !important; font-size:.72rem !important; font-family:monospace !important; }
div[data-testid="stMetricValue"] { color:#f1f5f9 !important; font-family:monospace !important; font-size:1.2rem !important; font-weight:900 !important; }

/* ── Inputs ── */
.stTextInput input, .stNumberInput input {
  background:#1e293b !important; color:#f1f5f9 !important;
  border:1px solid #334155 !important; border-radius:8px !important;
  font-family:monospace !important;
}
.stSelectbox > div > div { background:#1e293b !important; border:1px solid #334155 !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background:#1e293b !important; border:1px solid #334155 !important; border-radius:8px !important; }
.streamlit-expanderContent { background:#131c2e !important; border:1px solid #1e293b !important; }

/* ── Divider ── */
hr { border-color:#1e293b !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color:#0ea5e9 !important; }

/* ── Tables ── */
.stDataFrame { background:#1e293b !important; border-radius:10px !important; }
.stDataFrame th { background:#0f172a !important; color:#94a3b8 !important; }
.stDataFrame td { background:#1e293b !important; color:#e2e8f0 !important; }

/* ── Section labels ── */
.gl-section-label {
  font-family:monospace; font-weight:800; font-size:.72rem;
  color:#64748b; text-transform:uppercase; letter-spacing:.12em;
  padding:10px 0 4px; border-bottom:1px solid #1e293b; margin-bottom:10px;
}

/* ── Signal cards ── */
.sig-card {
  border-radius:12px; padding:14px 16px; margin:6px 0;
  border-left:4px solid; transition:transform .15s;
}
.sig-card:hover { transform:translateX(2px); }

/* ── Mobile ── */
@media (max-width:640px) {
  .block-container { padding:0.25rem 0.4rem 3rem !important; }
  div[data-testid="stMetricValue"] { font-size:1rem !important; }
  .sig-card { padding:10px 12px !important; }
  section[data-testid="stSidebar"] { width:85vw !important; }
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:#0b0f1a; }
::-webkit-scrollbar-thumb { background:#334155; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#475569; }

/* ── Toast overrides ── */
div[data-testid="stNotification"] { background:#1e293b !important; border:1px solid #334155 !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ─── SIDEBAR — SETTINGS ──────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
S = load_settings()
with st.sidebar:
    st.markdown('''
    <div style="padding:16px 0 4px;">
      <div style="font-family:monospace;font-size:1.05rem;font-weight:900;color:#38bdf8;letter-spacing:.05em;">📊 APEXAI</div>
      <div style="font-family:monospace;font-size:.62rem;color:#64748b;margin-top:2px;">G/L SCANNER — GATE.io Perpetuals</div>
      <div style="height:2px;background:linear-gradient(90deg,#0ea5e9,#6366f1,transparent);margin-top:8px;border-radius:2px;"></div>
    </div>''', unsafe_allow_html=True)

    with st.expander("📊 Scan Parameters", expanded=True):
        ns_top_n         = st.slider("Top N gainers/losers", 5, 50, int(S.get('gl_top_n', 20)))
        ns_min_gain      = st.slider("Min 4H gain %", 1.0, 10.0, float(S.get('gl_min_gain_pct', 3.0)), 0.5)
        ns_min_loss      = st.slider("Min 4H loss %", 1.0, 10.0, float(S.get('gl_min_loss_pct', 1.5)), 0.5)
        ns_pullback_min  = st.slider("Min pullback %", 0.5, 5.0, float(S.get('gl_pullback_min', 1.5)), 0.5)
        ns_pullback_max  = st.slider("Max pullback %", 3.0, 15.0, float(S.get('gl_pullback_max', 8.0)), 0.5)
        ns_min_rr        = st.slider("Min R:R", 1.0, 4.0, float(S.get('gl_min_rr', 1.5)), 0.1)
        ns_vol_expansion = st.slider("Vol expansion ×", 1.0, 3.0, float(S.get('gl_vol_expansion', 1.5)), 0.1)
        ns_rsi_ob        = st.slider("RSI overbought cap", 60, 85, int(S.get('gl_rsi_ob', 75)))
        ns_rsi_os        = st.slider("RSI oversold floor", 15, 45, int(S.get('gl_rsi_os', 30)))
        ns_interval      = st.number_input("Auto-scan interval (min)", 1, 60, int(S.get('gl_interval', 1)))

    with st.expander("🔔 Alert Types"):
        ns_pullback  = st.checkbox("🟢 Gainer Pullback",  value=S.get('gl_alert_pullback', True))
        ns_breakout  = st.checkbox("🚀 Gainer Breakout",  value=S.get('gl_alert_breakout', True))
        ns_bounce    = st.checkbox("🔄 Loser Bounce",     value=S.get('gl_alert_bounce', True))
        ns_breakdown = st.checkbox("📉 Loser Breakdown",  value=S.get('gl_alert_breakdown', True))
        ns_pregainer = st.checkbox("👀 Pre-Gainer Watch", value=S.get('gl_alert_pregainer', True))
        ns_preloser  = st.checkbox("⚠️ Pre-Loser Watch",  value=S.get('gl_alert_preloser', True))

    with st.expander("🔗 Integrations"):
        ns_discord  = st.text_input("Discord Webhook", value=S.get('discord_webhook',''), type="password")
        ns_groq     = st.text_input("Groq API Key",    value=S.get('groq_key',''),       type="password")
        ns_blacklist= st.text_input("Symbol Blacklist", value=S.get('symbol_blacklist','PAXG,WBTC,WETH,STETH,FDUSD,USDTUSDT'), help="Comma-separated")

    if st.button("💾 Save Settings", use_container_width=True):
        new_s = dict(S)
        new_s.update({
            'gl_top_n': ns_top_n, 'gl_min_gain_pct': ns_min_gain, 'gl_min_loss_pct': ns_min_loss,
            'gl_pullback_min': ns_pullback_min, 'gl_pullback_max': ns_pullback_max,
            'gl_min_rr': ns_min_rr, 'gl_vol_expansion': ns_vol_expansion,
            'gl_rsi_ob': ns_rsi_ob, 'gl_rsi_os': ns_rsi_os, 'gl_interval': ns_interval,
            'gl_alert_pullback': ns_pullback, 'gl_alert_breakout': ns_breakout,
            'gl_alert_bounce': ns_bounce, 'gl_alert_breakdown': ns_breakdown,
            'gl_alert_pregainer': ns_pregainer, 'gl_alert_preloser': ns_preloser,
            'discord_webhook': ns_discord, 'groq_key': ns_groq, 'symbol_blacklist': ns_blacklist,
        })
        save_settings(new_s)
        S = new_s
        st.success("✅ Settings saved")
        st.rerun()

    st.markdown('---')
    if st.button("📥 Upload gl_performance.csv", use_container_width=True):
        st.session_state['show_upload'] = True
    if st.session_state.get('show_upload'):
        uploaded = st.file_uploader("gl_performance.csv", type=['csv'], key='gl_up')
        if uploaded:
            with open(GL_PERF_FILE, 'wb') as f: f.write(uploaded.read())
            st.success("Uploaded ✅"); st.session_state['show_upload'] = False; st.rerun()

    st.markdown('''<div style="font-family:monospace;font-size:.55rem;color:#334155;padding-top:12px;line-height:1.6;">
    🟢 Gainer Pullback — 4H up, 5m dip<br>
    🚀 Gainer Breakout — new high + vol<br>
    🔄 Loser Bounce — 4H down, bounce<br>
    📉 Loser Breakdown — new low + vol<br>
    👀 Pre-Gainer — flat + vol spike<br>
    ⭐ Star — highest conviction pre-sig
    </div>''', unsafe_allow_html=True)

eff_s = load_settings()

# ═══════════════════════════════════════════════════════════════════════════
# ─── MAIN HEADER ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
utc_now, pkt_now = _dual_time()
st.markdown(f'''
<div style="background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid #1e3a5f;border-radius:14px;padding:16px 20px;margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">
    <div>
      <div style="font-family:monospace;font-size:1.15rem;font-weight:900;color:#38bdf8;letter-spacing:.02em;">
        📊 APEXAI — GAINERS &amp; LOSERS SCANNER
      </div>
      <div style="font-family:monospace;font-size:.65rem;color:#64748b;margin-top:3px;">
        4H Catalyst + 5m Entry &nbsp;·&nbsp; GATE.io Perpetuals &nbsp;·&nbsp; Live Tick Data
      </div>
    </div>
    <div style="text-align:right;">
      <div style="font-family:monospace;font-size:.62rem;color:#0ea5e9;">{utc_now}</div>
      <div style="font-family:monospace;font-size:.62rem;color:#38bdf8;font-weight:700;">{pkt_now}</div>
    </div>
  </div>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;">
    <span style="background:#0ea5e920;border:1px solid #0ea5e940;border-radius:20px;padding:2px 10px;font-family:monospace;font-size:.6rem;color:#38bdf8;">🟡 GATE.io</span>
    <span style="background:#10b98120;border:1px solid #10b98140;border-radius:20px;padding:2px 10px;font-family:monospace;font-size:.6rem;color:#34d399;">🔄 Auto-scan active</span>
    <span style="background:#6366f120;border:1px solid #6366f140;border-radius:20px;padding:2px 10px;font-family:monospace;font-size:.6rem;color:#a5b4fc;">⚡ 8 signal types</span>
  </div>
</div>''', unsafe_allow_html=True)

# ─── SCAN CONTROLS ───────────────────────────────────────────────────────────
ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 2, 2, 2])
with ctrl_c1:
    gl_run = st.button("▶ Run G/L Scan", use_container_width=True, key='gl_run_btn', type='primary')
with ctrl_c2:
    if st.button("🔄 Check Outcomes", use_container_width=True):
        with st.spinner("Checking outcomes..."):
            check_gl_outcomes()
        st.toast("✅ Outcomes updated", icon="✅")
with ctrl_c3:
    if st.button("🗑 Clear Results", use_container_width=True):
        st.session_state['gl_results'] = []
        st.rerun()
with ctrl_c4:
    _nsig = len(st.session_state.get('gl_results', []))
    _last = st.session_state.get('gl_last_scan', '—')
    _ivl  = int(eff_s.get('gl_interval', 1))
    _color = '#10b981' if _nsig > 0 else '#475569'
    st.markdown(f'<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:8px 12px;font-family:monospace;font-size:.62rem;color:{_color};text-align:center;line-height:1.7;">'
                f'<b>{_nsig}</b> signals &nbsp;·&nbsp; {_ivl}m auto</div>', unsafe_allow_html=True)

gl_status = st.empty()

# ─── STATUS BAR ──────────────────────────────────────────────────────────────
gl_last_ts  = st.session_state.get('gl_last_ts', 0)
gl_interval = int(eff_s.get('gl_interval', 1))
_time_since = int(time.time() - gl_last_ts) if gl_last_ts > 0 else -1
_next_in    = max(0, gl_interval * 60 - _time_since) if _time_since >= 0 else -1

if gl_last_ts > 0:
    _bar_pct = min(100, int((_time_since / (gl_interval * 60)) * 100))
    _bar_color = '#10b981' if _bar_pct < 70 else ('#f59e0b' if _bar_pct < 90 else '#0ea5e9')
    st.markdown(f'''
    <div style="background:#1e293b;border:1px solid #334155;border-radius:8px;padding:8px 14px;margin-bottom:8px;">
      <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.6rem;color:#64748b;margin-bottom:5px;">
        <span>Last: {st.session_state.get("gl_last_scan","—")}</span>
        <span style="color:#0ea5e9;">Next scan: {_next_in//60}m {_next_in%60}s</span>
      </div>
      <div style="background:#0f172a;border-radius:4px;height:4px;overflow:hidden;">
        <div style="height:4px;width:{_bar_pct}%;background:{_bar_color};border-radius:4px;transition:width .5s;"></div>
      </div>
    </div>''', unsafe_allow_html=True)

# ─── AUTO-SCAN TIMING ────────────────────────────────────────────────────────
gl_should_run = gl_run or (gl_last_ts > 0 and (time.time() - gl_last_ts >= gl_interval * 60))

# ─── TWO-PASS SCAN LOGIC ─────────────────────────────────────────────────────
_gl_active_done = st.session_state.pop('gl_active_done', False)
_gl_pre_pending = st.session_state.pop('gl_pre_pending', False)

if _gl_pre_pending:
    try:
        s_pre = dict(eff_s)
        s_pre['gl_alert_pullback']  = False
        s_pre['gl_alert_breakout']  = False
        s_pre['gl_alert_bounce']    = False
        s_pre['gl_alert_breakdown'] = False
        gl_status.markdown('<div style="background:#1e293b;border:1px solid #7c3aed40;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:.65rem;color:#a78bfa;">👀 Scanning flat coins for Pre-Gainer/Pre-Loser signals...</div>', unsafe_allow_html=True)
        pre_signals = run_gl_scan(s_pre, status_placeholder=gl_status,
                                  instant_webhook=eff_s.get('discord_webhook') or None,
                                  instant_alerted=st.session_state.get('gl_alerted', set()))
        if pre_signals:
            existing  = st.session_state.get('gl_results', [])
            active_only = [s for s in existing if s.get('type') not in ['PRE_GAINER','PRE_LOSER','PRE_GAINER_STAR','PRE_LOSER_STAR']]
            all_sigs  = active_only + pre_signals
            if all_sigs and eff_s.get('groq_key',''):
                _ai = _ai_analyse_gl(all_sigs, eff_s)
                for _gs in all_sigs:
                    _gsym = _gs.get('symbol','').replace('/USDT:USDT','').replace('/USDT','').upper()
                    if _gsym in _ai: _gs['_ai_data'] = _ai[_gsym]
            st.session_state['gl_results'] = all_sigs
            for sig in pre_signals:
                log_gl_signal(sig)
            if eff_s.get('discord_webhook'):
                alerted = st.session_state.get('gl_alerted', set())
                for sig in pre_signals:
                    _sid = f"{sig['symbol']}_{sig['type']}_{datetime.now().strftime('%Y%m%d%H')}"
                    if _sid not in alerted:
                        send_gl_discord_alert(eff_s['discord_webhook'], sig); alerted.add(_sid)
                st.session_state['gl_alerted'] = alerted
        history = st.session_state.get('gl_history', [])
        for sig in pre_signals:
            sig['_id'] = f"{sig['symbol']}_{sig['type']}_{sig.get('scan_time','')}"
            if sig['_id'] not in [h.get('_id') for h in history]: history.insert(0, sig)
        st.session_state['gl_history'] = history[:10]
        gl_status.markdown('<div style="background:#1e293b;border:1px solid #10b98140;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:.65rem;color:#34d399;">✅ Pre-signal scan complete</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"G/L pre-scan error: {e}")

elif gl_should_run and not _gl_active_done:
    with st.spinner("📊 Fetching live gainers/losers from GATE.io..."):
        try:
            s_active = dict(eff_s)
            s_active['gl_alert_pregainer'] = False
            s_active['gl_alert_preloser']  = False
            gl_status.markdown('<div style="background:#1e293b;border:1px solid #0ea5e940;border-radius:8px;padding:8px 14px;font-family:monospace;font-size:.65rem;color:#38bdf8;">⚡ Scanning top gainers/losers for active signals...</div>', unsafe_allow_html=True)
            active_signals = run_gl_scan(s_active, status_placeholder=gl_status,
                                         instant_webhook=eff_s.get('discord_webhook') or None,
                                         instant_alerted=st.session_state.get('gl_alerted', set()))
            existing = st.session_state.get('gl_results', [])
            st.session_state['gl_results'] = active_signals + [s for s in existing if s.get('type') in ['PRE_GAINER','PRE_LOSER','PRE_GAINER_STAR','PRE_LOSER_STAR']]
            st.session_state['gl_active_done'] = True
            st.session_state['gl_last_ts']     = time.time()
            st.session_state['gl_last_scan']   = _dual_time()[1]
            for sig in active_signals: log_gl_signal(sig)
            try: check_gl_outcomes()
            except: pass
            # AI analysis
            if active_signals and eff_s.get('groq_key',''):
                _ai = _ai_analyse_gl(active_signals, eff_s)
                for _gs in active_signals:
                    _gsym = _gs.get('symbol','').replace('/USDT:USDT','').replace('/USDT','').upper()
                    if _gsym in _ai: _gs['_ai_data'] = _ai[_gsym]
            # History
            history = st.session_state.get('gl_history', [])
            for sig in active_signals:
                sig['_id'] = f"{sig['symbol']}_{sig['type']}_{sig.get('scan_time','')}"
                if sig['_id'] not in [h.get('_id') for h in history]: history.insert(0, sig)
            st.session_state['gl_history'] = history[:10]
            # Journal
            try: save_to_journal(active_signals)
            except: pass
            # Discord
            if eff_s.get('discord_webhook'):
                alerted = st.session_state.get('gl_alerted', set())
                for sig in active_signals:
                    _sid = f"{sig['symbol']}_{sig['type']}_{datetime.now().strftime('%Y%m%d%H')}"
                    if _sid not in alerted:
                        send_gl_discord_alert(eff_s['discord_webhook'], sig); alerted.add(_sid)
                st.session_state['gl_alerted'] = alerted
            st.session_state['gl_pre_pending'] = True
            st.rerun()
        except Exception as e:
            st.error(f"G/L scan error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# ─── SIGNAL HISTORY STRIP (last 10) ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
gl_history = st.session_state.get('gl_history', [])
if gl_history:
    st.markdown('<div class="gl-section-label">📋 Recent Signals</div>', unsafe_allow_html=True)
    hist_cols = st.columns(min(len(gl_history), 5))
    to_remove = None
    _hcolors = {
        'GAINER_PULLBACK':  ('#0b2d1a','#10b981'), 'GAINER_BREAKOUT':  ('#0a2e1a','#34d399'),
        'LOSER_BOUNCE':     ('#0a1f3a','#38bdf8'), 'LOSER_BREAKDOWN':  ('#2d0a0a','#f87171'),
        'PRE_GAINER':       ('#2d2000','#fbbf24'), 'PRE_LOSER':        ('#2d1500','#fb923c'),
        'PRE_GAINER_STAR':  ('#2d2500','#fde68a'), 'PRE_LOSER_STAR':   ('#2d1a00','#fed7aa'),
    }
    for i, sig in enumerate(gl_history[:5]):
        bg, color = _hcolors.get(sig.get('type',''), ('#1e293b','#64748b'))
        sym = sig.get('symbol','').replace('/USDT:USDT','').replace('/USDT','')
        _gl_outcome_html = ''
        if os.path.exists(GL_PERF_FILE):
            try:
                _pdf = pd.read_csv(GL_PERF_FILE)
                _pid = f"{sig.get('symbol','')}_{sig.get('type','')}_{sig.get('scan_time','')}"
                _prow = _pdf[_pdf['signal_id'] == _pid]
                if not _prow.empty:
                    _oc   = str(_prow.iloc[0].get('outcome','PENDING'))
                    _pn_v = float(_prow.iloc[0].get('pnl_pct', 0) or 0)
                    _mx_v = float(_prow.iloc[0].get('max_move_pct', 0) or 0)
                    _mx_str = f' · max {_mx_v:+.1f}%' if _mx_v else ''
                    if _oc == 'WIN':
                        _gl_outcome_html = f'<div style="background:#0b2d1a;border:1px solid #10b981;border-radius:4px;padding:2px 5px;margin-top:3px;font-family:monospace;font-size:.55rem;color:#34d399;font-weight:700;">✅ +{_pn_v:.2f}%{_mx_str}</div>'
                    elif _oc == 'LOSS':
                        _gl_outcome_html = f'<div style="background:#2d0a0a;border:1px solid #ef4444;border-radius:4px;padding:2px 5px;margin-top:3px;font-family:monospace;font-size:.55rem;color:#f87171;font-weight:700;">❌ {_pn_v:.2f}%{_mx_str}</div>'
                    elif _oc == 'PENDING':
                        _mc = '#34d399' if _mx_v > 0 else '#f87171'
                        _gl_outcome_html = f'<div style="background:#1e293b;border:1px solid {_mc}60;border-radius:4px;padding:2px 5px;margin-top:3px;font-family:monospace;font-size:.55rem;color:{_mc};">⏳ {_mx_v:+.2f}%</div>'
                    elif _oc == 'EXPIRED':
                        _gl_outcome_html = f'<div style="background:#1e293b;border:1px solid #334155;border-radius:4px;padding:2px 5px;margin-top:3px;font-family:monospace;font-size:.55rem;color:#64748b;">⌛ EXPIRED</div>'
            except: pass

        with hist_cols[i]:
            st.markdown(f'''<div style="background:{bg};border:1px solid {color}40;border-left:3px solid {color};border-radius:8px;padding:8px 10px;margin:2px 0;">
              <div style="font-family:monospace;font-size:.68rem;font-weight:800;color:{color};">{sig.get("emoji","")} {sym}</div>
              <div style="font-family:monospace;font-size:.58rem;color:#94a3b8;">{sig.get("type","").replace("_"," ")}</div>
              <div style="font-family:monospace;font-size:.58rem;color:#64748b;">4H:{sig.get("chg_4h",0):+.1f}% R:{sig.get("rsi",0)}</div>
              {_gl_outcome_html}
            </div>''', unsafe_allow_html=True)
            if st.button("✕", key=f'gl_hist_{i}', help="Dismiss"):
                to_remove = i
    if to_remove is not None:
        gl_history.pop(to_remove)
        st.session_state['gl_history'] = gl_history
        st.rerun()
    st.markdown('---')


# ═══════════════════════════════════════════════════════════════════════════
# ─── PERFORMANCE SCOREBOARD ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
gl_stats = get_gl_stats()
if any(v['total'] > 0 for v in gl_stats.values()):
    st.markdown('<div class="gl-section-label">📊 Performance Tracker</div>', unsafe_allow_html=True)
    type_emojis = {
        'GAINER_PULLBACK':'🟢','GAINER_BREAKOUT':'🚀','LOSER_BOUNCE':'🔄','LOSER_BREAKDOWN':'📉',
        'PRE_GAINER':'👀','PRE_LOSER':'⚠️','PRE_GAINER_STAR':'⭐','PRE_LOSER_STAR':'⭐'
    }
    score_cols = st.columns(4)
    col_idx = 0
    for sig_type, stat in gl_stats.items():
        if stat['total'] == 0: continue
        emoji    = type_emojis.get(sig_type, '')
        wr       = stat['wr']
        bar_fill = int(wr / 10)
        bar      = '█' * bar_fill + '░' * (10 - bar_fill)
        color    = '#10b981' if wr >= 60 else ('#f59e0b' if wr >= 40 else '#ef4444')
        bg_dark  = '#0b2d1a' if wr >= 60 else ('#2d2000' if wr >= 40 else '#2d0a0a')
        with score_cols[col_idx % 4]:
            st.markdown(f'''<div style="background:{bg_dark};border:1px solid {color}40;border-radius:8px;padding:10px 12px;margin:3px 0;">
              <div style="font-family:monospace;font-size:.65rem;font-weight:700;color:{color};">{emoji} {sig_type.replace("_"," ")}</div>
              <div style="font-family:monospace;font-size:.75rem;color:{color};font-weight:800;">{bar} {wr}%</div>
              <div style="font-family:monospace;font-size:.6rem;color:#94a3b8;">{stat["wins"]}W / {stat["losses"]}L | avg: <span style="color:{color};">{stat["avg_pnl"]:+.1f}%</span></div>
              <div style="font-family:monospace;font-size:.55rem;color:#475569;">{stat["pending"]}⏳ {stat["expired"]}⌛</div>
            </div>''', unsafe_allow_html=True)
        col_idx += 1

    # Overall verdict bar
    if os.path.exists(GL_PERF_FILE):
        try:
            _perf_df = pd.read_csv(GL_PERF_FILE)
            _closed  = _perf_df[_perf_df['outcome'].isin(['WIN','LOSS'])]
            if len(_closed) >= 5:
                _wins    = _closed[pd.to_numeric(_closed['pnl_pct'], errors='coerce').fillna(0) > 0]
                _losses  = _closed[pd.to_numeric(_closed['pnl_pct'], errors='coerce').fillna(0) < 0]
                _total_p = pd.to_numeric(_closed['pnl_pct'], errors='coerce').fillna(0).sum()
                _wr      = len(_wins) / len(_closed) * 100
                _pf      = abs(pd.to_numeric(_wins['pnl_pct'], errors='coerce').fillna(0).sum()) / max(abs(pd.to_numeric(_losses['pnl_pct'], errors='coerce').fillna(0).sum()), 0.01)
                _gc      = "#10b981" if _total_p > 0 else "#ef4444"
                _gbg     = "#0b2d1a" if _total_p > 0 else "#2d0a0a"
                _gv      = "✅ PROFITABLE" if _total_p > 0 else "❌ LOSING"
                st.markdown(f'''<div style="background:{_gbg};border:1.5px solid {_gc};border-radius:10px;padding:14px 20px;margin:10px 0;">
                  <div style="font-family:monospace;font-size:.8rem;font-weight:900;color:{_gc};">{_gv}</div>
                  <div style="display:flex;gap:24px;flex-wrap:wrap;margin-top:6px;">
                    <span style="font-family:monospace;font-size:.65rem;color:#94a3b8;">Total P&L: <b style="color:{_gc};">{_total_p:+.1f}%</b></span>
                    <span style="font-family:monospace;font-size:.65rem;color:#94a3b8;">Win Rate: <b style="color:{_gc};">{_wr:.0f}%</b></span>
                    <span style="font-family:monospace;font-size:.65rem;color:#94a3b8;">Profit Factor: <b style="color:{_gc};">{_pf:.2f}</b></span>
                    <span style="font-family:monospace;font-size:.65rem;color:#94a3b8;">Closed: <b>{len(_closed)}</b></span>
                  </div>
                </div>''', unsafe_allow_html=True)
        except: pass
    st.markdown('---')


# ═══════════════════════════════════════════════════════════════════════════
# ─── LIVE SIGNALS DISPLAY ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════
gl_results = st.session_state.get('gl_results', [])

if gl_results:
    st.markdown(f'<div class="gl-section-label">🔴 Live Signals &nbsp;<span style="color:#10b981;font-size:.7rem;font-weight:700;">{len(gl_results)} found</span></div>', unsafe_allow_html=True)
    type_order = ['PRE_GAINER_STAR','PRE_LOSER_STAR','GAINER_PULLBACK','GAINER_BREAKOUT','LOSER_BOUNCE','LOSER_BREAKDOWN','PRE_GAINER','PRE_LOSER']
    # Dark theme card colors: bg, border, accent
    _card_colors = {
        'GAINER_PULLBACK':  ('#0b2d1a','#10b981','#34d399'),
        'GAINER_BREAKOUT':  ('#0a2e1a','#16a34a','#4ade80'),
        'LOSER_BOUNCE':     ('#0a1f3a','#0ea5e9','#38bdf8'),
        'LOSER_BREAKDOWN':  ('#2d0a0a','#ef4444','#f87171'),
        'PRE_GAINER':       ('#2d2000','#f59e0b','#fbbf24'),
        'PRE_LOSER':        ('#2d1500','#f97316','#fb923c'),
        'PRE_GAINER_STAR':  ('#2d2500','#d97706','#fde68a'),
        'PRE_LOSER_STAR':   ('#2d1a00','#ea580c','#fed7aa'),
    }
    for sig_type in type_order:
        type_sigs = [r for r in gl_results if r['type'] == sig_type]
        if not type_sigs: continue
        bg, border, accent = _card_colors.get(sig_type, ('#1e293b','#334155','#94a3b8'))
        first = type_sigs[0]
        st.markdown(f'''<div style="font-family:monospace;font-size:.65rem;font-weight:800;color:{accent};
          padding:8px 0 4px;letter-spacing:.05em;">{first["emoji"]} {first["label"]} — {len(type_sigs)} signal{"s" if len(type_sigs)>1 else ""}</div>''',
          unsafe_allow_html=True)
        for sig in type_sigs:
            sym      = sig["symbol"].replace("/USDT:USDT","").replace("/USDT","")
            tp_sl    = (f'🎯 <span style="color:#34d399;">${sig["tp"]:.6f}</span> &nbsp;|&nbsp; '
                        f'🛑 <span style="color:#f87171;">${sig["sl"]:.6f}</span> &nbsp;|&nbsp; '
                        f'R:R <span style="color:{accent};font-weight:700;">{sig["rr"]}</span>'
                        if sig.get("tp") else '👁 Watch for entry setup')
            ai_badge = _render_ai_badge(sig.get('_ai_data', {}))
            reasons  = (' · '.join(sig['reasons'][:5]) if sig.get('reasons') else '')
            chg_color = '#34d399' if sig.get('chg_4h',0) >= 0 else '#f87171'
            dir_badge_bg = '#0b2d1a' if sig.get('direction') == 'LONG' else ('#2d0a0a' if sig.get('direction') == 'SHORT' else '#1e293b')
            dir_badge_c  = '#34d399' if sig.get('direction') == 'LONG' else ('#f87171' if sig.get('direction') == 'SHORT' else '#94a3b8')
            st.markdown(f'''
            <div style="background:{bg};border:1px solid {border}50;border-left:4px solid {accent};
              border-radius:10px;padding:12px 16px;margin:5px 0;">
              <!-- Row 1: coin + direction + time -->
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:6px;">
                <div style="display:flex;align-items:center;gap:8px;">
                  <span style="font-family:monospace;font-weight:900;font-size:.9rem;color:{accent};">{sig["emoji"]} {sym}</span>
                  <span style="background:{dir_badge_bg};border:1px solid {dir_badge_c}60;border-radius:20px;
                    padding:1px 9px;font-family:monospace;font-size:.6rem;color:{dir_badge_c};font-weight:700;">{sig["direction"]}</span>
                </div>
                <span style="font-family:monospace;font-size:.58rem;color:#475569;">{sig.get("exchange","GATE")} · {sig.get("scan_time_pkt","–")}</span>
              </div>
              <!-- AI badge -->
              {ai_badge}
              <!-- Row 2: key metrics pill row -->
              <div style="display:flex;gap:6px;flex-wrap:wrap;margin:6px 0;">
                <span style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:3px 8px;
                  font-family:monospace;font-size:.62rem;color:{chg_color};">4H {sig["chg_4h"]:+.2f}%</span>
                <span style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:3px 8px;
                  font-family:monospace;font-size:.62rem;color:#94a3b8;">1H {sig["chg_1h"]:+.2f}%</span>
                <span style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:3px 8px;
                  font-family:monospace;font-size:.62rem;color:#94a3b8;">RSI {sig["rsi"]}</span>
                <span style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:3px 8px;
                  font-family:monospace;font-size:.62rem;color:#94a3b8;">VOL {sig["vol_ratio"]}×</span>
                <span style="background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:3px 8px;
                  font-family:monospace;font-size:.62rem;color:#e2e8f0;">Entry ${sig["close"]:.6f}</span>
              </div>
              <!-- Row 3: TP/SL -->
              <div style="font-family:monospace;font-size:.62rem;color:#94a3b8;margin-top:4px;">{tp_sl}</div>
              <!-- Row 4: reasons -->
              {f'<div style="font-family:monospace;font-size:.58rem;color:#475569;margin-top:4px;line-height:1.5;">{reasons}</div>' if reasons else ''}
            </div>''', unsafe_allow_html=True)
else:
    st.markdown('''
    <div style="background:#1e293b;border:1px dashed #334155;border-radius:12px;padding:40px 20px;text-align:center;margin:16px 0;">
      <div style="font-size:2rem;margin-bottom:8px;">📊</div>
      <div style="font-family:monospace;font-size:.8rem;color:#64748b;font-weight:700;">No signals yet</div>
      <div style="font-family:monospace;font-size:.65rem;color:#475569;margin-top:4px;">Click ▶ Run G/L Scan to start</div>
    </div>''', unsafe_allow_html=True)


# ─── RAW DATA EXPANDER ───────────────────────────────────────────────────────
if gl_results:
    with st.expander(f"📋 Raw Data Table ({len(gl_results)} signals)"):
        _raw_df = pd.DataFrame([{
            'Symbol':   s.get('symbol','').replace('/USDT:USDT',''),
            'Type':     s.get('type',''),
            'Dir':      s.get('direction',''),
            '4H%':      s.get('chg_4h',0),
            '1H%':      s.get('chg_1h',0),
            'RSI':      s.get('rsi',0),
            'VOL×':     s.get('vol_ratio',0),
            'Entry':    s.get('close',0),
            'TP':       s.get('tp',0),
            'SL':       s.get('sl',0),
            'R:R':      s.get('rr',0),
            'Exchange': s.get('exchange',''),
        } for s in gl_results])
        st.dataframe(_raw_df, use_container_width=True, hide_index=True)
        csv_data = _raw_df.to_csv(index=False)
        st.download_button("⬇️ Download Signals CSV", csv_data, "gl_signals.csv", "text/csv")

# ─── PERF FILE DOWNLOAD ──────────────────────────────────────────────────────
if os.path.exists(GL_PERF_FILE):
    _dl_c1, _dl_c2 = st.columns(2)
    with _dl_c1:
        with open(GL_PERF_FILE, 'r', encoding='utf-8') as _pf:
            st.download_button("⬇️ Download gl_performance.csv", _pf.read(), GL_PERF_FILE, "text/csv", key='dl_perf')


# ─── AUTO-REFRESH + FOOTER ───────────────────────────────────────────────────
if eff_s.get('gl_enabled', True):
    next_gl = int(eff_s.get('gl_interval', 1) * 60 - (time.time() - st.session_state.get('gl_last_ts', 0)))
    next_gl = max(0, next_gl)
    st.markdown(f'''
    <div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;padding:10px 16px;text-align:center;margin-top:16px;">
      <span style="font-family:monospace;font-size:.62rem;color:#0ea5e9;">📊 Auto-Scan active — Next refresh in
        <b style="color:#38bdf8;">{next_gl//60}m {next_gl%60}s</b>
      </span>
      <div style="font-family:monospace;font-size:.55rem;color:#334155;margin-top:3px;">
        APEXAI G/L Scanner &nbsp;·&nbsp; GATE.io Perpetuals &nbsp;·&nbsp; Running 24/7
      </div>
    </div>''', unsafe_allow_html=True)
    time.sleep(30)
    st.rerun()
