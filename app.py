import streamlit as st
import asyncio
import pandas as pd
import requests, time, csv, os, json
from datetime import datetime, timezone, timedelta

@st.cache_resource(show_spinner=False)
def _load_heavy():
    import ccxt.async_support as ccxt_mod
    import pandas_ta as ta_mod
    return ccxt_mod, ta_mod

ccxt, ta = _load_heavy()

try:
    import nest_asyncio
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed(): loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    except RuntimeError:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    nest_asyncio.apply(loop)
except: pass

st.set_page_config(page_title="APEX // Pump & Dump Scanner", page_icon="🔥", layout="wide", initial_sidebar_state="expanded")

_SS_DEFAULTS = {
    'results':[], 'last_scan':"—", 'scan_count':0, 'btc_price':0, 'btc_trend':"—",
    'fng_val':50, 'fng_txt':"Neutral", 'scan_errors':[], 'last_raw_count':0,
    'logged_sigs':set(), 'sentinel_active':False, 'sentinel_results':[],
    'sentinel_last_check':'—', 'sentinel_total_checked':0, 'sentinel_signals_found':0,
    'sentinel_universe_size':'?', 'social_cache':{}, 'social_last_fetch':0,
    'listing_cache':{}, 'listing_last_fetch':0, 'onchain_cache':{},
    'journal_last_autocheck':0, 'prev_results':{}, 'alerted_scores':{},
    'scan_modes':['mixed'], 'fng_last_fetch':0, 'alerted_sigs':set(),
    'last_daily_summary':0, 'nav_state':'🔥 Scanner',
}
for k,v in _SS_DEFAULTS.items():
    if k not in st.session_state: st.session_state[k] = v

JOURNAL_FILE="trade_journal.csv"; COOLDOWN_FILE="symbol_cooldowns.json"; SETTINGS_FILE="apex_settings.json"

DEFAULT_SETTINGS = {
    "scan_depth":40,"scan_modes":["mixed"],"fast_tf":"15m","slow_tf":"4h","min_score":10,
    "j_imminent":True,"j_building":True,"j_early":False,
    "j_filter_classes":["squeeze","breakout","whale_driven","early","god_tier"],
    "j_min_score":25,"j_require_technicals":[],
    "whale_min_usdt":250000,"btc_filter":True,"cooldown_on":True,"cooldown_hrs":4,
    "auto_scan":False,"auto_interval":5,"alert_min_score":60,
    "alert_longs":True,"alert_shorts":True,"alert_squeeze":True,
    "alert_breakout":True,"alert_early":False,"alert_whale":True,
    "cls_breakout_oi_min":7,"cls_breakout_vol_min":4,"cls_breakout_score_min":25,
    "cls_squeeze_fund_min":8,"cls_squeeze_ob_min":6,
    "ob_ratio_low":1.1,"ob_ratio_mid":1.5,"ob_ratio_high":2.5,
    "funding_low":0.0002,"funding_mid":0.0005,"funding_high":0.001,
    "oi_price_chg_low":0.5,"oi_price_chg_high":2.0,
    "vol_surge_low":1.5,"vol_surge_mid":2.0,"vol_surge_high":3.0,
    "liq_cluster_near":1.5,"liq_cluster_mid":3.0,"liq_cluster_far":5.0,
    "vol24h_low":1000000,"vol24h_mid":10000000,"vol24h_high":50000000,
    "rsi_oversold":40,"rsi_overbought":60,"min_reasons":1,"min_rr":1.5,
    "require_momentum":False,
    "pts_ob_low":5,"pts_ob_mid":15,"pts_ob_high":25,
    "pts_funding_low":5,"pts_funding_mid":15,"pts_funding_high":25,
    "pts_oi_low":7,"pts_oi_high":18,"pts_vol_low":3,"pts_vol_mid":8,"pts_vol_high":14,
    "pts_liq_near":15,"pts_liq_mid":8,"pts_liq_far":4,
    "pts_vol24_low":3,"pts_vol24_mid":6,"pts_vol24_high":12,
    "pts_macd":4,"pts_rsi":5,"pts_bb":4,"pts_ema":3,"pts_session":4,
    "pts_sentiment":20,"pts_taker":10,
    "pts_whale_near":25,"pts_whale_mid":15,"pts_whale_far":8,
    "cmc_key":"","tg_token":"","tg_chat_id":"",
    "discord_webhook":"https://discord.com/api/webhooks/1476606856599179265/74wKbIJEXNJ9h10Ab0Q9Vp7ZmeJ52XY18CP3lKxg3eR1BbpZSdX65IT8hbZjpEIXSqEg",
    "okx_key":"","okx_secret":"","okx_passphrase":"","gate_key":"","gate_secret":"",
    "min_vol_filter":300000,"min_active_signals":3,"spread_max_pct":0.5,
    "atr_min_pct":0.2,"atr_max_pct":10.0,"mtf_confirm":True,
    "pts_mtf":12,"pts_divergence":10,"pts_candle_pattern":8,"pts_oi_funding_combo":10,
    "dedup_symbols":True,"fng_long_threshold":30,"fng_short_threshold":70,
    "vol_surge_explosive":5.0,"pts_vol_explosive":20,"pts_orderflow":12,
    "orderflow_lookback":10,"pts_liq_map":15,"listing_alert_pts":25,
    "onchain_whale_min":500000,"pts_onchain_whale":15,
    "journal_autocheck_on":True,"journal_autocheck_mins":15,
    "late_entry_chg_thresh":8.0,"late_entry_penalty":20,"vol_exhaust_penalty":15,
    "near_top_penalty":15,"use_4h_ob_for_sl":True,
    "sentinel_score_threshold":70,"sentinel_batch_size":5,"sentinel_check_interval":30,
    "daily_summary_hour":8,"daily_summary_on":True,
    "social_enabled":True,"social_reddit_weight":8,"social_min_mentions":3,
    "social_buzz_threshold":10,"apify_token":"",
    "backtest_min_score":50,"backtest_days":30,
}

def load_settings():
    s=DEFAULT_SETTINGS.copy()
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE,'r',encoding='utf-8') as f: s.update(json.load(f))
        except: pass
    return s

def save_settings(s):
    with open(SETTINGS_FILE,'w',encoding='utf-8') as f: json.dump(s,f,indent=2)

S = load_settings()

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def ensure_journal():
    headers=["ts","symbol","exchange","type","pump_score","class","price","tp","sl","triggers","status","entry_touched","tp1","tp2","tp3"]
    if not os.path.exists(JOURNAL_FILE):
        with open(JOURNAL_FILE,'w',newline='',encoding='utf-8') as f: csv.writer(f).writerow(headers)
    else:
        try:
            df=pd.read_csv(JOURNAL_FILE); updated=False
            for col,default in [('status','ACTIVE'),('exchange','MEXC'),('entry_touched','0'),
                                  ('tp1','0'),('tp2','0'),('tp3','0')]:
                if col not in df.columns: df[col]=default; updated=True
            if updated: df.to_csv(JOURNAL_FILE,index=False)
        except: pass

def log_trade(res, force=False):
    """Log trade — respects journal filter settings unless force=True"""
    s = load_settings()
    if not force:
        cls = res.get('cls','early')
        score = res.get('pump_score',0)
        j_classes = s.get('j_filter_classes', list(DEFAULT_SETTINGS['j_filter_classes']))
        j_min = s.get('j_min_score', 25)
        j_req_tech = s.get('j_require_technicals', [])
        # Class filter
        if 'god_tier' in j_classes and score >= 90: pass
        elif cls not in j_classes: return
        # Score filter
        if score < j_min: return
        # Technical checklist
        bd = res.get('signal_breakdown', {})
        tech_map = {
            'ob': 'ob_imbalance', 'funding': 'funding', 'oi': 'oi_spike',
            'volume': 'vol_surge', 'whale': 'whale_wall', 'sentiment': 'sentiment',
            'mtf': 'mtf', 'orderflow': 'orderflow'
        }
        for req in j_req_tech:
            key = tech_map.get(req)
            if key and bd.get(key, 0) <= 0: return
    try:
        ensure_journal()
        with open(JOURNAL_FILE,'a',newline='',encoding='utf-8') as f:
            entry_lo = res.get('entry_lo', res['price'])
            entry_hi = res.get('entry_hi', res['price'])
            already_in_zone = entry_lo <= res['price'] <= entry_hi
            entry_touched_val = "1" if already_in_zone else "0"
            csv.writer(f).writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                res['symbol'], res.get('exchange','MEXC'), res['type'],
                res['pump_score'], res.get('cls','—'), round(res['price'],8),
                round(res.get('tp',0),8), round(res.get('sl',0),8),
                " | ".join(res['reasons']), "ACTIVE", entry_touched_val,
                round(res.get('tp1',res.get('tp',0)),8),
                round(res.get('tp2',res.get('tp',0)),8),
                round(res.get('tp3',res.get('tp',0)),8),
            ])
    except: pass

def _journal_check_hits(df, prices, s):
    updated=False; hits=[]
    for i, row in df.iterrows():
        if row.get('status') != 'ACTIVE': continue
        sym=str(row.get('symbol','')); price=prices.get(sym,0)
        if not price: continue
        try: tp=float(row['tp']); sl=float(row['sl']); entry_px=float(row['price'])
        except: continue
        # FIX: Only start tracking after entry zone is touched
        entry_touched = str(row.get('entry_touched','0')) == '1'
        if not entry_touched:
            # Check if price is within 0.5% of entry price
            if abs(price - entry_px) / entry_px * 100 <= 0.5:
                df.at[i,'entry_touched'] = '1'; updated=True; entry_touched=True
            else: continue
        sig=row.get('type','LONG'); hit=None
        if sig=='LONG':
            if price>=tp: hit='TP'
            elif price<=sl: hit='SL'
        else:
            if price<=tp: hit='TP'
            elif price>=sl: hit='SL'
        if hit:
            df.at[i,'status']=hit; updated=True
            hits.append({'symbol':sym,'hit':hit,'price':price,'tp':tp,'sl':sl,'type':sig})
    return df,hits,updated

def _fire_journal_alerts(hits, s, source="Scan"):
    for h in hits:
        em="✅" if h['hit']=='TP' else "🛑"
        st.toast(f"{em} [{source}] {h['symbol']} {h['hit']} @ ${h['price']:.6f}", icon=em)

def process_journal_tracking(tickers_dict, s):
    if not os.path.exists(JOURNAL_FILE): return
    try:
        df=pd.read_csv(JOURNAL_FILE)
        if df.empty or 'status' not in df.columns: return
        prices={}
        for t,data in tickers_dict.items():
            base=t.split('/')[0].split(':')[0]; px=float(data.get('last') or 0)
            if px: prices[base]=px
        df,hits,updated=_journal_check_hits(df,prices,s)
        if updated: df.to_csv(JOURNAL_FILE,index=False)
        if hits: _fire_journal_alerts(hits,s,"Scan")
    except: pass

def autocheck_journal_background(s):
    if not s.get('journal_autocheck_on',True): return
    if not os.path.exists(JOURNAL_FILE): return
    interval=s.get('journal_autocheck_mins',15)*60
    if time.time()-st.session_state.get('journal_last_autocheck',0)<interval: return
    st.session_state.journal_last_autocheck=time.time()
    try:
        df=pd.read_csv(JOURNAL_FILE)
        if df.empty or 'status' not in df.columns: return
        active_syms=df[df['status']=='ACTIVE']['symbol'].dropna().unique().tolist()
        if not active_syms: return
        prices={}
        for sym in active_syms[:5]:
            try:
                r=requests.get("https://www.okx.com/api/v5/market/ticker",
                    params={"instId":f"{sym}-USDT-SWAP"},timeout=1)
                if r.status_code==200:
                    d=r.json().get('data',[])
                    if d: prices[sym]=float(d[0].get('last',0) or 0)
            except: pass
        df,hits,updated=_journal_check_hits(df,prices,s)
        if updated: df.to_csv(JOURNAL_FILE,index=False)
        if hits: _fire_journal_alerts(hits,s,"Auto-Check")
    except: pass

def check_daily_summary(s):
    """Fire daily journal summary to Discord/Telegram once per 24h"""
    if not s.get('daily_summary_on', True): return
    last=st.session_state.get('last_daily_summary',0)
    if time.time()-last < 82800: return  # ~23h guard
    now=datetime.now(timezone.utc)
    target_h=s.get('daily_summary_hour',8)
    if now.hour != target_h: return
    st.session_state.last_daily_summary=time.time()
    try:
        if not os.path.exists(JOURNAL_FILE): return
        df=pd.read_csv(JOURNAL_FILE)
        if df.empty: return
        df['ts']=pd.to_datetime(df['ts'],errors='coerce')
        df24=df[df['ts']>=(datetime.now()-timedelta(hours=24))]
        total=len(df24); longs=len(df24[df24['type']=='LONG']); shorts=len(df24[df24['type']=='SHORT'])
        tps=len(df24[df24['status']=='TP']); sls=len(df24[df24['status']=='SL'])
        active=len(df24[df24['status']=='ACTIVE'])
        wr=(tps/(tps+sls)*100) if (tps+sls)>0 else 0
        msg=(f"📊 **APEX 24h Journal Summary**\n"
             f"🕐 {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
             f"━━━━━━━━━━━━━━━━━\n"
             f"Total Signals: **{total}** | 📗 Long: **{longs}** | 📕 Short: **{shorts}**\n"
             f"✅ TP Hits: **{tps}** | 🛑 SL Hits: **{sls}** | 🔄 Active: **{active}**\n"
             f"🎯 Win Rate: **{wr:.1f}%**\n"
             f"━━━━━━━━━━━━━━━━━\n"
             f"*Powered by APEX Intelligence Terminal*")
        tg_msg=msg.replace("**","<b>").replace("**","</b>")
        if s.get('tg_token') and s.get('tg_chat_id'):
            send_tg(s['tg_token'],s['tg_chat_id'],tg_msg.replace("**","").replace("*",""))
        if s.get('discord_webhook'):
            send_discord(s['discord_webhook'],{
                "title":"📊 APEX 24h Journal Summary",
                "color":0x2563eb,
                "description":msg,
                "footer":{"text":f"APEX Terminal • {now.strftime('%H:%M UTC')}"}
            })
    except: pass

def is_on_cooldown(sym,hrs):
    if not os.path.exists(COOLDOWN_FILE): return False
    try:
        with open(COOLDOWN_FILE,'r',encoding='utf-8') as f: cd=json.load(f)
        if sym in cd: return (datetime.now()-datetime.fromisoformat(cd[sym])).total_seconds()/3600<hrs
    except: pass
    return False

def set_cooldown(sym):
    cd={}
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE,'r',encoding='utf-8') as f: cd=json.load(f)
        except: pass
    cd[sym]=datetime.now().isoformat()
    with open(COOLDOWN_FILE,'w',encoding='utf-8') as f: json.dump(cd,f)

def get_session():
    h=datetime.now(timezone.utc).hour
    if 12<=h<13: return "London/NY Overlap",1.5,"#7c3aed"
    elif 13<=h<17: return "New York Open",1.4,"#059669"
    elif 8<=h<12: return "London Open",1.3,"#2563eb"
    elif 0<=h<4: return "Asian Session",0.85,"#d97706"
    else: return "Off-Hours",0.7,"#9ca3af"

def fmt(n):
    if abs(n)>=1e9: return f"${n/1e9:.2f}B"
    if abs(n)>=1e6: return f"${n/1e6:.2f}M"
    if abs(n)>=1e3: return f"${n/1e3:.1f}K"
    return f"${n:.0f}"

def pump_color(score, is_sniper=False):
    if is_sniper or score>=90: return "#ff0000"
    if score>=70: return "#dc2626"
    if score>=45: return "#d97706"
    if score>=25: return "#2563eb"
    return "#6b7280"

def pump_label(score, sig, is_sniper=False):
    if is_sniper or score>=90: return "🎯 GOD-TIER SETUP"
    if score>=70: return "🔥 PUMP IMMINENT" if sig=="LONG" else "🩸 DUMP IMMINENT"
    if score>=45: return "⚡ BUILDING PUMP" if sig=="LONG" else "⚡ BUILDING DUMP"
    if score>=25: return "📡 EARLY LONG" if sig=="LONG" else "📡 EARLY SHORT"
    return "— WEAK"

def classify(res):
    bd=res.get('signal_breakdown',{}); sc=res['pump_score']; cfg=res.get('_cls_cfg',{})
    br_oi=cfg.get('breakout_oi_min',7); br_vol=cfg.get('breakout_vol_min',4)
    br_sc=cfg.get('breakout_sc_min',25); sq_fd=cfg.get('squeeze_fund_min',8); sq_ob=cfg.get('squeeze_ob_min',6)
    squeeze_score=bd.get('funding',0)+bd.get('funding_hist',0)+bd.get('ob_imbalance',0)
    breakout_score=bd.get('oi_spike',0)+bd.get('vol_surge',0)+bd.get('orderflow',0)
    whale_score=bd.get('whale_wall',0)+bd.get('liq_cluster',0)
    if bd.get('funding',0)>=sq_fd and bd.get('ob_imbalance',0)>=sq_ob and sc>=25: return 'squeeze'
    if bd.get('oi_spike',0)>=br_oi and bd.get('vol_surge',0)>=br_vol and sc>=br_sc: return 'breakout'
    if bd.get('vol_surge',0)>=8 and bd.get('orderflow',0)>=6 and sc>=30: return 'breakout'
    if bd.get('whale_wall',0)>=8 or (bd.get('liq_cluster',0)>=8 and bd.get('vol_surge',0)>=3): return 'whale_driven'
    if sc>=90:
        return max({'squeeze':squeeze_score,'breakout':breakout_score,'whale_driven':whale_score},
                   key=lambda k:{'squeeze':squeeze_score,'breakout':breakout_score,'whale_driven':whale_score}[k])
    if sc>=70:
        if squeeze_score>=breakout_score and squeeze_score>=whale_score: return 'squeeze'
        if breakout_score>=squeeze_score: return 'breakout'
    return 'early'

def send_tg(token,cid,msg):
    if not token or not cid: return
    try:
        r=requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id":cid,"text":msg,"parse_mode":"HTML"},timeout=8)
        if r.status_code!=200: st.toast(f"⚠️ Telegram failed: {r.status_code}",icon="⚠️")
    except Exception as e: st.toast(f"⚠️ Telegram error: {e}",icon="⚠️")

def send_discord(webhook_url,embed_dict):
    if not webhook_url: return
    try:
        r=requests.post(webhook_url,json={"embeds":[embed_dict]},timeout=8)
        if r.status_code not in (200,204): st.toast(f"⚠️ Discord failed: {r.status_code}",icon="⚠️")
    except Exception as e: st.toast(f"⚠️ Discord error: {e}",icon="⚠️")

# ─── CSS (condensed) ──────────────────────────────────────────────────────────
st.markdown("""<style>
:root{--bg:#f7f8fc;--surface:#ffffff;--panel:#f0f2f8;--border:#e2e5f0;--border2:#c8cde0;
--text:#0f1117;--text2:#3d4461;--muted:#7a82a0;--green:#059669;--green-bg:#ecfdf5;
--green-bd:#a7f3d0;--red:#dc2626;--red-bg:#fef2f2;--red-bd:#fecaca;--amber:#d97706;
--amber-bg:#fffbeb;--amber-bd:#fde68a;--blue:#2563eb;--blue-bg:#eff6ff;--blue-bd:#bfdbfe;
--purple:#7c3aed;--purple-bg:#f5f3ff;--purple-bd:#ddd6fe;
--sh:0 1px 4px rgba(15,17,23,.06),0 4px 16px rgba(15,17,23,.04);
--sh-lg:0 8px 32px rgba(15,17,23,.10),0 2px 8px rgba(15,17,23,.06);}
*,*::before,*::after{box-sizing:border-box;}
html,body,.stApp{background:var(--bg)!important;font-family:sans-serif!important;color:var(--text)!important;}
#MainMenu,footer,.stDeployButton{display:none!important;}
header{background-color:transparent!important;}
section[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
section[data-testid="stSidebar"] *{color:var(--text)!important;}
section[data-testid="stSidebar"] .stMarkdown h3{font-family:monospace!important;font-size:0.58rem!important;letter-spacing:.15em!important;color:var(--muted)!important;text-transform:uppercase!important;border-bottom:1px solid var(--border)!important;padding-bottom:5px!important;margin-bottom:10px!important;}
div.stButton>button:first-child{background:var(--text)!important;color:#fff!important;font-family:monospace!important;font-size:0.72rem!important;font-weight:600!important;letter-spacing:.1em!important;text-transform:uppercase!important;border:none!important;border-radius:6px!important;padding:13px 24px!important;transition:all .18s ease!important;}
div.stButton>button:first-child:hover{background:var(--blue)!important;transform:translateY(-1px)!important;box-shadow:0 4px 16px rgba(37,99,235,.28)!important;}
div[data-testid="metric-container"]{background:var(--surface)!important;border:1px solid var(--border)!important;border-radius:8px!important;padding:12px!important;}
div[data-testid="metric-container"] label{font-family:monospace!important;font-size:.55rem!important;letter-spacing:.1em!important;text-transform:uppercase!important;color:var(--muted)!important;}
div[data-testid="metric-container"] div[data-testid="metric-value"]{font-family:monospace!important;font-size:1rem!important;font-weight:600!important;color:var(--text)!important;}
.stTabs [data-baseweb="tab-list"]{background:transparent!important;border-bottom:2px solid var(--border)!important;gap:0!important;}
.stTabs [data-baseweb="tab"]{font-family:monospace!important;font-size:.65rem!important;letter-spacing:.08em!important;font-weight:600!important;color:var(--muted)!important;padding:11px 20px!important;border-bottom:2px solid transparent!important;text-transform:uppercase!important;}
.stTabs [aria-selected="true"]{color:var(--text)!important;border-bottom:2px solid var(--text)!important;background:transparent!important;}
.stProgress>div>div{background:var(--text)!important;}
::-webkit-scrollbar{width:4px;height:4px;}::-webkit-scrollbar-thumb{background:var(--border2);border-radius:2px;}
.ticker-bar{background:#0f1117;color:#fff;border-radius:8px;padding:10px 20px;display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:18px;font-family:monospace;font-size:.68rem;}
.t-lbl{color:rgba(255,255,255,.38);font-size:.54rem;letter-spacing:.1em;text-transform:uppercase;}
.t-val{color:#fff;font-weight:600;}
.pump-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin-bottom:10px;transition:box-shadow .18s,border-color .18s;position:relative;overflow:hidden;}
.pump-card:hover{box-shadow:var(--sh-lg);border-color:var(--border2);}
.pump-card::before{content:'';position:absolute;top:0;left:0;width:4px;height:100%;border-radius:10px 0 0 10px;}
.pc-long::before{background:var(--green);}.pc-short::before{background:var(--red);}
.score-ring{width:54px;height:54px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:monospace;font-size:.95rem;font-weight:700;border:3px solid;flex-shrink:0;}
.px-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0;}
.px-cell{background:var(--panel);border-radius:6px;padding:8px 10px;text-align:center;}
.px-lbl{font-family:monospace;font-size:.52rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:3px;}
.px-val{font-family:monospace;font-size:.8rem;font-weight:600;color:var(--text);}
.sig-pips{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0;}
.pip-item{display:flex;align-items:center;gap:5px;font-family:monospace;font-size:.6rem;color:var(--muted);}
.pip{width:7px;height:7px;border-radius:2px;}.pip-on{background:var(--text);}.pip-half{background:var(--border2);}.pip-off{background:var(--panel);border:1px solid var(--border);}
.reasons-list .r{font-size:.74rem;color:var(--text2);padding:4px 0;border-bottom:1px solid var(--border);}
.tab-desc{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:12px 16px;margin-bottom:14px;font-size:.78rem;color:var(--text2);line-height:1.5;}
.stat-strip{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:12px 18px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:14px;}
.ss-val{font-family:monospace;font-size:1.3rem;font-weight:700;color:var(--text);line-height:1;}
.ss-lbl{font-family:monospace;font-size:.52rem;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;margin-top:3px;}
.empty-st{text-align:center;padding:50px 20px;color:var(--muted);font-family:monospace;font-size:.68rem;letter-spacing:.1em;}
.section-h{font-family:monospace;font-size:.58rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);margin-bottom:10px;padding-bottom:6px;border-bottom:1px solid var(--border);}
.stg-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px 20px;margin-bottom:14px;}
.stg-title{font-family:monospace;font-size:.62rem;font-weight:700;letter-spacing:.15em;text-transform:uppercase;color:var(--text);margin-bottom:14px;padding-bottom:8px;border-bottom:1px solid var(--border);}
.hint{font-size:.72rem;color:var(--muted);line-height:1.4;margin-top:2px;padding:6px 10px;background:var(--panel);border-radius:5px;border-left:3px solid var(--border2);}
.hint b{color:var(--text2);}
.setting-help{font-size:.65rem;color:#7c3aed;background:#f5f3ff;border-left:3px solid #7c3aed;padding:4px 8px;border-radius:4px;margin-top:3px;line-height:1.4;}
.sentiment-bar{display:flex;align-items:center;gap:10px;margin:8px 0;padding:8px 12px;background:var(--panel);border-radius:6px;font-family:monospace;font-size:.62rem;}
.sbar-label{color:var(--muted);width:90px;flex-shrink:0;}
.sbar-track{flex:1;height:8px;background:var(--border);border-radius:4px;overflow:hidden;position:relative;}
.sbar-fill{height:100%;border-radius:4px;transition:width .3s;}
.sbar-val{color:var(--text);font-weight:600;width:40px;text-align:right;}
.momentum-badge{display:inline-flex;align-items:center;gap:4px;padding:3px 8px;border-radius:4px;font-family:monospace;font-size:.58rem;font-weight:700;letter-spacing:.06em;}
.dual-confirm{background:linear-gradient(135deg,#ff6b0015,#dc262615);border:2px solid #dc2626;border-radius:8px;padding:6px 14px;font-family:monospace;font-size:.7rem;font-weight:700;color:#dc2626;margin:6px 0;text-align:center;}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ENGINE
# ═══════════════════════════════════════════════════════════════════════════
class PrePumpScreener:
    def __init__(self, cmc_key="", okx_key="", okx_secret="", okx_passphrase="", gate_key="", gate_secret=""):
        self.cmc_key=cmc_key
        okx_p={'enableRateLimit':True,'rateLimit':50,'timeout':8000,'options':{'defaultType':'swap'}}
        if okx_key and okx_secret and okx_passphrase:
            okx_p.update({'apiKey':okx_key,'secret':okx_secret,'password':okx_passphrase})
        self.okx=ccxt.okx(okx_p)
        self.mexc=ccxt.mexc({'enableRateLimit':True,'rateLimit':60,'timeout':8000,'options':{'defaultType':'swap'}})
        gate_p={'enableRateLimit':True,'rateLimit':50,'timeout':8000,'options':{'defaultType':'swap'}}
        if gate_key and gate_secret: gate_p.update({'apiKey':gate_key,'secret':gate_secret})
        self.gate=ccxt.gateio(gate_p)

    async def fetch_ohlcv(self,exch,sym,tf,n=200):
        try:
            raw=await exch.fetch_ohlcv(sym,tf,limit=n)
            if not raw: return pd.DataFrame()
            df=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
            df['ts']=pd.to_datetime(df['ts'],unit='ms')
            return df
        except: return pd.DataFrame()

    async def fetch_btc(self):
        try:
            df=await self.fetch_ohlcv(self.okx,"BTC/USDT:USDT","1h",80)
            if df.empty: df=await self.fetch_ohlcv(self.gate,"BTC/USDT:USDT","1h",80)
            if df.empty: df=await self.fetch_ohlcv(self.mexc,"BTC/USDT:USDT","1h",80)
            if df.empty: return "NEUTRAL",0,50
            df.ta.ema(length=20,append=True); df.ta.ema(length=50,append=True); df.ta.rsi(length=14,append=True)
            e20=[c for c in df.columns if 'EMA_20' in c]; e50=[c for c in df.columns if 'EMA_50' in c]
            rc=[c for c in df.columns if 'RSI' in c]
            if not e20 or not e50 or not rc: return "NEUTRAL",df['close'].iloc[-1],50
            p=df['close'].iloc[-1]; r=df[rc[0]].iloc[-1]
            if p<df[e20[0]].iloc[-1] and p<df[e50[0]].iloc[-1] and df[e20[0]].iloc[-1]<df[e50[0]].iloc[-1] and r<45:
                return "BEARISH",p,r
            elif p>df[e20[0]].iloc[-1] and p>df[e50[0]].iloc[-1] and df[e20[0]].iloc[-1]>df[e50[0]].iloc[-1] and r>55:
                return "BULLISH",p,r
            return "NEUTRAL",p,r
        except: return "NEUTRAL",0,50

    async def safe_fetch(self,exch,sym):
        fi={}; tick={}; ob={'bids':[],'asks':[]}
        try: fi=await exch.fetch_funding_rate(sym)
        except: fi={'fundingRate':0}
        try: tick=await exch.fetch_ticker(sym)
        except: pass
        try: ob=await exch.fetch_order_book(sym,limit=100)
        except: pass
        return fi,tick,ob

    async def fetch_oi(self,exch,sym):
        try:
            oi=await exch.fetch_open_interest(sym)
            return float(oi.get('openInterestValue') or oi.get('openInterest') or 0)
        except: return 0.0

    async def fetch_funding_history(self,exch,sym):
        try:
            h=await exch.fetch_funding_rate_history(sym,limit=24)
            return [float(x.get('fundingRate',0)) for x in h if x.get('fundingRate') is not None]
        except: return []

    async def fetch_oi_history(self,exch,sym):
        try:
            h=await exch.fetch_open_interest_history(sym,"1h",limit=24)
            return [float(x.get('openInterestValue',0) or x.get('openInterest',0)) for x in h]
        except: return []

    async def fetch_recent_trades(self,exch,sym,min_usdt=50000):
        try:
            trades=await exch.fetch_trades(sym,limit=100)
            wb=[]; ws=[]
            for t in trades:
                cost=float(t.get('cost',0) or 0); side=t.get('side',''); price=float(t.get('price',0) or 0)
                if cost>=min_usdt:
                    (wb if side=='buy' else ws).append({'cost':cost,'price':price})
            return wb,ws
        except: return [],[]

    async def fetch_orderflow_imbalance(self,exch,sym,lookback=10):
        try:
            df=await self.fetch_ohlcv(exch,sym,"5m",lookback+5)
            if df.empty or len(df)<lookback: return 0.0,'NEUTRAL'
            r=df.tail(lookback)
            bv=float(r[r['close']>r['open']]['volume'].sum())
            sv=float(r[r['close']<=r['open']]['volume'].sum())
            total=bv+sv
            if total==0: return 0.0,'NEUTRAL'
            bp=bv/total*100
            return bp,('BUY' if bp>55 else ('SELL' if bp<45 else 'NEUTRAL'))
        except: return 0.0,'NEUTRAL'

    async def fetch_4h_ob_and_fvg(self, exch, sym, price):
        """
        Fetch 4H Order Blocks AND Fair Value Gaps for HTF SL placement.
        OB: strong momentum candle (body>55% range) with move-away confirmation.
        FVG: gap between candle[i-1].high and candle[i+1].low (bullish) or vice versa (bearish).
        """
        bull_obs=[]; bear_obs=[]; bull_fvgs=[]; bear_fvgs=[]
        try:
            df=await self.fetch_ohlcv(exch,sym,"4h",60)
            if df.empty or len(df)<10: return bull_obs,bear_obs,bull_fvgs,bear_fvgs
            closes=df['close'].values; opens=df['open'].values
            highs=df['high'].values; lows=df['low'].values

            # ── Order Blocks ───────────────────────────────────────────────
            for i in range(len(df)-5,5,-1):
                c_open=float(opens[i]); c_close=float(closes[i])
                c_high=float(highs[i]); c_low=float(lows[i])
                c_range=c_high-c_low
                if c_range<=0: continue
                body=abs(c_close-c_open); body_pct=body/c_range
                if body_pct<0.55: continue
                if i+2>=len(df): continue
                if c_close>c_open:  # Bullish OB
                    move=(float(highs[i+2])-c_close)/c_close*100
                    if move>=0.8:
                        oh=c_high; ol=max(c_open,c_low)
                        if price>ol and price>oh*0.98:
                            dist=(price-oh)/price*100
                            if dist<=15: bull_obs.append({'zone_hi':oh,'zone_lo':ol,'dist_pct':dist,'body_pct':round(body_pct,2),'tf':'4H','type':'OB'})
                else:  # Bearish OB
                    move=(c_close-float(lows[i+2]))/c_close*100
                    if move>=0.8:
                        ol=c_low; oh=min(c_open,c_high)
                        if price<oh and price<ol*1.02:
                            dist=(ol-price)/price*100
                            if dist<=15: bear_obs.append({'zone_hi':oh,'zone_lo':ol,'dist_pct':dist,'body_pct':round(body_pct,2),'tf':'4H','type':'OB'})

            # ── Fair Value Gaps ────────────────────────────────────────────
            # Bullish FVG: candle[i-1].high < candle[i+1].low (gap up — support)
            # Bearish FVG: candle[i-1].low > candle[i+1].high (gap down — resistance)
            for i in range(2, len(df)-1):
                prev_h=float(highs[i-1]); prev_l=float(lows[i-1])
                next_h=float(highs[i+1]); next_l=float(lows[i+1])
                # Bullish FVG: gap between prev candle high and next candle low
                if prev_h < next_l:
                    fvg_lo=prev_h; fvg_hi=next_l; mid=(fvg_lo+fvg_hi)/2
                    if price>fvg_lo:
                        dist=(price-fvg_hi)/price*100 if price>fvg_hi else 0
                        if dist<=12:
                            bull_fvgs.append({'zone_hi':fvg_hi,'zone_lo':fvg_lo,'mid':mid,'dist_pct':dist,'tf':'4H','type':'FVG'})
                # Bearish FVG: gap between next candle high and prev candle low
                if prev_l > next_h:
                    fvg_hi=prev_l; fvg_lo=next_h; mid=(fvg_lo+fvg_hi)/2
                    if price<fvg_hi:
                        dist=(fvg_lo-price)/price*100 if price<fvg_lo else 0
                        if dist<=12:
                            bear_fvgs.append({'zone_hi':fvg_hi,'zone_lo':fvg_lo,'mid':mid,'dist_pct':dist,'tf':'4H','type':'FVG'})

            bull_obs.sort(key=lambda x:x['dist_pct']); bear_obs.sort(key=lambda x:x['dist_pct'])
            bull_fvgs.sort(key=lambda x:x['dist_pct']); bear_fvgs.sort(key=lambda x:x['dist_pct'])
        except: pass
        return bull_obs,bear_obs,bull_fvgs,bear_fvgs

    async def fetch_liquidation_map(self,exch,sym,price):
        clusters=[]
        try:
            dsym_c=sym.split(':')[0].replace('/USDT','')
            r=requests.get("https://www.okx.com/api/v5/public/liquidation-orders",
                params={"instType":"SWAP","instId":f"{dsym_c}-USDT-SWAP","state":"unfilled","limit":"100"},timeout=5)
            if r.status_code==200:
                raw=r.json().get('data',[]); items=raw[0] if raw else []; buckets={}
                for item in items:
                    try:
                        liq_px=float(item.get('bkPx',0)); liq_sz=float(item.get('sz',0))*liq_px
                        side='SHORT_LIQ' if item.get('side')=='sell' else 'LONG_LIQ'
                        if liq_px<=0 or liq_sz<10000: continue
                        bucket=round(liq_px/price,3); key=(bucket,side)
                        if key not in buckets: buckets[key]={'price':liq_px,'side':side,'size_usd':0,'count':0}
                        buckets[key]['size_usd']+=liq_sz; buckets[key]['count']+=1
                    except: pass
                for (bkt,side),data in buckets.items():
                    if data['size_usd']<50000: continue
                    dist=abs(data['price']-price)/price*100
                    clusters.append({**data,'dist_pct':dist})
                clusters.sort(key=lambda x:x['dist_pct'])
        except: pass
        return clusters

    def fetch_new_listings(self,s):
        cache=st.session_state.get('listing_cache',{})
        if cache.get('ts') and time.time()-cache['ts']<1800: return cache.get('data',[])
        listings=[]; now=time.time()
        try:
            r=requests.get("https://www.okx.com/api/v5/public/instruments",params={"instType":"SWAP"},timeout=7)
            if r.status_code==200:
                cutoff_ms=(now-7*86400)*1000
                for inst in r.json().get('data',[]):
                    try:
                        lt=float(inst.get('listTime',0))
                        if lt>=cutoff_ms:
                            sym_c=inst['instId'].replace('-USDT-SWAP','')
                            listings.append({'symbol':sym_c,'exchange':'OKX','listed_ts':lt,'listed_ago_h':(now*1000-lt)/3600000})
                    except: pass
        except: pass
        try:
            r2=requests.get("https://fx-api.gateio.ws/api/v4/futures/usdt/contracts",timeout=7)
            if r2.status_code==200:
                cutoff_s=now-7*86400; existing={l['symbol'] for l in listings}
                for c in r2.json():
                    try:
                        ct=float(c.get('create_time',0))
                        if ct>=cutoff_s:
                            sym_c=c['name'].replace('_USDT','')
                            if sym_c not in existing:
                                listings.append({'symbol':sym_c,'exchange':'GATE','listed_ts':ct*1000,'listed_ago_h':(now-ct)/3600})
                    except: pass
        except: pass
        st.session_state.listing_cache={'ts':now,'data':listings}
        return listings

    def fetch_onchain_whale(self,sym,s):
        min_usd=s.get('onchain_whale_min',500000); cache=st.session_state.get('onchain_cache',{}); now=time.time()
        if sym in cache and now-cache[sym].get('ts',0)<600: return cache[sym]
        result={'available':False,'signal':'NEUTRAL','detail':'','inflow':0,'outflow':0,'ts':now}
        try:
            r=requests.get("https://api.whale-alert.io/v1/transactions",
                params={"api_key":"free","min_value":str(int(min_usd)),"currency":sym.lower().replace('1000',''),
                        "limit":"20","start":str(int(now-3600))},timeout=5)
            if r.status_code==200:
                txns=r.json().get('transactions',[])
                inflow=sum(t.get('amount_usd',0) for t in txns if t.get('to',{}).get('owner_type')=='exchange')
                outflow=sum(t.get('amount_usd',0) for t in txns if t.get('from',{}).get('owner_type')=='exchange')
                if inflow+outflow>=min_usd:
                    net=outflow-inflow
                    result={'available':True,'ts':now,'signal':'BULLISH' if net>0 else 'BEARISH',
                            'detail':f"ExchIn ${inflow/1e6:.1f}M | ExchOut ${outflow/1e6:.1f}M",'inflow':inflow,'outflow':outflow}
        except: pass
        if not result['available']:
            try:
                CG={'BTC':'bitcoin','ETH':'ethereum','SOL':'solana','BNB':'binancecoin','XRP':'ripple',
                    'ADA':'cardano','DOGE':'dogecoin','AVAX':'avalanche-2','LINK':'chainlink',
                    'DOT':'polkadot','MATIC':'matic-network','OP':'optimism','ARB':'arbitrum',
                    'SUI':'sui','APT':'aptos','PEPE':'pepe','WIF':'dogwifcoin','TON':'the-open-network'}
                cg_id=CG.get(sym.upper(),sym.lower())
                r2=requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                    params={"localization":"false","tickers":"false","market_data":"true","developer_data":"false"},timeout=6)
                if r2.status_code==200:
                    md=r2.json().get('market_data',{})
                    vol=float(md.get('total_volume',{}).get('usd',0) or 0)
                    mcp=float(md.get('market_cap',{}).get('usd',0) or 0)
                    c1h=float(md.get('price_change_percentage_1h_in_currency',{}).get('usd',0) or 0)
                    vr=vol/mcp if mcp>0 else 0
                    if vr>0.08 and abs(c1h)>0.8:
                        result={'available':True,'ts':now,'signal':'BULLISH' if c1h>0 else 'BEARISH',
                                'detail':f"Vol/MCap {vr:.2f}x | 1h {c1h:+.1f}% (CG proxy)",'inflow':0,'outflow':0}
            except: pass
        if 'onchain_cache' not in st.session_state: st.session_state.onchain_cache={}
        st.session_state.onchain_cache[sym]=result
        return result

    def fetch_sentiment_data(self,sym_base,exch_name):
        result={'top_long_pct':50.0,'top_short_pct':50.0,'retail_long_pct':50.0,'taker_buy_pct':50.0,'available':False,'source':''}
        try:
            if exch_name=="GATE":
                ccy=sym_base.replace("-USDT-SWAP","").replace("USDT","").replace("_USDT","")
                r1=requests.get("https://fx-api.gateio.ws/api/v4/futures/usdt/contract_stats",
                    params={"contract":f"{ccy}_USDT","interval":"1h","limit":8},timeout=4)
                if r1.status_code==200:
                    rows=r1.json()
                    if rows:
                        lsr=float(rows[-1].get('lsr_account',1.0) or 1.0)
                        lp=(lsr/(1+lsr))*100
                        result.update({'top_long_pct':lp,'top_short_pct':100-lp,'retail_long_pct':lp,'available':True,'source':'Gate'})
                r2=requests.get("https://fx-api.gateio.ws/api/v4/futures/usdt/trades",
                    params={"contract":f"{ccy}_USDT","limit":100},timeout=4)
                if r2.status_code==200:
                    trades=r2.json()
                    bv=sum(abs(float(t.get('size',0))) for t in trades if float(t.get('size',0))>0)
                    sv=sum(abs(float(t.get('size',0))) for t in trades if float(t.get('size',0))<0)
                    total=bv+sv
                    if total>0: result['taker_buy_pct']=(bv/total)*100
            elif exch_name=="OKX":
                r1=requests.get("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
                    params={"ccy":sym_base.replace("-USDT-SWAP","").replace("USDT",""),"period":"1H"},timeout=4)
                if r1.status_code==200:
                    rows=r1.json().get('data',[])
                    if rows:
                        ls_ratio=float(rows[0][1]); lp=(ls_ratio/(ls_ratio+1))*100
                        result.update({'top_long_pct':lp,'top_short_pct':100-lp,'retail_long_pct':lp,'available':True,'source':'OKX'})
                r2=requests.get("https://www.okx.com/api/v5/rubik/stat/taker-volume",
                    params={"ccy":sym_base.replace("-USDT-SWAP","").replace("USDT",""),"instType":"CONTRACTS","period":"5m"},timeout=4)
                if r2.status_code==200:
                    rows=r2.json().get('data',[])[:12]
                    if rows:
                        bv=sum(float(r[1]) for r in rows); sv=sum(float(r[2]) for r in rows)
                        total=bv+sv
                        if total>0: result['taker_buy_pct']=(bv/total)*100
        except: pass
        return result

    def fetch_reddit_buzz(self,coin_sym,s):
        if not s.get('social_enabled',True): return {'mentions':0,'score':0,'available':False,'source':''}
        sym=coin_sym.upper().replace('1000','').replace('10000','')
        cache=st.session_state.get('social_cache',{}); now=time.time()
        if sym in cache and (now-cache[sym].get('ts',0))<300: return cache[sym]
        result={'mentions':0,'score':0,'upvote_avg':0,'available':False,'source':'','sentiment':'NEUTRAL','top_post':'','ts':now}
        max_pts=s.get('social_reddit_weight',8); min_ment=s.get('social_min_mentions',3); buzz_thr=s.get('social_buzz_threshold',10)
        try:
            subs="CryptoCurrency+CryptoMoonShots+SatoshiStreetBets+altcoin+CryptoMarkets"
            rr=requests.get(f"https://www.reddit.com/r/{subs}/search.json",
                params={"q":sym,"sort":"new","t":"hour","limit":25,"restrict_sr":"1"},
                headers={"User-Agent":"Mozilla/5.0 APEX/3.0"},timeout=6)
            if rr.status_code==200:
                posts=rr.json().get('data',{}).get('children',[])
                if posts:
                    mentions=len(posts); upvotes=[p['data'].get('score',0) for p in posts]
                    avg_up=sum(upvotes)/max(1,len(upvotes))
                    bull_kw=['moon','pump','buy','bullish','launch','listing','breakout','gem','surge','ath']
                    bear_kw=['dump','crash','sell','bearish','scam','rug','dead','rekt','fraud','fail']
                    bh=0; brh=0
                    for p in posts:
                        txt=(p['data'].get('title','')+' '+p['data'].get('selftext','')).lower()
                        bh+=sum(1 for w in bull_kw if w in txt); brh+=sum(1 for w in bear_kw if w in txt)
                    sent=('BULLISH' if bh>brh+1 else 'BEARISH' if brh>bh+1 else 'NEUTRAL')
                    sc=min(max_pts,int((mentions/buzz_thr)*max_pts)) if mentions>=min_ment else 0
                    result={'mentions':mentions,'score':sc,'upvote_avg':avg_up,'available':True,'source':'Reddit',
                            'sentiment':sent,'top_post':posts[0]['data'].get('title','')[:60],'ts':now}
        except: pass
        if not result['available']:
            try:
                cg_map={'BTC':'bitcoin','ETH':'ethereum','SOL':'solana','BNB':'binancecoin','XRP':'ripple',
                        'ADA':'cardano','DOGE':'dogecoin','AVAX':'avalanche-2','LINK':'chainlink',
                        'DOT':'polkadot','MATIC':'matic-network','LTC':'litecoin','UNI':'uniswap',
                        'ATOM':'cosmos','XLM':'stellar','NEAR':'near','APT':'aptos','ARB':'arbitrum',
                        'OP':'optimism','INJ':'injective-protocol','SUI':'sui','TIA':'celestia',
                        'PEPE':'pepe','SHIB':'shiba-inu','WIF':'dogwifhat','BONK':'bonk'}
                cg_id=cg_map.get(sym,sym.lower())
                cg=requests.get(f"https://api.coingecko.com/api/v3/coins/{cg_id}",
                    params={"localization":"false","tickers":"false","market_data":"true","community_data":"true","developer_data":"false"},timeout=6)
                if cg.status_code==200:
                    data=cg.json(); cd=data.get('community_data',{}); md=data.get('market_data',{})
                    subs_count=cd.get('reddit_subscribers',0) or 0; active=cd.get('reddit_accounts_active_48h',0) or 0
                    tw_foll=cd.get('twitter_followers',0) or 0
                    c1h=md.get('price_change_percentage_1h_in_currency',{}).get('usd',0) or 0
                    sent=('BULLISH' if c1h>1.5 else 'BEARISH' if c1h<-1.5 else 'NEUTRAL')
                    mp=min(50,int(active/50)) if active else (2 if subs_count>50000 else 0)
                    sc=min(max_pts,int((mp/buzz_thr)*max_pts)) if mp>=min_ment else 0
                    top=(f"r/ {subs_count:,} subs | {active:,} active 48h"+(f" | Twitter {tw_foll:,}" if tw_foll else "")+(f" | 1h {c1h:+.2f}%" if c1h else ""))
                    result={'mentions':mp,'score':sc,'upvote_avg':0,'available':True,'source':'CoinGecko',
                            'sentiment':sent,'top_post':top[:80],'ts':now}
            except: pass
        if 'social_cache' not in st.session_state: st.session_state.social_cache={}
        st.session_state.social_cache[sym]=result
        return result

    def cmc_data(self,sym):
        if not self.cmc_key: return None
        try:
            r=requests.get("https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest",
                headers={"X-CMC_PRO_API_KEY":self.cmc_key},params={"symbol":sym.replace("1000","")},timeout=3)
            if r.status_code==200:
                d=r.json()['data']; coin=d[list(d.keys())[0]]; q=coin['quote']['USD']
                return {'rank':coin.get('cmc_rank',9999),'mcap':q.get('market_cap') or 0,
                        'vol24':q.get('volume_24h') or 0,'change24':q.get('percent_change_24h') or 0}
        except: pass
        return None

    def ob_score_calc(self,bids,asks,sig,s):
        try:
            def valid(levels):
                out=[]
                for lv in levels:
                    try:
                        p,q=float(lv[0]),float(lv[1])
                        if p>0 and q>0: out.append((p,q))
                    except: pass
                return out
            b=valid(bids); a=valid(asks)
            if not b or not a: return 0,"",{'bid_pct':50,'ratio':1.0,'whale_bid_val':0,'whale_ask_val':0,'whale_bid_px':0,'whale_ask_px':0}
            bv=sum(p*q for p,q in b); av=sum(p*q for p,q in a)
            tot=bv+av; bid_pct=(bv/tot*100) if tot>0 else 50; ratio=bv/av if av>0 else 1.0
            wb=max(b,key=lambda x:x[0]*x[1]); wa=max(a,key=lambda x:x[0]*x[1])
            wbv=wb[0]*wb[1]; wav=wa[0]*wa[1]; score=0; msg=""
            if sig=="LONG":
                r=ratio
                if r>=s['ob_ratio_high']:   score=s['pts_ob_high']; msg=f"⚖️ OB {r:.2f}× bid-heavy — strong buy pressure ({bid_pct:.0f}% bid)"
                elif r>=s['ob_ratio_mid']:  score=s['pts_ob_mid'];  msg=f"⚖️ OB {r:.2f}× bid-heavy ({bid_pct:.0f}% bid)"
                elif r>=s['ob_ratio_low']:  score=s['pts_ob_low'];  msg=f"⚖️ OB {r:.2f}× slight bid pressure ({bid_pct:.0f}% bid)"
            else:
                inv=1/ratio if ratio>0 else 1
                if inv>=s['ob_ratio_high']:  score=s['pts_ob_high']; msg=f"⚖️ OB {inv:.2f}× ask-heavy — strong sell pressure ({100-bid_pct:.0f}% ask)"
                elif inv>=s['ob_ratio_mid']: score=s['pts_ob_mid'];  msg=f"⚖️ OB {inv:.2f}× ask-heavy ({100-bid_pct:.0f}% ask)"
                elif inv>=s['ob_ratio_low']: score=s['pts_ob_low'];  msg=f"⚖️ OB {inv:.2f}× slight ask pressure"
            return score,msg,{'bid_pct':bid_pct,'ratio':ratio,'whale_bid_val':wbv,'whale_ask_val':wav,'whale_bid_px':wb[0],'whale_ask_px':wa[0]}
        except: return 0,"",{'bid_pct':50,'ratio':1.0,'whale_bid_val':0,'whale_ask_val':0,'whale_bid_px':0,'whale_ask_px':0}

    def find_whale_walls(self,bids,asks,price,sig,s):
        wmin=s['whale_min_usdt']; whale_sc=0; whale_details=[]; reasons_out=[]
        def parse_walls(levels,side):
            walls=[]
            for lv in levels:
                try:
                    p,q=float(lv[0]),float(lv[1]); val=p*q
                    if val>=wmin: walls.append({'price':p,'value':val,'dist_pct':abs(p-price)/price*100,'side':side})
                except: pass
            return sorted(walls,key=lambda x:x['value'],reverse=True)
        bid_walls=parse_walls(bids,'BID'); ask_walls=parse_walls(asks,'ASK')
        near=s.get('pts_whale_near',25); mid=s.get('pts_whale_mid',15); far=s.get('pts_whale_far',8)
        if sig=="LONG" and bid_walls:
            w=bid_walls[0]
            if w['dist_pct']<=0.5: whale_sc=near
            elif w['dist_pct']<=1.5: whale_sc=mid
            elif w['dist_pct']<=3.0: whale_sc=far
            if whale_sc>0:
                whale_details.append({'side':'BUY','value':w['value'],'price':w['price'],'dist_pct':w['dist_pct']})
                reasons_out.append(f"🐋 BUY WALL {fmt(w['value'])} @ ${w['price']:.6f} ({w['dist_pct']:.2f}% below)")
            if len(bid_walls)>=3: whale_sc+=5; reasons_out.append(f"🐋 {len(bid_walls)} stacked bid walls — layered buy support")
        elif sig=="SHORT" and ask_walls:
            w=ask_walls[0]
            if w['dist_pct']<=0.5: whale_sc=near
            elif w['dist_pct']<=1.5: whale_sc=mid
            elif w['dist_pct']<=3.0: whale_sc=far
            if whale_sc>0:
                whale_details.append({'side':'SELL','value':w['value'],'price':w['price'],'dist_pct':w['dist_pct']})
                reasons_out.append(f"🐋 SELL WALL {fmt(w['value'])} @ ${w['price']:.6f} ({w['dist_pct']:.2f}% above)")
            if len(ask_walls)>=3: whale_sc+=5; reasons_out.append(f"🐋 {len(ask_walls)} stacked ask walls — layered sell resistance")
        whale_str=""
        if whale_details:
            w=whale_details[0]
            whale_str=f"{'BUY' if w['side']=='BUY' else 'SELL'} {fmt(w['value'])} @ ${w['price']:.6f}"
        return whale_sc,whale_str,whale_details,reasons_out

    async def analyze(self,exch_name,exch_obj,sym,s,btc_trend):
        dsym=sym.split(':')[0].replace('/USDT','')
        if s['cooldown_on'] and is_on_cooldown(dsym,s['cooldown_hrs']): return None
        df_f=await self.fetch_ohlcv(exch_obj,sym,s['fast_tf'],200)
        df_slow=await self.fetch_ohlcv(exch_obj,sym,s['slow_tf'],100)
        fi,tick,ob=await self.safe_fetch(exch_obj,sym)
        oi=await self.fetch_oi(exch_obj,sym)
        if df_f.empty or df_slow.empty: return None
        price=float(tick.get('last',0) or 0)
        if price<=0: return None
        try:
            df_slow.ta.ema(length=50,append=True)
            df_f.ta.rsi(length=14,append=True); df_f.ta.atr(length=14,append=True)
            df_f.ta.macd(fast=12,slow=26,signal=9,append=True); df_f.ta.bbands(length=20,std=2,append=True)
            e50_cols=[c for c in df_slow.columns if 'EMA_50' in c or ('EMA' in c and '50' in c)]
            rsi_cols=[c for c in df_f.columns if 'RSI_14' in c or 'RSI' in c]
            atr_cols=[c for c in df_f.columns if 'ATRr_14' in c or 'ATRr' in c]
            macd_cols=[c for c in df_f.columns if c.startswith('MACD_') and not c.startswith('MACDh') and not c.startswith('MACDs')]
            macds_cols=[c for c in df_f.columns if c.startswith('MACDs_')]
            bbl_cols=[c for c in df_f.columns if c.startswith('BBL_')]
            bbu_cols=[c for c in df_f.columns if c.startswith('BBU_')]
            if not e50_cols or not rsi_cols or not atr_cols: return None
            e50=float(df_slow[e50_cols[0]].iloc[-1]); rsi=float(df_f[rsi_cols[0]].iloc[-1])
            atr=float(df_f[atr_cols[0]].iloc[-1]); macd=float(df_f[macd_cols[0]].iloc[-1]) if macd_cols else 0
            macds=float(df_f[macds_cols[0]].iloc[-1]) if macds_cols else 0
            bbl=float(df_f[bbl_cols[0]].iloc[-1]) if bbl_cols else price*0.97
            bbu=float(df_f[bbu_cols[0]].iloc[-1]) if bbu_cols else price*1.03
            vma=float(df_f['volume'].rolling(20).mean().iloc[-1]); lvol=float(df_f['volume'].iloc[-1])
            if vma==0: vma=1
            rsi_series_raw=df_f[rsi_cols[0]].dropna().values
        except: return None

        sig="LONG" if float(df_slow['close'].iloc[-1])>e50 else "SHORT"
        if s['btc_filter'] and btc_trend=="BEARISH" and sig=="LONG": return None
        qv_now=float(tick.get('quoteVolume',0) or 0)
        if s.get('min_vol_filter',300000)>0 and qv_now<s['min_vol_filter']: return None
        atr_pct=(atr/price*100) if price>0 else 0
        if atr_pct<s.get('atr_min_pct',0.2): return None
        if atr_pct>s.get('atr_max_pct',10.0): return None
        spread_max=s.get('spread_max_pct',0.5)
        if spread_max>0:
            bids_raw=ob.get('bids',[]); asks_raw=ob.get('asks',[])
            if bids_raw and asks_raw:
                best_bid=float(bids_raw[0][0]); best_ask=float(asks_raw[0][0])
                if best_bid>0 and (best_ask-best_bid)/best_bid*100>spread_max: return None

        pump_score=0; reasons=[]; bd={}; warnings=[]
        try: pc20=((price-float(df_f['close'].iloc[-20]))/float(df_f['close'].iloc[-20]))*100
        except: pc20=0

        # ── Late Entry Detector ───────────────────────────────────────────
        late_entry_penalty=0; late_entry_flag=False
        late_chg_thresh=s.get('late_entry_chg_thresh',8.0)
        if abs(pc20)>late_chg_thresh:
            try:
                closes=df_f['close'].values
                base_price=float(df_f['close'].iloc[-21]) if len(df_f)>=21 else float(df_f['close'].iloc[0])
                crossed_at=None
                for ci in range(len(closes)-1,max(len(closes)-25,0),-1):
                    if abs((float(closes[ci])-base_price)/base_price*100)<late_chg_thresh*0.5: crossed_at=ci; break
                candles_since=len(closes)-1-(crossed_at or (len(closes)-15))
                if candles_since>10:
                    late_entry_penalty=s.get('late_entry_penalty',20); late_entry_flag=True
                    d="LATE ENTRY" if sig=="LONG" else "LATE SHORT"
                    warnings.append(f"⚠️ {d}: {abs(pc20):.1f}% move was {candles_since} candles ago — entry window likely closed")
            except: pass
        pump_score-=late_entry_penalty; bd['late_entry_penalty']=-late_entry_penalty

        # ── Volume Exhaustion ─────────────────────────────────────────────
        vol_exhaust_penalty=0; vol_exhaust_flag=False
        try:
            vol5_avg=float(df_f['volume'].rolling(5).mean().iloc[-1])
            high20=float(df_f['high'].rolling(20).max().iloc[-1])
            if high20>0 and abs(price-high20)/high20*100<3.0 and vol5_avg>0 and lvol<vol5_avg*0.60 and sig=="LONG":
                vol_exhaust_penalty=s.get('vol_exhaust_penalty',15); vol_exhaust_flag=True
                warnings.append(f"⚠️ DISTRIBUTION PHASE: Volume {lvol/vol5_avg:.1f}× avg near 20c high — smart money selling")
        except: pass
        pump_score-=vol_exhaust_penalty; bd['vol_exhaust_penalty']=-vol_exhaust_penalty

        # ── RSI Direction ─────────────────────────────────────────────────
        rsi_direction='FLAT'; rsi_dir_pts=0; rsi_dir_reason=""
        try:
            if len(rsi_series_raw)>=4:
                r3=rsi_series_raw[-3:]; p3=rsi_series_raw[-6:-3] if len(rsi_series_raw)>=6 else rsi_series_raw[:3]
                rna=float(r3.mean()); rpa=float(p3.mean()); slope=rna-rpa
                if sig=="LONG":
                    if slope>3: rsi_direction='RISING'; rsi_dir_pts=8; rsi_dir_reason=f"📈 RSI rising ({rpa:.1f}→{rna:.1f}) — momentum building UP"
                    elif slope<-3: rsi_direction='FALLING'; rsi_dir_pts=-8; rsi_dir_reason=f"⚠️ RSI declining ({rpa:.1f}→{rna:.1f}) — LONG momentum fading"
                else:
                    if slope<-3: rsi_direction='FALLING'; rsi_dir_pts=8; rsi_dir_reason=f"📉 RSI falling ({rpa:.1f}→{rna:.1f}) — momentum building DOWN"
                    elif slope>3: rsi_direction='RISING'; rsi_dir_pts=-8; rsi_dir_reason=f"⚠️ RSI rising ({rpa:.1f}→{rna:.1f}) — SHORT momentum fading"
        except: pass
        if rsi_dir_pts!=0:
            pump_score+=rsi_dir_pts
            if rsi_dir_reason and rsi_dir_pts>0: reasons.append(rsi_dir_reason)
            elif rsi_dir_pts<0 and abs(rsi_dir_pts)>=8: warnings.append(rsi_dir_reason)
        bd['rsi_direction']=rsi_dir_pts

        # ── Near Local Top Penalty ────────────────────────────────────────
        near_top_penalty=0; near_top_flag=False
        try:
            high20_val=float(df_f['high'].rolling(20).max().iloc[-1])
            if high20_val>0:
                dist_from_top=abs(price-high20_val)/high20_val*100
                md_declining=False; rsi_declining=False
                if macd_cols and len(df_f)>=3:
                    mv=df_f[macd_cols[0]].values
                    md_declining=float(mv[-1])<float(mv[-2])<float(mv[-3])
                if len(rsi_series_raw)>=3:
                    rsi_declining=float(rsi_series_raw[-1])<float(rsi_series_raw[-2])<float(rsi_series_raw[-3])
                if dist_from_top<3.0 and md_declining and rsi_declining and sig=="LONG":
                    near_top_penalty=s.get('near_top_penalty',15); near_top_flag=True
                    warnings.append(f"⚠️ NEAR LOCAL TOP: Price {dist_from_top:.1f}% from 20c high, MACD+RSI declining")
        except: pass
        pump_score-=near_top_penalty; bd['near_top_penalty']=-near_top_penalty

        # 1 ── ORDER BOOK ──────────────────────────────────────────────────
        ob_sc,ob_msg,ob_data=self.ob_score_calc(ob.get('bids',[]),ob.get('asks',[]),sig,s)
        pump_score+=ob_sc; bd['ob_imbalance']=ob_sc
        if ob_msg: reasons.append(ob_msg)

        # 2 ── WHALE WALLS ─────────────────────────────────────────────────
        whale_sc,whale_str,whale_details,whale_reasons=self.find_whale_walls(ob.get('bids',[]),ob.get('asks',[]),price,sig,s)
        pump_score+=whale_sc; bd['whale_wall']=whale_sc; reasons.extend(whale_reasons)

        # 3 ── FUNDING RATE ────────────────────────────────────────────────
        fr=float(fi.get('fundingRate',0) or 0); fr_sc=0
        if sig=="LONG":
            if fr<=-s['funding_high']:   fr_sc=s['pts_funding_high']; reasons.append(f"⚡ Extreme neg funding {fr*100:.4f}% — shorts will be squeezed")
            elif fr<=-s['funding_mid']:  fr_sc=s['pts_funding_mid'];  reasons.append(f"⚡ Strong neg funding {fr*100:.4f}% — squeeze building")
            elif fr<=-s['funding_low']:  fr_sc=s['pts_funding_low'];  reasons.append(f"⚡ Neg funding {fr*100:.4f}%")
            elif fr<0:                   fr_sc=3;                     reasons.append(f"⚡ Slightly neg funding {fr*100:.4f}%")
        else:
            if fr>=s['funding_high']:    fr_sc=s['pts_funding_high']; reasons.append(f"⚡ Extreme pos funding {fr*100:.4f}% — longs squeezed")
            elif fr>=s['funding_mid']:   fr_sc=s['pts_funding_mid'];  reasons.append(f"⚡ Strong pos funding {fr*100:.4f}%")
            elif fr>=s['funding_low']:   fr_sc=s['pts_funding_low'];  reasons.append(f"⚡ Pos funding {fr*100:.4f}%")
            elif fr>0:                   fr_sc=3;                     reasons.append(f"⚡ Slightly pos funding {fr*100:.4f}%")
        pump_score+=fr_sc; bd['funding']=fr_sc

        # 4 ── FUNDING HISTORY ─────────────────────────────────────────────
        funding_history=await self.fetch_funding_history(exch_obj,sym); funding_hist_sc=0; funding_age_penalty=0
        if len(funding_history)>=6 and abs(pc20)>5.0:
            try:
                early=funding_history[:6]; late=funding_history[-6:]
                avg_e=sum(early)/len(early); avg_l=sum(late)/len(late)
                if sig=="LONG" and pc20>5.0:
                    if avg_e>-0.0001 and avg_l<-0.0003:
                        funding_age_penalty=10; warnings.append(f"⚠️ FUNDING TRAP: Negative funding appeared AFTER pump — shorts may be right")
                    elif avg_e<-0.0003 and avg_l<-0.0001:
                        funding_hist_sc+=8; reasons.append(f"⚡ Funding negative BEFORE price surge — classic true squeeze setup")
            except: pass
        if len(funding_history)>=6:
            last_6=funding_history[-6:]
            if sig=="LONG" and all(r<0 for r in last_6) and funding_age_penalty==0:
                funding_hist_sc=12; reasons.append(f"⚡ Funding negative 6+ consecutive periods — deep squeeze setup")
            elif sig=="SHORT" and all(r>0 for r in last_6):
                funding_hist_sc=12; reasons.append(f"⚡ Funding positive 6+ consecutive periods — prolonged long squeeze")
            elif sig=="LONG" and sum(1 for r in last_6 if r<0)>=4 and funding_age_penalty==0:
                funding_hist_sc=6; reasons.append(f"⚡ Funding mostly negative last 6 periods")
            elif sig=="SHORT" and sum(1 for r in last_6 if r>0)>=4:
                funding_hist_sc=6; reasons.append(f"⚡ Funding mostly positive last 6 periods")
        pump_score+=funding_hist_sc-funding_age_penalty; bd['funding_hist']=funding_hist_sc; bd['funding_age_penalty']=-funding_age_penalty

        # 5 ── OPEN INTEREST ───────────────────────────────────────────────
        oi_sc=0; oi_history=await self.fetch_oi_history(exch_obj,sym); oi_change_6h=0.0
        if len(oi_history)>=6 and oi_history[-6]>0:
            oi_change_6h=(oi_history[-1]-oi_history[-6])/oi_history[-6]*100
        if oi>0:
            if sig=="LONG" and pc20>=s['oi_price_chg_high']:   oi_sc=s['pts_oi_high']; reasons.append(f"📈 OI + price up {pc20:.1f}% — confirmed accumulation")
            elif sig=="LONG" and pc20>=s['oi_price_chg_low']:  oi_sc=s['pts_oi_low'];  reasons.append(f"📈 OI growing, price +{pc20:.1f}%")
            elif sig=="SHORT" and pc20<=-s['oi_price_chg_high']: oi_sc=s['pts_oi_high']; reasons.append(f"📉 OI + price down {pc20:.1f}%")
            elif sig=="SHORT" and pc20<=-s['oi_price_chg_low']:  oi_sc=s['pts_oi_low'];  reasons.append(f"📉 OI building on drop {pc20:.1f}%")
            if abs(oi_change_6h)>=20:   oi_sc+=10; reasons.append(f"📈 OI spiked {oi_change_6h:+.1f}% in 6h — new money entering NOW")
            elif abs(oi_change_6h)>=10: oi_sc+=5;  reasons.append(f"📈 OI up {oi_change_6h:+.1f}% in 6h")
        else:
            if sig=="LONG" and pc20>=s['oi_price_chg_high']:   oi_sc=8; reasons.append(f"📈 Price up {pc20:.1f}% (OI unavailable)")
            elif sig=="SHORT" and pc20<=-s['oi_price_chg_high']: oi_sc=8; reasons.append(f"📉 Price down {pc20:.1f}% (OI unavailable)")
        pump_score+=oi_sc; bd['oi_spike']=oi_sc

        # 6 ── VOLUME SURGE ────────────────────────────────────────────────
        vsurge=lvol/vma; v_sc=0; explosive_thresh=s.get('vol_surge_explosive',5.0)
        if vsurge>=explosive_thresh:         v_sc=s.get('pts_vol_explosive',20); reasons.append(f"🚀 EXPLOSIVE volume {vsurge:.1f}×avg — institutional move NOW")
        elif vsurge>=s['vol_surge_high']:    v_sc=s['pts_vol_high']; reasons.append(f"🔥 Volume {vsurge:.1f}×avg — major activity NOW")
        elif vsurge>=s['vol_surge_mid']:     v_sc=s['pts_vol_mid'];  reasons.append(f"📊 Volume {vsurge:.1f}×avg — elevated")
        elif vsurge>=s['vol_surge_low']:     v_sc=s['pts_vol_low'];  reasons.append(f"📊 Volume {vsurge:.1f}×avg — above normal")
        pump_score+=v_sc; bd['vol_surge']=v_sc

        # 7 ── LIQUIDATION CLUSTERS ────────────────────────────────────────
        liq_sc=0; liq_target=0; liq_detail=""
        try:
            top3=df_f.nlargest(3,'volume')
            for _,row in top3.iterrows():
                mid=(float(row['high'])+float(row['low']))/2; dist=abs(price-mid)/price*100
                target=mid*1.02 if sig=="LONG" else mid*0.98
                if dist<=s['liq_cluster_near']:   liq_sc=s['pts_liq_near']; liq_target=target; liq_detail=f"${target:.6f}"; reasons.append(f"🧲 Liq cluster {dist:.1f}% away @ ${mid:.5f}"); break
                elif dist<=s['liq_cluster_mid'] and liq_sc<s['pts_liq_mid']:  liq_sc=s['pts_liq_mid']
                elif dist<=s['liq_cluster_far'] and liq_sc<s['pts_liq_far']:  liq_sc=s['pts_liq_far']
        except: pass
        pump_score+=liq_sc; bd['liq_cluster']=liq_sc

        # 8 ── 24H VOLUME ──────────────────────────────────────────────────
        cmc=self.cmc_data(dsym); vm_sc=0; vol_mcap_ratio=0
        if cmc and cmc.get('mcap',0)>0:
            vol_mcap_ratio=cmc['vol24']/cmc['mcap']
            if vol_mcap_ratio>1.0:    vm_sc=s['pts_vol24_high']+3; reasons.append(f"🌐 Vol/MCap {vol_mcap_ratio:.2f}× — extreme vs float")
            elif vol_mcap_ratio>0.5:  vm_sc=s['pts_vol24_high'];   reasons.append(f"🌐 Vol/MCap {vol_mcap_ratio:.2f}× — very high")
            elif vol_mcap_ratio>0.15: vm_sc=s['pts_vol24_mid'];    reasons.append(f"🌐 Vol/MCap {vol_mcap_ratio:.2f}×")
            elif vol_mcap_ratio>0.05: vm_sc=s['pts_vol24_low'];    reasons.append(f"🌐 Vol/MCap {vol_mcap_ratio:.2f}×")
        else:
            qv=float(tick.get('quoteVolume',0) or 0)
            if qv>=s['vol24h_high']:   vm_sc=s['pts_vol24_high']; reasons.append(f"📊 24h vol {fmt(qv)} — high activity")
            elif qv>=s['vol24h_mid']:  vm_sc=s['pts_vol24_mid'];  reasons.append(f"📊 24h vol {fmt(qv)}")
            elif qv>=s['vol24h_low']:  vm_sc=s['pts_vol24_low'];  reasons.append(f"📊 24h vol {fmt(qv)}")
        pump_score+=vm_sc; bd['vol_mcap']=vm_sc

        # 9 ── TECHNICALS ──────────────────────────────────────────────────
        tech_sc=0
        if sig=="LONG" and macd>macds:     tech_sc+=s['pts_macd']; reasons.append("📊 MACD bullish cross")
        elif sig=="SHORT" and macd<macds:  tech_sc+=s['pts_macd']; reasons.append("📊 MACD bearish cross")
        if sig=="LONG" and rsi<s['rsi_oversold']:      tech_sc+=s['pts_rsi']; reasons.append(f"📉 RSI oversold {rsi:.1f}")
        elif sig=="SHORT" and rsi>s['rsi_overbought']: tech_sc+=s['pts_rsi']; reasons.append(f"📈 RSI overbought {rsi:.1f}")
        if sig=="LONG" and float(df_f['close'].iloc[-1])<=bbl:    tech_sc+=s['pts_bb']; reasons.append("🎯 Price at lower BB")
        elif sig=="SHORT" and float(df_f['close'].iloc[-1])>=bbu: tech_sc+=s['pts_bb']; reasons.append("🎯 Price at upper BB")
        tech_sc+=s['pts_ema']; reasons.append(f"✅ EMA50 trend: {sig}")
        # RSI Divergence
        try:
            cl=df_f['close']; ra=rsi_series_raw
            if len(ra)>=10 and len(cl)>=10:
                r_arr=ra[-10:]; c_arr=cl.values[-10:]
                plnow=min(c_arr[-3:]); plprev=min(c_arr[:5]); rlnow=min(r_arr[-3:]); rlprev=min(r_arr[:5])
                phnow=max(c_arr[-3:]); phprev=max(c_arr[:5]); rhnow=max(r_arr[-3:]); rhprev=max(r_arr[:5])
                dp=s.get('pts_divergence',10)
                if sig=="LONG":
                    if plnow<plprev and rlnow>rlprev: tech_sc+=dp; reasons.append(f"📐 Bullish RSI divergence — price lower low, RSI higher low")
                    elif plnow>plprev and rlnow<rlprev: tech_sc+=dp//2; reasons.append(f"📐 Hidden bullish divergence")
                elif sig=="SHORT":
                    if phnow>phprev and rhnow<rhprev: tech_sc+=dp; reasons.append(f"📐 Bearish RSI divergence — price higher high, RSI lower high")
                    elif phnow<phprev and rhnow>rhprev: tech_sc+=dp//2; reasons.append(f"📐 Hidden bearish divergence")
        except: pass
        # Candle Patterns
        try:
            pp=s.get('pts_candle_pattern',8)
            c0=df_f.iloc[-1]; c1=df_f.iloc[-2]; c2=df_f.iloc[-3]
            b0=float(c0['close'])-float(c0['open']); b1=float(c1['close'])-float(c1['open'])
            r0=float(c0['high'])-float(c0['low']); r1=float(c1['high'])-float(c1['low'])
            if r0>0 and r1>0:
                if sig=="LONG":
                    lw=min(float(c0['open']),float(c0['close']))-float(c0['low'])
                    if lw/r0>0.6 and abs(b0)/r0<0.3: tech_sc+=pp; reasons.append("🔨 Hammer candle — buyers defended strongly")
                    elif b0>0 and b1<0 and abs(b0)>abs(b1)*1.1: tech_sc+=pp; reasons.append("🕯️ Bullish engulfing")
                    elif all(float(df_f.iloc[-k]['close'])>float(df_f.iloc[-k]['open']) for k in [1,2,3]) and float(df_f['volume'].iloc[-1])>float(df_f['volume'].iloc[-2])>float(df_f['volume'].iloc[-3]):
                        tech_sc+=pp; reasons.append("📈 3 consecutive green candles with rising volume")
                else:
                    uw=float(c0['high'])-max(float(c0['open']),float(c0['close']))
                    if uw/r0>0.6 and abs(b0)/r0<0.3: tech_sc+=pp; reasons.append("🌠 Shooting star — sellers rejected rally")
                    elif b0<0 and b1>0 and abs(b0)>abs(b1)*1.1: tech_sc+=pp; reasons.append("🕯️ Bearish engulfing")
                    elif all(float(df_f.iloc[-k]['close'])<float(df_f.iloc[-k]['open']) for k in [1,2,3]) and float(df_f['volume'].iloc[-1])>float(df_f['volume'].iloc[-2])>float(df_f['volume'].iloc[-3]):
                        tech_sc+=pp; reasons.append("📉 3 consecutive red candles with rising volume")
        except: pass
        pump_score+=tech_sc; bd['technicals']=tech_sc

        # 10 ── SESSION ────────────────────────────────────────────────────
        sname,smult,_=get_session(); ses_sc=0
        if smult>=1.4: ses_sc=s['pts_session']; reasons.append(f"⏰ {sname} — peak session")
        elif smult>=1.3: ses_sc=max(1,s['pts_session']//2); reasons.append(f"⏰ {sname}")
        pump_score+=ses_sc; bd['session']=ses_sc

        # 11 ── MOMENTUM ───────────────────────────────────────────────────
        candle_body=float(df_f['close'].iloc[-1])-float(df_f['open'].iloc[-1])
        recent_momentum=float(df_f['close'].iloc[-1])-float(df_f['close'].iloc[-3])
        momentum_confirmed=False; mom_sc=0
        if sig=="LONG" and candle_body>0 and recent_momentum>0:
            momentum_confirmed=True; mom_sc=8; reasons.append(f"✅ Momentum confirmed: price UP last 3 candles ({recent_momentum/price*100:+.2f}%)")
        elif sig=="SHORT" and candle_body<0 and recent_momentum<0:
            momentum_confirmed=True; mom_sc=8; reasons.append(f"✅ Momentum confirmed: price DOWN last 3 candles ({recent_momentum/price*100:+.2f}%)")
        pump_score+=mom_sc; bd['momentum']=mom_sc
        if s.get('require_momentum',False) and not momentum_confirmed: return None

        # 11b ── MTF CONFIRMATION ──────────────────────────────────────────
        mtf_sc=0
        if s.get('mtf_confirm',True):
            try:
                df_med=await self.fetch_ohlcv(exch_obj,sym,"1h",60)
                if not df_med.empty:
                    df_med.ta.ema(length=50,append=True)
                    ec=[c for c in df_med.columns if 'EMA_50' in c]
                    if ec:
                        e50m=float(df_med[ec[0]].iloc[-1]); cm=float(df_med['close'].iloc[-1])
                        mt="LONG" if cm>e50m else "SHORT"; ft="LONG" if candle_body>0 else "SHORT"
                        aligned=sum(1 for t in [sig,mt,ft] if t==sig)
                        if aligned==3:   mtf_sc=s.get('pts_mtf',12);      reasons.append(f"🎯 All 3 TFs aligned {sig} — {s['fast_tf']}+1h+{s['slow_tf']}")
                        elif aligned==2: mtf_sc=s.get('pts_mtf',12)//2;  reasons.append(f"🎯 2/3 TFs aligned {sig}")
                        else:            mtf_sc=-5;                        reasons.append(f"⚠️ TF conflict — {s['fast_tf']} vs 1h vs {s['slow_tf']}")
            except: pass
        pump_score+=mtf_sc; bd['mtf']=mtf_sc

        # 12 ── SENTIMENT ──────────────────────────────────────────────────
        sentiment={'top_long_pct':50,'top_short_pct':50,'retail_long_pct':50,'taker_buy_pct':50,'available':False,'source':''}
        sent_sc=0
        if exch_name in ("GATE","OKX"):
            sb=dsym+("-USDT-SWAP" if exch_name=="OKX" else "_USDT")
            sentiment=self.fetch_sentiment_data(sb,exch_name)
            if sentiment['available']:
                pts=s.get('pts_sentiment',20); pt=s.get('pts_taker',10)
                if sig=="LONG":
                    if sentiment['top_short_pct']>65 and sentiment['retail_long_pct']>60: sent_sc=pts; reasons.append(f"🧠 Smart money {sentiment['top_short_pct']:.0f}% SHORT vs retail {sentiment['retail_long_pct']:.0f}% LONG — squeeze fuel")
                    elif sentiment['top_short_pct']>55: sent_sc=pts//2; reasons.append(f"🧠 Top traders {sentiment['top_short_pct']:.0f}% short")
                    if sentiment['taker_buy_pct']>62:   sent_sc+=pt;    reasons.append(f"💚 Taker buy {sentiment['taker_buy_pct']:.0f}%")
                    elif sentiment['taker_buy_pct']>55: sent_sc+=pt//2; reasons.append(f"💚 Taker buy slightly dominant {sentiment['taker_buy_pct']:.0f}%")
                else:
                    if sentiment['top_long_pct']>65 and sentiment['retail_long_pct']>65: sent_sc=pts; reasons.append(f"🧠 Smart money {sentiment['top_long_pct']:.0f}% LONG + retail crowded — distribution")
                    elif sentiment['top_long_pct']>55: sent_sc=pts//2; reasons.append(f"🧠 Top traders {sentiment['top_long_pct']:.0f}% long")
                    if sentiment['taker_buy_pct']<38:   sent_sc+=pt;    reasons.append(f"🔴 Taker sell {100-sentiment['taker_buy_pct']:.0f}%")
                    elif sentiment['taker_buy_pct']<45: sent_sc+=pt//2; reasons.append(f"🔴 Taker sell slightly dominant")
        pump_score+=sent_sc; bd['sentiment']=sent_sc

        # 12b ── OI+FUNDING COMBO ──────────────────────────────────────────
        combo_sc=0
        if bd.get('oi_spike',0)>=10 and (bd.get('funding',0)>=s.get('pts_funding_high',25) or bd.get('funding_hist',0)>=12):
            combo_sc=s.get('pts_oi_funding_combo',10); reasons.append("💥 OI surge + extreme funding combo — maximum squeeze pressure")
        pump_score+=combo_sc; bd['oi_funding_combo']=combo_sc

        # 13 ── SOCIAL BUZZ ────────────────────────────────────────────────
        social_data=self.fetch_reddit_buzz(dsym,s); social_sc=0
        if social_data.get('available') and social_data['mentions']>=s.get('social_min_mentions',3):
            social_sc=social_data['score']
            em="🚀" if social_data.get('sentiment')=='BULLISH' else ("🩸" if social_data.get('sentiment')=='BEARISH' else "💬")
            reasons.append(f"{em} {social_data.get('source','Reddit')}: {social_data['mentions']} mentions/hr | {social_data.get('sentiment','?')}")
            if social_data.get('top_post'): reasons.append(f"📢 Top post: \"{social_data['top_post'][:55]}...\"")
        pump_score+=social_sc; bd['social_buzz']=social_sc

        # 14 ── ORDER FLOW ─────────────────────────────────────────────────
        of_pct,of_dir=await self.fetch_orderflow_imbalance(exch_obj,sym,s.get('orderflow_lookback',10))
        of_sc=0; pts_of=s.get('pts_orderflow',12); sell_pct=100-of_pct
        if sig=='LONG':
            if of_dir=='BUY' and of_pct>=65:   of_sc=pts_of;   reasons.append(f"📊 Order flow {of_pct:.0f}% BUY — sustained accumulation")
            elif of_dir=='BUY' and of_pct>=58: of_sc=pts_of//2; reasons.append(f"📊 Order flow mildly bullish ({of_pct:.0f}% buy)")
        elif sig=='SHORT':
            if of_dir=='SELL' and sell_pct>=65:   of_sc=pts_of;   reasons.append(f"📊 Order flow {sell_pct:.0f}% SELL — sustained distribution")
            elif of_dir=='SELL' and sell_pct>=58: of_sc=pts_of//2; reasons.append(f"📊 Order flow mildly bearish ({sell_pct:.0f}% sell)")
        pump_score+=of_sc; bd['orderflow']=of_sc

        # 15 ── LIQUIDATION MAP ────────────────────────────────────────────
        liq_map=await self.fetch_liquidation_map(exch_obj,sym,price); liq_map_sc=0; pts_lm=s.get('pts_liq_map',15)
        if liq_map:
            nearest=liq_map[0]; d=nearest['dist_pct']; sz_m=nearest['size_usd']/1e6; slbl=nearest['side']
            em_lm="💥" if ((slbl=='SHORT_LIQ' and sig=='LONG') or (slbl=='LONG_LIQ' and sig=='SHORT')) else "🧲"
            if d<=1.5:   liq_map_sc=pts_lm;    reasons.append(f"{em_lm} Liq cluster {slbl} ${sz_m:.1f}M @ ${nearest['price']:.4f} ({d:.2f}% away)")
            elif d<=3.0: liq_map_sc=pts_lm//2; reasons.append(f"🧲 Liq cluster {slbl} ${sz_m:.1f}M at {d:.1f}%")
        pump_score+=liq_map_sc; bd['liq_map']=liq_map_sc

        # 16 ── LISTING DETECTOR ───────────────────────────────────────────
        new_listings=self.fetch_new_listings(s); listing_sc=0; listing_info={}; pts_lst=s.get('listing_alert_pts',25)
        for lst in new_listings:
            if lst['symbol'].upper()==dsym.upper():
                h=lst.get('listed_ago_h',999); listing_info=lst
                if h<=24:   listing_sc=pts_lst;     reasons.append(f"🆕 BRAND NEW LISTING on {lst['exchange']} {h:.0f}h ago")
                elif h<=72: listing_sc=pts_lst//2;  reasons.append(f"🆕 Recent listing {h:.0f}h ago")
                elif h<=168:listing_sc=pts_lst//4;  reasons.append(f"🆕 Listed this week ({h:.0f}h ago)")
                break
        pump_score+=listing_sc; bd['listing']=listing_sc

        # 17 ── ON-CHAIN WHALE ─────────────────────────────────────────────
        onchain=self.fetch_onchain_whale(dsym,s); onchain_sc=0; pts_oc=s.get('pts_onchain_whale',15)
        if onchain.get('available'):
            if onchain['signal']=='BULLISH' and sig=='LONG':   onchain_sc=pts_oc;       reasons.append(f"🐋 On-chain: {onchain['detail']} — leaving exchanges")
            elif onchain['signal']=='BEARISH' and sig=='SHORT': onchain_sc=pts_oc;       reasons.append(f"🐋 On-chain: {onchain['detail']} — entering exchanges")
            elif onchain['signal']=='BULLISH' and sig=='SHORT': onchain_sc=-(pts_oc//2); reasons.append(f"⚠️ On-chain bullish conflicts with SHORT")
            elif onchain['signal']=='BEARISH' and sig=='LONG':  onchain_sc=-(pts_oc//2); reasons.append(f"⚠️ On-chain bearish conflicts with LONG")
        pump_score+=onchain_sc; bd['onchain']=onchain_sc
        pump_score=max(0,min(pump_score,100))

        # ── ACCURACY GATES ────────────────────────────────────────────────
        active_cats=sum(1 for k,v in bd.items() if v>0 and k not in ('session','mtf'))
        if active_cats<s.get('min_active_signals',3): return None
        if len(reasons)<s.get('min_reasons',1): return None
        fng=st.session_state.get('fng_val',50)
        if sig=="LONG" and fng<s.get('fng_long_threshold',30) and pump_score<60: return None
        if sig=="SHORT" and fng>s.get('fng_short_threshold',70) and pump_score<60: return None

        # ── SL: 4H OB + FVG ──────────────────────────────────────────────
        bull_obs_4h,bear_obs_4h,bull_fvgs,bear_fvgs=[],[],[],[]
        if s.get('use_4h_ob_for_sl',True):
            bull_obs_4h,bear_obs_4h,bull_fvgs,bear_fvgs=await self.fetch_4h_ob_and_fvg(exch_obj,sym,price)

        try:
            rh=df_f['high'].rolling(5).max().dropna(); rl=df_f['low'].rolling(5).min().dropna()
            swing_high=float(rh.iloc[-2]) if len(rh)>=2 else price*1.03
            swing_low=float(rl.iloc[-2]) if len(rl)>=2 else price*0.97
            if sig=="LONG":
                structure_sl=swing_low*0.995
                whale_bid_px=whale_details[0]['price'] if whale_details and whale_details[0]['side']=='BUY' else 0
                whale_sl=whale_bid_px*0.995 if 0<whale_bid_px<price else structure_sl
                # Best SL from 4H OB or FVG
                ob_4h_sl=structure_sl; ob_4h_used=False; fvg_sl=structure_sl; fvg_used=False
                if bull_obs_4h:
                    c=bull_obs_4h[0]; cand=c['zone_lo']*0.995
                    if cand<price and cand>structure_sl*0.98:
                        ob_4h_sl=cand; ob_4h_used=True
                        reasons.append(f"🧱 4H OB support ${c['zone_lo']:.5f}–${c['zone_hi']:.5f} — SL anchored below")
                if bull_fvgs:
                    f=bull_fvgs[0]; cand=f['zone_lo']*0.995
                    if cand<price and cand>structure_sl*0.97:
                        fvg_sl=cand; fvg_used=True
                        reasons.append(f"🔷 4H FVG support ${f['zone_lo']:.5f}–${f['zone_hi']:.5f} — SL anchored in gap")
                sl_cands=[x for x in [structure_sl,whale_sl,ob_4h_sl,fvg_sl] if x<price]
                sl=max(sl_cands) if sl_cands else price-atr*1.5
                if sl>=price: sl=price-atr*1.5
                sl_dist=price-sl; min_rr_tp=price+sl_dist*max(s.get('min_rr',1.5),1.5)
                tp=max(swing_high,min_rr_tp)
                if liq_target>price: tp=min(tp,liq_target) if liq_target<tp else tp
            else:
                structure_sl=swing_high*1.005
                whale_ask_px=whale_details[0]['price'] if whale_details and whale_details[0]['side']=='SELL' else 0
                whale_sl=whale_ask_px*1.005 if whale_ask_px>price else structure_sl
                ob_4h_sl=structure_sl
                if bear_obs_4h:
                    c=bear_obs_4h[0]; cand=c['zone_hi']*1.005
                    if cand>price and cand<structure_sl*1.02:
                        ob_4h_sl=cand
                        reasons.append(f"🧱 4H OB resistance ${c['zone_lo']:.5f}–${c['zone_hi']:.5f} — SL above it")
                fvg_sl=structure_sl
                if bear_fvgs:
                    f=bear_fvgs[0]; cand=f['zone_hi']*1.005
                    if cand>price and cand<structure_sl*1.02:
                        fvg_sl=cand
                        reasons.append(f"🔷 4H FVG resistance ${f['zone_lo']:.5f}–${f['zone_hi']:.5f}")
                sl_cands=[x for x in [structure_sl,whale_sl,ob_4h_sl,fvg_sl] if x>price]
                sl=min(sl_cands) if sl_cands else price+atr*1.5
                if sl<=price: sl=price+atr*1.5
                sl_dist=sl-price; min_rr_tp=price-sl_dist*max(s.get('min_rr',1.5),1.5)
                tp=min(swing_low,min_rr_tp)
                if liq_target>0 and liq_target<price: tp=max(tp,liq_target)
        except:
            tp=price+atr*3 if sig=="LONG" else price-atr*3
            sl=price-atr*1.5 if sig=="LONG" else price+atr*1.5

        try:
            sl_dist=abs(price-sl) if abs(price-sl)>0 else atr
            if sig=="LONG":
                tp1=price+sl_dist*1.5; tp2=tp; tp3=max(price+sl_dist*4.0,liq_target) if liq_target>swing_high else price+sl_dist*4.0
                tp3=max(tp3,price+sl_dist*3.0)
            else:
                tp1=price-sl_dist*1.5; tp2=tp; tp3=min(price-sl_dist*4.0,liq_target) if liq_target>0 and liq_target<swing_low else price-sl_dist*4.0
                tp3=min(tp3,price-sl_dist*3.0)
        except:
            tp1=price+atr*1.5 if sig=="LONG" else price-atr*1.5
            tp2=price+atr*3   if sig=="LONG" else price-atr*3
            tp3=price+atr*5   if sig=="LONG" else price-atr*5

        try:
            rr=abs(tp-price)/abs(price-sl)
            if rr<s.get('min_rr',1.5): return None
        except: pass

        result={
            'symbol':dsym,'exchange':exch_name,'price':price,'pump_score':pump_score,
            'tp1':tp1,'tp2':tp2,'tp3':tp3,'type':sig,'reasons':reasons,'tp':tp,'sl':sl,
            'rsi':rsi,'funding':fr,'atr':atr,'ob':ob_data,'cmc':cmc,'oi':oi,
            'oi_change_6h':oi_change_6h,'liq_target':liq_target,'liq_detail':liq_detail,
            'whale_str':whale_str,'whale_details':whale_details,'vol_mcap':vol_mcap_ratio,
            'signal_breakdown':bd,'session':sname,'price_chg_20':pc20,
            'timestamp':datetime.now().strftime('%H:%M:%S'),
            'quote_vol':float(tick.get('quoteVolume',0) or 0),'sentiment':sentiment,
            'momentum_confirmed':momentum_confirmed,
            'funding_history':funding_history[-8:] if funding_history else [],
            'social_data':social_data,'atr_pct':round(atr_pct,2),
            'pct_24h':float(tick.get('percentage') or tick.get('change') or 0),
            'vol_surge_ratio':round(vsurge,2),'liq_map_data':liq_map[:3] if liq_map else [],
            'listing_data':listing_info,'onchain_data':onchain,
            'orderflow_data':{'pct':round(of_pct,1),'dir':of_dir},
            'entry_lo':round(price-atr*0.35,8) if sig=='LONG' else round(price+atr*0.1,8),
            'entry_hi':round(price+atr*0.1,8)  if sig=='LONG' else round(price+atr*0.35,8),
            'warnings':warnings,'late_entry_flag':late_entry_flag,
            'vol_exhaust_flag':vol_exhaust_flag,'near_top_flag':near_top_flag,
            'rsi_direction':rsi_direction,
            'bull_obs_4h':bull_obs_4h[:2] if bull_obs_4h else [],
            'bear_obs_4h':bear_obs_4h[:2] if bear_obs_4h else [],
            'bull_fvgs':bull_fvgs[:2] if bull_fvgs else [],
            'bear_fvgs':bear_fvgs[:2] if bear_fvgs else [],
        }
        result['_cls_cfg']={
            'breakout_oi_min':s.get('cls_breakout_oi_min',7),'breakout_vol_min':s.get('cls_breakout_vol_min',4),
            'breakout_sc_min':s.get('cls_breakout_score_min',25),'squeeze_fund_min':s.get('cls_squeeze_fund_min',8),
            'squeeze_ob_min':s.get('cls_squeeze_ob_min',6),
        }
        result['cls']=classify(result)
        if s['cooldown_on']: set_cooldown(dsym)
        return result

    async def run(self, s):
        btc_trend,btc_px,btc_rsi=await self.fetch_btc()
        await asyncio.gather(self.okx.load_markets(),self.mexc.load_markets(),self.gate.load_markets())
        tk_okx,tk_mexc,tk_gate={},{},{}
        try: tk_okx=await self.okx.fetch_tickers()
        except: pass
        try: tk_mexc=await self.mexc.fetch_tickers()
        except: pass
        try: tk_gate=await self.gate.fetch_tickers()
        except: pass
        process_journal_tracking({**tk_okx,**tk_mexc,**tk_gate},s)
        swaps_dict={}
        for sym,t in tk_mexc.items():
            if sym.endswith(':USDT') and t.get('quoteVolume'):
                swaps_dict[sym]={'vol':float(t['quoteVolume'] or 0),'pct':float(t.get('percentage') or t.get('change') or 0),'exch_name':'MEXC','exch_obj':self.mexc}
        for sym,t in tk_gate.items():
            if sym.endswith(':USDT') and t.get('quoteVolume'):
                swaps_dict[sym]={'vol':float(t['quoteVolume'] or 0),'pct':float(t.get('percentage') or t.get('change') or 0),'exch_name':'GATE','exch_obj':self.gate}
        for sym,t in tk_okx.items():
            if sym.endswith(':USDT') and t.get('quoteVolume'):
                swaps_dict[sym]={'vol':float(t['quoteVolume'] or 0),'pct':float(t.get('percentage') or t.get('change') or 0),'exch_name':'OKX','exch_obj':self.okx}
        scan_modes=s.get('scan_modes',['mixed'])
        if isinstance(scan_modes,str): scan_modes=[scan_modes]
        depth=s['scan_depth']
        def get_mode_symbols(mode,d,n):
            if mode=='gainers': return sorted(d.items(),key=lambda x:x[1]['pct'],reverse=True)[:n]
            elif mode=='losers': return sorted(d.items(),key=lambda x:x[1]['pct'])[:n]
            elif mode=='mixed':
                bg=sorted(d.items(),key=lambda x:x[1]['pct'],reverse=True)
                bl=sorted(d.items(),key=lambda x:x[1]['pct'])
                half=n//2; seen=set(); mixed=[]
                for item in bg[:half]+bl[:half]:
                    if item[0] not in seen: mixed.append(item); seen.add(item[0])
                return mixed
            else: return sorted(d.items(),key=lambda x:x[1]['vol'],reverse=True)[:n]
        combined={}; per_mode=max(depth//max(1,len(scan_modes)),10)
        for mode in scan_modes:
            for sym,data in get_mode_symbols(mode,swaps_dict,per_mode):
                if sym not in combined: combined[sym]=data
        symbols=list(combined.items())[:depth]
        results=[]; errors=[]; pb=st.progress(0); st_=st.empty()
        for i,(sym,data) in enumerate(symbols):
            exch_name=data['exch_name']; exch_obj=data['exch_obj']
            st_.markdown(f"<span style='font-family:monospace;font-size:.68rem;color:#7a82a0;'>Scanning {i+1}/{len(symbols)} — <b>{sym.split(':')[0]}</b> ({exch_name}) — found: <b style='color:#0f1117'>{len(results)}</b></span>",unsafe_allow_html=True)
            try:
                r=await self.analyze(exch_name,exch_obj,sym,s,btc_trend)
                if r: results.append(r)
            except Exception as e: errors.append(f"{sym} ({exch_name}): {str(e)[:80]}")
            pb.progress((i+1)/len(symbols)); await asyncio.sleep(0.5)
        st_.empty(); pb.empty()
        try:
            await self.okx.close(); await self.mexc.close(); await self.gate.close()
        except: pass
        results.sort(key=lambda x:x['pump_score'],reverse=True)
        if s.get('dedup_symbols',True):
            seen={}
            for r in results:
                sb=r['symbol']
                if sb not in seen or r['pump_score']>seen[sb]['pump_score']: seen[sb]=r
            results=sorted(seen.values(),key=lambda x:x['pump_score'],reverse=True)
        return results,btc_trend,btc_px,btc_rsi,errors
# ─── CARD RENDERER ────────────────────────────────────────────────────────────
def render_card(res, is_sniper=False, dual_confirmed=False):
    sc=res['pump_score']; col=pump_color(sc,is_sniper); lbl=pump_label(sc,res['type'],is_sniper)
    sig=res['type']; bd=res.get('signal_breakdown',{}); ob_data=res.get('ob',{})
    cmc=res.get('cmc') or {}; card_cls="pc-long" if sig=="LONG" else "pc-short"
    sig_col="var(--green)" if sig=="LONG" else "var(--red)"
    sentiment=res.get('sentiment',{})
    if not isinstance(sentiment,dict): sentiment={}
    momentum_confirmed=res.get('momentum_confirmed',False)
    warnings_list=res.get('warnings',[])
    exch=res.get('exchange','MEXC')
    exch_colors={"OKX":"#00bcd4","GATE":"#e040fb","MEXC":"#2563eb"}
    exch_col=exch_colors.get(exch,"#2563eb")
    if exch=="OKX": trade_link=f"https://www.okx.com/trade-swap/{res['symbol'].lower()}-usdt-swap"
    elif exch=="GATE": trade_link=f"https://www.gate.io/futures_trade/USDT/{res['symbol']}_USDT"
    else: trade_link=f"https://www.mexc.com/exchange/{res['symbol']}_USDT"

    def pip(v,lo=5,hi=12):
        if v>=hi: return "<span class='pip pip-on'></span>"
        if v>=lo: return "<span class='pip pip-half'></span>"
        return "<span class='pip pip-off'></span>"

    bid_pct=ob_data.get('bid_pct',50)
    chg=cmc.get('change24',0)
    cmc_html=""
    if cmc:
        chg_c="var(--green)" if chg>=0 else "var(--red)"
        cmc_html=(f'<div style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;">'
                  f'<span style="background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:2px 7px;font-family:monospace;font-size:.6rem;color:var(--muted);">Rank #{cmc.get("rank","?")}</span>'
                  f'<span style="background:var(--panel);border:1px solid var(--border);border-radius:4px;padding:2px 7px;font-family:monospace;font-size:.6rem;color:var(--muted);">MCap {fmt(cmc.get("mcap",0))}</span>'
                  f'<span style="background:{"var(--green-bg)" if chg>=0 else "var(--red-bg)"};border:1px solid {"var(--green-bd)" if chg>=0 else "var(--red-bd)"};border-radius:4px;padding:2px 7px;font-family:monospace;font-size:.6rem;color:{chg_c};">{"+".join(["" if chg<0 else "+"])}{chg:.2f}%</span></div>')

    dual_html=""
    if dual_confirmed:
        dual_html='<div class="dual-confirm">🔥 DUAL CONFIRMED — Scanner + Sentinel both flagged this coin — HIGH CONVICTION SIGNAL</div>'

    warnings_html=""
    if warnings_list:
        wi="".join([f'<div style="padding:3px 0;font-size:.68rem;">{w}</div>' for w in warnings_list])
        warnings_html=(f'<div style="background:#fef3c7;border:1px solid #f59e0b;border-left:4px solid #f59e0b;border-radius:6px;padding:8px 12px;margin:6px 0;">'
                       f'<div style="font-family:monospace;font-size:.6rem;font-weight:700;color:#92400e;margin-bottom:4px;">⚠️ RISK WARNINGS</div>'
                       f'<div style="font-family:monospace;color:#92400e;">{wi}</div></div>')

    # 4H OB + FVG display
    ob4h_html=""
    bull_obs=res.get('bull_obs_4h',[]); bear_obs=res.get('bear_obs_4h',[])
    bull_fvgs=res.get('bull_fvgs',[]); bear_fvgs=res.get('bear_fvgs',[])
    if sig=="LONG":
        if bull_obs:
            o=bull_obs[0]; ob4h_html+=f'<div style="background:#ecfdf5;border-left:3px solid #059669;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:.66rem;color:#065f46;margin:4px 0;">🧱 4H OB Support: ${o["zone_lo"]:.5f}–${o["zone_hi"]:.5f} | body {o["body_pct"]*100:.0f}% | {o["dist_pct"]:.1f}% below</div>'
        if bull_fvgs:
            f=bull_fvgs[0]; ob4h_html+=f'<div style="background:#eff6ff;border-left:3px solid #2563eb;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:.66rem;color:#1e40af;margin:4px 0;">🔷 4H FVG Gap: ${f["zone_lo"]:.5f}–${f["zone_hi"]:.5f} | {f["dist_pct"]:.1f}% away — SL in gap</div>'
    elif sig=="SHORT":
        if bear_obs:
            o=bear_obs[0]; ob4h_html+=f'<div style="background:#fef2f2;border-left:3px solid #dc2626;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:.66rem;color:#7f1d1d;margin:4px 0;">🧱 4H OB Resistance: ${o["zone_lo"]:.5f}–${o["zone_hi"]:.5f} | {o["dist_pct"]:.1f}% above</div>'
        if bear_fvgs:
            f=bear_fvgs[0]; ob4h_html+=f'<div style="background:#fff7ed;border-left:3px solid #ea580c;border-radius:6px;padding:6px 12px;font-family:monospace;font-size:.66rem;color:#7c2d12;margin:4px 0;">🔷 4H FVG Gap: ${f["zone_lo"]:.5f}–${f["zone_hi"]:.5f} | {f["dist_pct"]:.1f}% away</div>'

    social_html=""
    sd=res.get('social_data',{})
    if not isinstance(sd,dict): sd={}
    if sd.get('available'):
        scfg={'BULLISH':('#059669','#ecfdf5','#a7f3d0','📈'),'BEARISH':('#dc2626','#fef2f2','#fecaca','📉'),'NEUTRAL':('#d97706','#fffbeb','#fde68a','💬')}
        sc_color,sc_bg,sc_bd,sc_emoji=scfg.get(sd.get('sentiment','NEUTRAL'),scfg['NEUTRAL'])
        bp=min(100,int((sd.get('mentions',0)/max(1,S.get('social_buzz_threshold',10)))*100))
        avg_line=f' · avg {sd.get("upvote_avg",0):.0f} upvotes' if sd.get('upvote_avg',0)>0 else ""
        tp60=sd.get('top_post','')[:60]
        tl=f'<div style="font-size:.6rem;color:{sc_color};margin-top:4px;opacity:.85;">"{tp60}..."</div>' if tp60 else ""
        social_html=(f'<div style="background:{sc_bg};border:1px solid {sc_bd};border-left:4px solid {sc_color};border-radius:6px;padding:8px 12px;margin:6px 0;">'
                     f'<div style="display:flex;justify-content:space-between;"><span style="font-family:monospace;font-size:.62rem;font-weight:700;color:{sc_color};">{sc_emoji} {sd.get("source","Reddit")} Buzz</span>'
                     f'<span style="font-family:monospace;font-size:.7rem;font-weight:700;color:{sc_color};">{sd.get("sentiment","NEUTRAL")}</span></div>'
                     f'<div style="display:flex;align-items:center;gap:8px;margin-top:5px;"><div style="flex:1;height:4px;background:rgba(0,0,0,.08);border-radius:2px;"><div style="width:{bp}%;height:100%;background:{sc_color};border-radius:2px;"></div></div>'
                     f'<span style="font-family:monospace;font-size:.6rem;color:{sc_color};">{sd.get("mentions",0)} mentions/hr{avg_line}</span></div>{tl}</div>')

    whale_html=""
    wd=res.get('whale_details',[])
    if wd:
        w=wd[0]; wc="var(--green)" if w['side']=='BUY' else "var(--red)"; wbg="var(--green-bg)" if w['side']=='BUY' else "var(--red-bg)"
        whale_html=(f'<div style="background:{wbg};border-left:3px solid {wc};border-radius:6px;padding:7px 12px;font-family:monospace;font-size:.72rem;color:{wc};margin:6px 0;font-weight:600;display:flex;justify-content:space-between;">'
                    f'<span>🐋 {w["side"]} WALL {fmt(w["value"])} @ ${w["price"]:.6f}</span>'
                    f'<span style="color:var(--muted);font-size:.58rem;">({w["dist_pct"]:.2f}% {"below" if w["side"]=="BUY" else "above"})</span></div>')

    liq_html=""
    if res.get('liq_target',0):
        liq_html=f'<div style="background:var(--purple-bg);border-left:3px solid var(--purple);border-radius:6px;padding:7px 12px;font-size:.72rem;color:var(--purple);margin:6px 0;">🧲 Liq magnet @ {res["liq_detail"]}</div>'

    mom_html=""
    if momentum_confirmed:
        mc="var(--green)" if sig=="LONG" else "var(--red)"; mb="var(--green-bg)" if sig=="LONG" else "var(--red-bg)"
        mom_html=f'<span class="momentum-badge" style="background:{mb};color:{mc};border:1px solid {mc};">{"▲" if sig=="LONG" else "▼"} MOMENTUM LIVE</span>'

    rsi_dir=res.get('rsi_direction','FLAT'); rsi_dir_html=""
    if rsi_dir=='RISING' and sig=="LONG":   rsi_dir_html='<span style="background:#ecfdf5;color:#059669;border:1px solid #a7f3d0;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:.56rem;font-weight:700;">RSI ↑</span>'
    elif rsi_dir=='FALLING' and sig=="SHORT": rsi_dir_html='<span style="background:#fef2f2;color:#dc2626;border:1px solid #fecaca;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:.56rem;font-weight:700;">RSI ↓</span>'
    elif (rsi_dir=='FALLING' and sig=="LONG") or (rsi_dir=='RISING' and sig=="SHORT"): rsi_dir_html='<span style="background:#fef3c7;color:#92400e;border:1px solid #fde68a;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:.56rem;font-weight:700;">RSI ⚠</span>'

    sentiment_html=""
    if sentiment.get('available'):
        def sbar(label,pct,ch,cl):
            c=ch if pct>50 else cl
            return (f'<div class="sentiment-bar"><div class="sbar-label">{label}</div>'
                    f'<div class="sbar-track"><div class="sbar-fill" style="width:{pct:.0f}%;background:{c};"></div></div>'
                    f'<div class="sbar-val" style="color:{c};">{pct:.0f}%</div></div>')
        sentiment_html=(f'<div style="margin:8px 0;padding:10px;background:var(--panel);border-radius:8px;border:1px solid var(--border);">'
                        f'<div style="font-family:monospace;font-size:.55rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:6px;">{sentiment.get("source","Exchange")} Sentiment</div>'
                        f'{sbar("Longs",sentiment["top_long_pct"],"var(--green)","var(--red)")}'
                        f'{sbar("Retail Long",sentiment["retail_long_pct"],"var(--green)","var(--red)")}'
                        f'{sbar("Taker Buy Vol",sentiment["taker_buy_pct"],"var(--green)","var(--red)")}</div>')

    oi_change=res.get('oi_change_6h',0); oi_change_html=""
    if abs(oi_change)>=5:
        oc="var(--green)" if oi_change>0 else "var(--red)"; ob_bg="var(--green-bg)" if oi_change>0 else "var(--red-bg)"
        oi_change_html=f'<span style="background:{ob_bg};color:{oc};border:1px solid;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:.6rem;font-weight:600;">OI {oi_change:+.1f}% 6h</span>'

    # ALL reasons (no truncation on card)
    def _esc(s): return str(s).replace("{","{{").replace("}","}}")
    reasons_html="".join([f"<div class='r'><span style='color:var(--muted);margin-right:6px;'>▸</span>{_esc(r)}</div>" for r in res['reasons']])

    _price=float(res.get('price') or 0); _tp=float(res.get('tp') or 0); _sl=float(res.get('sl') or 0)
    _rsi=float(res.get('rsi') or 0)
    _tp1=float(res.get('tp1') or _tp or 0); _tp2=float(res.get('tp2') or _tp or 0); _tp3=float(res.get('tp3') or _tp or 0)
    tp1_pct=f"+{abs(_tp1-_price)/_price*100:.2f}%" if _price>0 and _tp1>0 else ""
    tp2_pct=f"+{abs(_tp2-_price)/_price*100:.2f}%" if _price>0 and _tp2>0 else ""
    tp3_pct=f"+{abs(_tp3-_price)/_price*100:.2f}%" if _price>0 and _tp3>0 else ""
    sl_pct=f"-{abs(_price-_sl)/_price*100:.2f}%" if _price>0 and _sl>0 else ""
    pct_24h=float(res.get('pct_24h',0) or 0); pct_col="var(--green)" if pct_24h>=0 else "var(--red)"
    pct_display=f"{'+'if pct_24h>=0 else ''}{pct_24h:.2f}%"
    _elo_d=f"${float(res.get('entry_lo') or 0):.6f}"; _ehi_d=f"${float(res.get('entry_hi') or 0):.6f}"
    sym_cls=res.get('cls','—').upper(); ts=res.get('timestamp',''); session=res.get('session','')
    new_badge='<span style="background:#0ea5e9;color:#fff;font-size:.52rem;font-weight:700;padding:1px 6px;border-radius:3px;margin-right:4px;">🆕 NEW</span>' if res.get('is_new') else ''
    jump_sc=res.get('score_jump',0)
    jump_badge=f'<span style="background:#f59e0b22;color:#f59e0b;font-size:.52rem;font-weight:700;padding:1px 6px;border-radius:3px;margin-right:4px;">⬆️ +{jump_sc}pt</span>' if jump_sc>=15 else ''
    warn_count_html=f'<span style="background:#fef3c7;color:#92400e;border:1px solid #f59e0b;padding:2px 7px;border-radius:4px;font-family:monospace;font-size:.58rem;font-weight:700;">⚠️ {len(warnings_list)} RISK</span>' if warnings_list else ''
    try:
        rr=abs(_tp-_price)/abs(_price-_sl) if abs(_price-_sl)>0 else 0
        rr_str=f"{rr:.1f}:1"; rr_col="var(--green)" if rr>=2 else ("var(--amber)" if rr>=1.5 else "var(--red)")
    except: rr_str="N/A"; rr_col="var(--muted)"
    exch_bg=exch_col+"22"; exch_bd=exch_col+"44"; sig_bg=sig_col+"18"; sig_bd=sig_col+"44"
    score_bg=col+"14"; sniper_cls=" sniper" if is_sniper else ""
    try:
        card_html=f"""<div class="pump-card {card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;">
    <div style="display:flex;align-items:center;gap:12px;">
      <div class="score-ring{sniper_cls}" style="color:{col};border-color:{col};background:{score_bg};">{sc}</div>
      <div>
        <div style="font-family:monospace;font-size:1.2rem;font-weight:700;">{res['symbol']}</div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:3px;flex-wrap:wrap;">
          <span style="background:{exch_bg};border:1px solid {exch_bd};color:{exch_col};padding:1px 7px;border-radius:3px;font-family:monospace;font-size:.58rem;font-weight:700;">{exch}</span>
          <span style="background:{sig_bg};border:1px solid {sig_bd};color:{sig_col};padding:2px 10px;border-radius:3px;font-family:monospace;font-size:.72rem;font-weight:800;">{'🟢 LONG' if sig=='LONG' else '🔴 SHORT'}</span>
          <span style="font-family:monospace;font-size:.6rem;color:{col};font-weight:600;">{lbl}</span>
          {mom_html}{rsi_dir_html}{oi_change_html}{warn_count_html}{new_badge}{jump_badge}
          <span style="font-family:monospace;font-size:.58rem;color:var(--muted);">{ts}</span>
        </div>
      </div>
    </div>
    <div style="text-align:right;font-family:monospace;font-size:.58rem;color:var(--muted);">
      R:R <span style="color:{rr_col};font-weight:600;">{rr_str}</span><br>RSI <span style="color:var(--text);">{_rsi:.1f}</span><br>{session}
    </div>
  </div>
  {dual_html}{warnings_html}
  <div class="sig-pips">
    <div class="pip-item">{pip(bd.get('ob_imbalance',0),4,14)} OB</div>
    <div class="pip-item">{pip(bd.get('funding',0)+bd.get('funding_hist',0),3,15)} FUNDING</div>
    <div class="pip-item">{pip(bd.get('oi_spike',0),5,14)} OI</div>
    <div class="pip-item">{pip(bd.get('vol_surge',0),3,10)} VOLUME</div>
    <div class="pip-item">{pip(bd.get('liq_cluster',0),4,12)} LIQ</div>
    <div class="pip-item">{pip(bd.get('whale_wall',0),4,7)} WHALE</div>
    <div class="pip-item">{pip(bd.get('technicals',0),5,12)} TECH</div>
    <div class="pip-item">{pip(bd.get('sentiment',0),8,20)} SENT</div>
    <div class="pip-item">{pip(bd.get('momentum',0),4,8)} MOM</div>
    <div class="pip-item">{pip(bd.get('social_buzz',0),3,7)} SOC</div>
    <div class="pip-item">{pip(max(0,bd.get('mtf',0)),6,12)} MTF</div>
    <div class="pip-item">{pip(bd.get('orderflow',0),6,12)} FLOW</div>
    <div class="pip-item">{pip(bd.get('liq_map',0),7,15)} LIQ-MAP</div>
    <div class="pip-item">{pip(bd.get('listing',0),6,25)} LISTING</div>
    <div class="pip-item">{pip(max(0,bd.get('onchain',0)),7,15)} ONCHAIN</div>
  </div>
  <div style="display:flex;height:5px;border-radius:3px;overflow:hidden;margin:6px 0;background:var(--panel);">
    <div style="width:{bid_pct:.0f}%;background:var(--green);"></div>
    <div style="width:{100-bid_pct:.0f}%;background:var(--red);"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-family:monospace;font-size:.56rem;color:var(--muted);margin-bottom:8px;">
    <span>BID {bid_pct:.0f}%</span><span>ASK {100-bid_pct:.0f}%</span>
  </div>
  {ob4h_html}
  <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;padding:6px 12px;">
    <span style="font-family:monospace;font-size:.55rem;color:#2563eb;font-weight:700;">📍 BEST ENTRY ZONE</span>
    <span style="font-family:monospace;font-size:.68rem;color:#1d4ed8;font-weight:700;margin-left:10px;">{_elo_d}</span>
    <span style="font-family:monospace;font-size:.55rem;color:#2563eb;margin:0 6px;">–</span>
    <span style="font-family:monospace;font-size:.68rem;color:#1d4ed8;font-weight:700;">{_ehi_d}</span>
    <span style="font-family:monospace;font-size:.5rem;color:#60a5fa;margin-left:8px;">wait for pullback into zone</span>
  </div>
  <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:6px;margin:8px 0;">
    <div class="px-cell"><div class="px-lbl">Entry</div><div class="px-val" style="color:var(--blue);font-size:.72rem;">${_price:.6f}</div><div style="font-family:monospace;font-size:.5rem;color:{pct_col};">{pct_display} 24h</div></div>
    <div class="px-cell" style="border-top:2px solid #86efac44;"><div class="px-lbl">TP1 <span style="color:#86efac;font-size:.5rem;">scalp</span></div><div class="px-val" style="color:#86efac;font-size:.68rem;">${_tp1:.6f}</div><div style="font-family:monospace;font-size:.5rem;color:var(--muted);">{tp1_pct}</div></div>
    <div class="px-cell" style="border-top:2px solid var(--green);"><div class="px-lbl">TP2 <span style="color:var(--green);font-size:.5rem;">target</span></div><div class="px-val" style="color:var(--green);font-size:.72rem;">${_tp2:.6f}</div><div style="font-family:monospace;font-size:.5rem;color:var(--muted);">{tp2_pct}</div></div>
    <div class="px-cell" style="border-top:2px solid #f59e0b;"><div class="px-lbl">TP3 <span style="color:#f59e0b;font-size:.5rem;">max run</span></div><div class="px-val" style="color:#f59e0b;font-size:.68rem;">${_tp3:.6f}</div><div style="font-family:monospace;font-size:.5rem;color:var(--muted);">{tp3_pct}</div></div>
    <div class="px-cell" style="border-top:2px solid var(--red);"><div class="px-lbl">Stop Loss</div><div class="px-val" style="color:var(--red);font-size:.72rem;">${_sl:.6f}</div><div style="font-family:monospace;font-size:.52rem;color:var(--muted);">{sl_pct}</div></div>
  </div>
  {whale_html}{liq_html}{social_html}{cmc_html}{sentiment_html}
  <div class="reasons-list" style="margin-top:8px;">{reasons_html}</div>
  <div style="margin-top:10px;display:flex;gap:12px;align-items:center;">
    <a href="{trade_link}" target="_blank" style="font-family:monospace;font-size:.62rem;color:var(--blue);text-decoration:none;font-weight:600;">Trade {res['symbol']} on {exch} →</a>
    <span style="font-family:monospace;font-size:.58rem;color:var(--muted);">Class: <b style="color:var(--text);">{sym_cls}</b></span>
  </div>
</div>"""
        card_html="\n".join(line.lstrip() for line in card_html.splitlines())
        st.markdown(card_html,unsafe_allow_html=True)
    except Exception as ce:
        st.error(f"Card render error [{res.get('symbol','?')}]: {ce}")
        st.code(f"{res.get('symbol','?')} | Score:{res.get('pump_score','?')} | {res.get('type','?')} | ${res.get('price',0):.6f}")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ APEX")
    st.caption("Pump & Dump Intelligence")
    # FIX: Use session state for nav to prevent journal→scanner glitch
    nav_options=["🔥 Scanner","⚙️ Settings","📒 Journal","📊 Backtest"]
    nav=st.radio("Navigation",nav_options,label_visibility="collapsed",
                 index=nav_options.index(st.session_state.get('nav_state','🔥 Scanner')))
    if nav!=st.session_state.nav_state:
        st.session_state.nav_state=nav
    st.divider()
    st.subheader("Quick Controls")
    q_depth=st.slider("Coins to Scan",10,100,S['scan_depth'],step=10,key="q_depth")
    q_min=st.slider("Min Score",1,80,S['min_score'],key="q_min")
    q_btc=st.toggle("BTC Bear blocks LONGs",S['btc_filter'],key="q_btc")
    q_mom=st.toggle("Require Momentum",S.get('require_momentum',False),key="q_mom")
    st.caption("Scan Focus (select multiple):")
    _saved_modes=S.get('scan_modes',['mixed'])
    if isinstance(_saved_modes,str): _saved_modes=[_saved_modes]
    q_mode_vol=st.checkbox("Volume",value='volume' in _saved_modes,key="q_vol")
    q_mode_gain=st.checkbox("Gainers",value='gainers' in _saved_modes,key="q_gain")
    q_mode_loss=st.checkbox("Losers",value='losers' in _saved_modes,key="q_loss")
    q_mode_mixed=st.checkbox("Mixed",value='mixed' in _saved_modes,key="q_mixed")
    selected_modes=([m for m,v in [('volume',q_mode_vol),('gainers',q_mode_gain),('losers',q_mode_loss),('mixed',q_mode_mixed)] if v]) or ['mixed']
    st.divider()
    st.subheader("Auto-Pilot")
    q_auto=st.toggle("Continuous Scan",S.get('auto_scan',False))
    q_auto_int=st.number_input("Interval (mins)",1,60,S.get('auto_interval',5))
    st.divider()
    if st.button("Clear Cooldowns"):
        if os.path.exists(COOLDOWN_FILE): os.remove(COOLDOWN_FILE)
        st.success("Done")
    st.subheader("Sentinel")
    st.caption("Scans TOP 100 coins continuously.")
    q_sentinel=st.toggle("Enable Sentinel",st.session_state.get("sentinel_active",False))
    if q_sentinel!=st.session_state.get("sentinel_active",False):
        st.session_state.sentinel_active=q_sentinel
        st.session_state.sentinel_total_checked=0; st.session_state.sentinel_signals_found=0
    if st.session_state.get("sentinel_active"):
        st.write(f"Checked: {st.session_state.get('sentinel_total_checked',0)} | Signals: {st.session_state.get('sentinel_signals_found',0)}")
    sn,sm,sc_=get_session()
    st.divider(); st.write(f"Session: **{sn}** ({sm}x)")


# ─── HEADER / TICKER ─────────────────────────────────────────────────────────
st.markdown('<div style="padding:18px 0 14px;"><div style="font-family:monospace;font-size:1.5rem;font-weight:700;color:#0f1117;">APEX</div><div style="font-family:monospace;font-size:.56rem;font-weight:400;letter-spacing:.16em;color:#7a82a0;text-transform:uppercase;margin-top:2px;">Pump & Dump Intelligence Terminal v3.0 — Dual Confirm + FVG + Backtest</div></div>',unsafe_allow_html=True)

if time.time()-st.session_state.get('fng_last_fetch',0)>300:
    try:
        fg=requests.get("https://api.alternative.me/fng/?limit=1",timeout=2).json()
        st.session_state.fng_val=int(fg['data'][0]['value']); st.session_state.fng_txt=fg['data'][0]['value_classification']
        st.session_state.fng_last_fetch=time.time()
    except: pass

fng_v=st.session_state.fng_val; fng_t=st.session_state.fng_txt
fng_c="#059669" if fng_v>=60 else ("#dc2626" if fng_v<=40 else "#d97706")
btc_c="#059669" if st.session_state.btc_trend=="BULLISH" else ("#dc2626" if st.session_state.btc_trend=="BEARISH" else "#7a82a0")
sn_,_,sc_now=get_session()
st.markdown(f"""<div class="ticker-bar">
  <div><div class="t-lbl">BTC</div><div class="t-val">${st.session_state.btc_price:,.0f} <span style="color:{btc_c};">{st.session_state.btc_trend}</span></div></div>
  <div><div class="t-lbl">Fear &amp; Greed</div><div class="t-val" style="color:{fng_c};">{fng_v} — {fng_t.upper()}</div></div>
  <div><div class="t-lbl">Session</div><div class="t-val" style="color:{sc_now};">{sn_}</div></div>
  <div><div class="t-lbl">Last Scan</div><div class="t-val">{st.session_state.last_scan}</div></div>
  <div><div class="t-lbl">Raw / Filtered</div><div class="t-val">{st.session_state.last_raw_count} / {len(st.session_state.results)}</div></div>
  <div><div class="t-lbl">Scans</div><div class="t-val">#{st.session_state.scan_count}</div></div>
  <div><div class="t-lbl">UTC</div><div class="t-val">{datetime.now(timezone.utc).strftime('%H:%M:%S')}</div></div>
</div>""",unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SETTINGS
# ═══════════════════════════════════════════════════════════════════════════
if nav=="⚙️ Settings":
    st.markdown('<div class="section-h">Settings — all thresholds and scoring weights</div>',unsafe_allow_html=True)
    with st.form("settings_form"):
        # ── ALERTS ────────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🔔 Notification Controls</div>',unsafe_allow_html=True)
        ac1,ac2,ac3=st.columns(3)
        with ac1: ns_alert_score=st.slider("Min Score for Alerts",10,100,S.get('alert_min_score',60))
        with ac2:
            st.markdown("**Types:**")
            ns_al_long=st.checkbox("Alert LONGs",S.get('alert_longs',True))
            ns_al_short=st.checkbox("Alert SHORTs",S.get('alert_shorts',True))
        with ac3:
            st.markdown("**Classes:**")
            ns_al_sq=st.checkbox("Squeeze",S.get('alert_squeeze',True))
            ns_al_br=st.checkbox("Breakout",S.get('alert_breakout',True))
            ns_al_wh=st.checkbox("Whale",S.get('alert_whale',True))
            ns_al_ea=st.checkbox("Early",S.get('alert_early',False))
        st.markdown("**Daily Summary Alert:**")
        ds1,ds2=st.columns(2)
        with ds1: ns_ds_on=st.toggle("Daily Journal Summary to Discord/Telegram",S.get('daily_summary_on',True))
        with ds2: ns_ds_hr=st.slider("Summary hour (UTC)",0,23,int(S.get('daily_summary_hour',8)))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── JOURNAL FILTERS (NEW) ─────────────────────────────────────────
        st.markdown('<div class="stg-card" style="border-color:#2563eb;"><div class="stg-title" style="color:#2563eb;">📒 Journal Logging Filters — What Gets Saved</div>',unsafe_allow_html=True)
        jf1,jf2,jf3=st.columns(3)
        with jf1:
            st.markdown("**Signal Classes to Log:**")
            j_cls_opts=["god_tier","squeeze","breakout","whale_driven","early"]
            saved_j_cls=S.get('j_filter_classes',j_cls_opts)
            ns_j_cls=st.multiselect("Classes",j_cls_opts,default=saved_j_cls,key="j_cls_ms")
            st.markdown('<div class="setting-help">Only signals matching these classes will be saved to journal</div>',unsafe_allow_html=True)
        with jf2:
            ns_j_min_sc=st.slider("Min score to journal",1,80,int(S.get('j_min_score',25)))
            st.markdown('<div class="setting-help">Signals below this score won\'t be journaled regardless of class</div>',unsafe_allow_html=True)
        with jf3:
            st.markdown("**Required Technicals (ALL must be present):**")
            tech_opts=["ob","funding","oi","volume","whale","sentiment","mtf","orderflow"]
            saved_j_tech=S.get('j_require_technicals',[])
            ns_j_tech=st.multiselect("Must have signals",tech_opts,default=saved_j_tech,key="j_tech_ms")
            st.markdown('<div class="setting-help">Leave empty = no checklist. Add items = ALL must be active for signal to be journaled</div>',unsafe_allow_html=True)
        st.markdown('</div>',unsafe_allow_html=True)

        # ── RISK FILTERS ──────────────────────────────────────────────────
        st.markdown('<div class="stg-card" style="border-color:#f59e0b;"><div class="stg-title" style="color:#92400e;">⚠️ Risk & Anti-False-Positive Filters</div>',unsafe_allow_html=True)
        rf1,rf2,rf3=st.columns(3)
        with rf1:
            ns_late_thresh=st.number_input("Late Entry threshold %",3.0,20.0,float(S.get('late_entry_chg_thresh',8.0)),0.5,format="%.1f")
            ns_late_pen=st.slider("Late Entry penalty pts",5,30,int(S.get('late_entry_penalty',20)))
        with rf2:
            ns_exhaust_pen=st.slider("Volume Exhaustion penalty pts",5,25,int(S.get('vol_exhaust_penalty',15)))
            ns_near_top_pen=st.slider("Near Local Top penalty pts",5,25,int(S.get('near_top_penalty',15)))
        with rf3:
            ns_use_4h_ob=st.toggle("Use 4H OB+FVG for SL",S.get('use_4h_ob_for_sl',True))
            st.info("FVG (Fair Value Gaps) on 4H are now also used as SL zones alongside Order Blocks.")
        st.markdown('</div>',unsafe_allow_html=True)

        # ── SCAN ──────────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🔍 Scan Configuration</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1: ns_depth=st.slider("Coins to scan",10,200,S['scan_depth'],step=10)
        with c2: ns_fast=st.selectbox("Signal TF",["1m","5m","15m","1h"],index=["1m","5m","15m","1h"].index(S['fast_tf']))
        with c3: ns_slow=st.selectbox("Trend TF",["1h","4h","1d"],index=["1h","4h","1d"].index(S['slow_tf']))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── FILTERS ───────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🎯 Core Filters</div>',unsafe_allow_html=True)
        c1,c2,c3,c4,c5,c6=st.columns(6)
        with c1: ns_min=st.slider("Min score to show",1,80,S['min_score'])
        with c2:
            ns_ji=st.checkbox("Log IMMINENT (70+)",S.get('j_imminent',True))
            ns_jb=st.checkbox("Log BUILDING (45+)",S.get('j_building',True))
            ns_je=st.checkbox("Log EARLY (25+)",S.get('j_early',False))
        with c3:
            ns_minr=st.slider("Min signals",1,6,S.get('min_reasons',1))
        with c4:
            ns_btc=st.toggle("BTC Bear block",S['btc_filter'])
            ns_mom=st.toggle("Require momentum",S.get('require_momentum',False))
        with c5:
            ns_cd=st.toggle("Symbol cooldown",S['cooldown_on'])
            ns_cdh=st.slider("Cooldown hrs",1,24,S['cooldown_hrs']) if ns_cd else S['cooldown_hrs']
        with c6:
            ns_min_rr=st.slider("Min R:R",1.0,4.0,float(S.get('min_rr',1.5)),step=0.1)
        st.markdown('</div>',unsafe_allow_html=True)

        # ── WHALE WALLS ───────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🐋 Whale Wall Detection</div>',unsafe_allow_html=True)
        c1,c2,c3,c4=st.columns(4)
        with c1: ns_whale=st.selectbox("Min wall size",[50000,100000,250000,500000,1000000,5000000],index=[50000,100000,250000,500000,1000000,5000000].index(S['whale_min_usdt']),format_func=lambda x:f"${x:,.0f}")
        with c2: ns_pts_wh_near=st.slider("Points ≤0.5%",10,35,S.get('pts_whale_near',25))
        with c3: ns_pts_wh_mid=st.slider("Points 0.5–1.5%",5,25,S.get('pts_whale_mid',15))
        with c4: ns_pts_wh_far=st.slider("Points 1.5–3%",3,15,S.get('pts_whale_far',8))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── ORDER BOOK ────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">⚖️ Order Book / Funding / OI / Volume</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            ns_ob_l=st.number_input("OB low ratio",0.5,3.0,S['ob_ratio_low'],0.05,format="%.2f"); ns_pts_ob_l=st.slider("OB pts low",1,15,S['pts_ob_low'])
            ns_ob_m=st.number_input("OB mid ratio",0.5,5.0,S['ob_ratio_mid'],0.1,format="%.2f"); ns_pts_ob_m=st.slider("OB pts mid",5,20,S['pts_ob_mid'])
            ns_ob_h=st.number_input("OB high ratio",1.0,10.0,S['ob_ratio_high'],0.25,format="%.2f"); ns_pts_ob_h=st.slider("OB pts high",10,30,S['pts_ob_high'])
        with c2:
            ns_fr_l=st.number_input("Funding low",0.00001,0.005,S['funding_low'],0.00005,format="%.5f"); ns_pts_fr_l=st.slider("Funding pts low",1,15,S['pts_funding_low'],key="pfrl")
            ns_fr_m=st.number_input("Funding mid",0.0001,0.01,S['funding_mid'],0.0001,format="%.5f"); ns_pts_fr_m=st.slider("Funding pts mid",5,20,S['pts_funding_mid'],key="pfrm")
            ns_fr_h=st.number_input("Funding high",0.0005,0.05,S['funding_high'],0.0005,format="%.5f"); ns_pts_fr_h=st.slider("Funding pts high",10,30,S['pts_funding_high'],key="pfrh")
        with c3:
            ns_oi_l=st.number_input("OI price chg low%",0.1,5.0,S['oi_price_chg_low'],0.1,format="%.1f"); ns_pts_oi_l=st.slider("OI pts low",1,15,S['pts_oi_low'],key="poil")
            ns_oi_h=st.number_input("OI price chg high%",0.5,10.0,S['oi_price_chg_high'],0.5,format="%.1f"); ns_pts_oi_h=st.slider("OI pts high",5,25,S['pts_oi_high'],key="poih")
            ns_vs_l=st.number_input("Vol surge low×",1.0,3.0,S['vol_surge_low'],0.1,format="%.1f"); ns_pts_vs_l=st.slider("Vol pts low",1,10,S['pts_vol_low'],key="pvl")
            ns_vs_m=st.number_input("Vol surge mid×",1.5,5.0,S['vol_surge_mid'],0.25,format="%.1f"); ns_pts_vs_m=st.slider("Vol pts mid",3,15,S['pts_vol_mid'],key="pvm")
            ns_vs_h=st.number_input("Vol surge high×",2.0,10.0,S['vol_surge_high'],0.5,format="%.1f"); ns_pts_vs_h=st.slider("Vol pts high",5,20,S['pts_vol_high'],key="pvh")
        st.markdown('</div>',unsafe_allow_html=True)

        # ── LIQ CLUSTERS / TECHNICALS / SENTIMENT ─────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🧲 Liq / Tech / Sentiment</div>',unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        with c1:
            ns_lq_n=st.number_input("Liq near %",0.5,5.0,S['liq_cluster_near'],0.25,format="%.2f"); ns_pts_lq_n=st.slider("Pts near",5,20,S['pts_liq_near'],key="pln")
            ns_lq_m=st.number_input("Liq mid %",1.0,10.0,S['liq_cluster_mid'],0.5,format="%.1f"); ns_pts_lq_m=st.slider("Pts mid",3,15,S['pts_liq_mid'],key="plm")
            ns_lq_f=st.number_input("Liq far %",2.0,20.0,S['liq_cluster_far'],1.0,format="%.1f"); ns_pts_lq_f=st.slider("Pts far",1,10,S['pts_liq_far'],key="plf")
        with c2:
            ns_rsi_os=st.slider("RSI oversold (LONG)",20,55,S['rsi_oversold'])
            ns_rsi_ob=st.slider("RSI overbought (SHORT)",45,80,S['rsi_overbought'])
            ns_pts_macd=st.slider("MACD pts",1,10,S['pts_macd']); ns_pts_rsi=st.slider("RSI pts",1,10,S['pts_rsi'])
            ns_pts_bb=st.slider("BB pts",1,10,S['pts_bb']); ns_pts_ema=st.slider("EMA pts",1,8,S['pts_ema'])
        with c3:
            ns_pts_sent=st.slider("Sentiment pts",5,30,S.get('pts_sentiment',20))
            ns_pts_taker=st.slider("Taker pts",3,20,S.get('pts_taker',10))
            ns_v24_l=st.number_input("24h vol low $",100000,10000000,S['vol24h_low'],100000,format="%d"); ns_pts_v24_l=st.slider("Pts 24h low",1,8,S['pts_vol24_low'],key="pv24l")
            ns_v24_m=st.number_input("24h vol mid $",1000000,50000000,S['vol24h_mid'],1000000,format="%d"); ns_pts_v24_m=st.slider("Pts 24h mid",2,12,S['pts_vol24_mid'],key="pv24m")
            ns_v24_h=st.number_input("24h vol high $",5000000,200000000,S['vol24h_high'],5000000,format="%d"); ns_pts_v24_h=st.slider("Pts 24h high",4,15,S['pts_vol24_high'],key="pv24h")
        st.markdown('</div>',unsafe_allow_html=True)

        # ── ACCURACY & INTELLIGENCE ────────────────────────────────────────
        st.markdown('<div class="stg-card" style="border-color:#059669;"><div class="stg-title" style="color:#059669;">🎯 Accuracy + Intelligence Signals</div>',unsafe_allow_html=True)
        af1,af2,af3=st.columns(3)
        with af1:
            ns_min_vol=st.number_input("Min 24h vol $",0,5000000,int(S.get('min_vol_filter',300000)),50000,format="%d")
            ns_min_sigs=st.slider("Min signal cats",1,6,int(S.get('min_active_signals',3)))
            ns_dedup=st.toggle("Dedup coins",S.get('dedup_symbols',True))
            ns_atr_min=st.number_input("ATR min %",0.0,2.0,float(S.get('atr_min_pct',0.2)),0.05,format="%.2f")
            ns_atr_max=st.number_input("ATR max %",2.0,20.0,float(S.get('atr_max_pct',10.0)),0.5,format="%.1f")
            ns_spread_max=st.number_input("Max spread %",0.0,2.0,float(S.get('spread_max_pct',0.5)),0.05,format="%.2f")
        with af2:
            ns_fng_lt=st.slider("F&G LONG min",10,45,int(S.get('fng_long_threshold',30)))
            ns_fng_st=st.slider("F&G SHORT min",55,90,int(S.get('fng_short_threshold',70)))
            ns_mtf=st.toggle("MTF confirmation",S.get('mtf_confirm',True))
            ns_pts_mtf=st.slider("MTF pts",0,20,int(S.get('pts_mtf',12)))
            ns_pts_div=st.slider("RSI div pts",0,20,int(S.get('pts_divergence',10)))
            ns_pts_cpat=st.slider("Candle pattern pts",0,15,int(S.get('pts_candle_pattern',8)))
        with af3:
            ns_pts_combo=st.slider("OI+Funding combo pts",0,20,int(S.get('pts_oi_funding_combo',10)))
            ns_vol_exp_t=st.number_input("Explosive vol ×avg",3.0,20.0,float(S.get('vol_surge_explosive',5.0)),0.5,format="%.1f")
            ns_pts_vol_e=st.slider("Explosive vol pts",10,30,int(S.get('pts_vol_explosive',20)))
            ns_of_lb=st.slider("Order flow lookback",5,30,int(S.get('orderflow_lookback',10)))
            ns_pts_of=st.slider("Order flow pts",3,20,int(S.get('pts_orderflow',12)))
            ns_pts_lm=st.slider("Liq map pts",3,25,int(S.get('pts_liq_map',15)))
            ns_pts_lst=st.slider("New listing pts",5,35,int(S.get('listing_alert_pts',25)))
            ns_oc_min=st.number_input("On-chain whale min $",100000,5000000,int(S.get('onchain_whale_min',500000)),100000,format="%d")
            ns_pts_oc=st.slider("On-chain pts",3,25,int(S.get('pts_onchain_whale',15)))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── SENTINEL ──────────────────────────────────────────────────────
        st.markdown('<div class="stg-card" style="border-color:#7c3aed;"><div class="stg-title" style="color:#7c3aed;">🛰️ Sentinel Mode</div>',unsafe_allow_html=True)
        sc1,sc2,sc3=st.columns(3)
        with sc1: ns_sent_score=st.slider("Alert threshold",40,95,int(S.get('sentinel_score_threshold',70)))
        with sc2: ns_sent_batch=st.slider("Coins per batch",2,20,int(S.get('sentinel_batch_size',5)))
        with sc3: ns_sent_interval=st.slider("Secs between batches",5,120,int(S.get('sentinel_check_interval',30)))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── CLASSIFIER ────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🏷️ Tab Classifier Thresholds</div>',unsafe_allow_html=True)
        _ca,_cb,_cc=st.columns(3)
        with _ca:
            st.caption('🟡 Breakout')
            ns_br_oi=st.slider('Min OI spike pts',1,20,int(S.get('cls_breakout_oi_min',7)),key='br_oi')
            ns_br_vol=st.slider('Min Vol surge pts',1,15,int(S.get('cls_breakout_vol_min',4)),key='br_vol')
            ns_br_sc=st.slider('Min score',10,60,int(S.get('cls_breakout_score_min',25)),key='br_sc')
        with _cb:
            st.caption('🔴 Squeeze')
            ns_sq_fd=st.slider('Min Funding pts',1,20,int(S.get('cls_squeeze_fund_min',8)),key='sq_fd')
            ns_sq_ob=st.slider('Min OB pts',1,15,int(S.get('cls_squeeze_ob_min',6)),key='sq_ob')
        with _cc:
            st.caption('ℹ️ Note'); st.info('Unmatched → Early tab.')
        st.markdown('</div>',unsafe_allow_html=True)

        # ── AUTO-JOURNAL CHECK ────────────────────────────────────────────
        st.markdown('<div class="stg-card" style="border-color:#059669;"><div class="stg-title" style="color:#059669;">📒 Auto-Journal Exit Tracking</div>',unsafe_allow_html=True)
        aj1,aj2=st.columns(2)
        with aj1:
            ns_aj_on=st.toggle("Enable Auto-Journal Tracking",S.get('journal_autocheck_on',True))
            ns_aj_mins=st.slider("Check interval (mins)",1,60,int(S.get('journal_autocheck_mins',15)))
        with aj2:
            ns_aj_force=st.checkbox("🔄 Force check on Save",value=False)
        st.markdown('</div>',unsafe_allow_html=True)

        # ── SOCIAL ────────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">📡 Social Media</div>',unsafe_allow_html=True)
        so1,so2,so3,so4=st.columns(4)
        with so1: ns_social_en=st.toggle("Enable Reddit Buzz",S.get('social_enabled',True))
        with so2: ns_social_wt=st.slider("Max Reddit pts",3,20,int(S.get('social_reddit_weight',8)))
        with so3: ns_social_min=st.slider("Min mentions",1,10,int(S.get('social_min_mentions',3)))
        with so4: ns_social_buzz=st.slider("Mentions for max pts",5,50,int(S.get('social_buzz_threshold',10)))
        ns_apify=st.text_input("Apify Token (optional)",S.get('apify_token',''),type="password")
        st.markdown('</div>',unsafe_allow_html=True)

        # ── BACKTEST ──────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">📊 Backtest Defaults</div>',unsafe_allow_html=True)
        bt1,bt2=st.columns(2)
        with bt1: ns_bt_min_sc=st.slider("Backtest min score",1,100,int(S.get('backtest_min_score',50)))
        with bt2: ns_bt_days=st.slider("Backtest days",7,90,int(S.get('backtest_days',30)))
        st.markdown('</div>',unsafe_allow_html=True)

        # ── APIs ──────────────────────────────────────────────────────────
        st.markdown('<div class="stg-card"><div class="stg-title">🔑 API Keys</div>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ns_tg_tok=st.text_input("Telegram Bot Token",S.get('tg_token',''),type="password")
            ns_tg_cid=st.text_input("Telegram Chat ID",S.get('tg_chat_id',''))
            ns_okx_key=st.text_input("OKX API Key",S.get('okx_key',''),type="password")
            ns_okx_sec=st.text_input("OKX API Secret",S.get('okx_secret',''),type="password")
            ns_okx_pass=st.text_input("OKX Passphrase",S.get('okx_passphrase',''),type="password")
        with c2:
            ns_dc_web=st.text_input("Discord Webhook URL",S.get('discord_webhook',''),type="password")
            ns_cmc=st.text_input("CMC API Key",S.get('cmc_key',''),type="password")
            ns_gate_key=st.text_input("Gate.io API Key",S.get('gate_key',''),type="password")
            ns_gate_sec=st.text_input("Gate.io API Secret",S.get('gate_secret',''),type="password")
        st.markdown('</div>',unsafe_allow_html=True)

        submitted=st.form_submit_button("💾 SAVE ALL SETTINGS",use_container_width=True)
        if submitted:
            new_s={
                'scan_depth':ns_depth,'fast_tf':ns_fast,'slow_tf':ns_slow,'scan_modes':selected_modes,
                'min_score':ns_min,'j_imminent':ns_ji,'j_building':ns_jb,'j_early':ns_je,
                'j_filter_classes':ns_j_cls,'j_min_score':ns_j_min_sc,'j_require_technicals':ns_j_tech,
                'min_reasons':ns_minr,'btc_filter':ns_btc,'require_momentum':ns_mom,
                'cooldown_on':ns_cd,'cooldown_hrs':ns_cdh,'whale_min_usdt':ns_whale,'min_rr':ns_min_rr,
                'alert_min_score':ns_alert_score,'alert_longs':ns_al_long,'alert_shorts':ns_al_short,
                'alert_squeeze':ns_al_sq,'alert_breakout':ns_al_br,'alert_early':ns_al_ea,'alert_whale':ns_al_wh,
                'daily_summary_on':ns_ds_on,'daily_summary_hour':ns_ds_hr,
                'late_entry_chg_thresh':ns_late_thresh,'late_entry_penalty':ns_late_pen,
                'vol_exhaust_penalty':ns_exhaust_pen,'near_top_penalty':ns_near_top_pen,'use_4h_ob_for_sl':ns_use_4h_ob,
                'ob_ratio_low':ns_ob_l,'ob_ratio_mid':ns_ob_m,'ob_ratio_high':ns_ob_h,
                'pts_ob_low':ns_pts_ob_l,'pts_ob_mid':ns_pts_ob_m,'pts_ob_high':ns_pts_ob_h,
                'funding_low':ns_fr_l,'funding_mid':ns_fr_m,'funding_high':ns_fr_h,
                'pts_funding_low':ns_pts_fr_l,'pts_funding_mid':ns_pts_fr_m,'pts_funding_high':ns_pts_fr_h,
                'oi_price_chg_low':ns_oi_l,'oi_price_chg_high':ns_oi_h,'pts_oi_low':ns_pts_oi_l,'pts_oi_high':ns_pts_oi_h,
                'vol_surge_low':ns_vs_l,'vol_surge_mid':ns_vs_m,'vol_surge_high':ns_vs_h,
                'pts_vol_low':ns_pts_vs_l,'pts_vol_mid':ns_pts_vs_m,'pts_vol_high':ns_pts_vs_h,
                'liq_cluster_near':ns_lq_n,'liq_cluster_mid':ns_lq_m,'liq_cluster_far':ns_lq_f,
                'pts_liq_near':ns_pts_lq_n,'pts_liq_mid':ns_pts_lq_m,'pts_liq_far':ns_pts_lq_f,
                'vol24h_low':ns_v24_l,'vol24h_mid':ns_v24_m,'vol24h_high':ns_v24_h,
                'pts_vol24_low':ns_pts_v24_l,'pts_vol24_mid':ns_pts_v24_m,'pts_vol24_high':ns_pts_v24_h,
                'rsi_oversold':ns_rsi_os,'rsi_overbought':ns_rsi_ob,
                'pts_macd':ns_pts_macd,'pts_rsi':ns_pts_rsi,'pts_bb':ns_pts_bb,'pts_ema':ns_pts_ema,
                'pts_session':S['pts_session'],'pts_sentiment':ns_pts_sent,'pts_taker':ns_pts_taker,
                'pts_whale_near':ns_pts_wh_near,'pts_whale_mid':ns_pts_wh_mid,'pts_whale_far':ns_pts_wh_far,
                'min_vol_filter':ns_min_vol,'min_active_signals':ns_min_sigs,'dedup_symbols':ns_dedup,
                'atr_min_pct':ns_atr_min,'atr_max_pct':ns_atr_max,'spread_max_pct':ns_spread_max,
                'fng_long_threshold':ns_fng_lt,'fng_short_threshold':ns_fng_st,'mtf_confirm':ns_mtf,
                'pts_mtf':ns_pts_mtf,'pts_divergence':ns_pts_div,'pts_candle_pattern':ns_pts_cpat,
                'pts_oi_funding_combo':ns_pts_combo,'vol_surge_explosive':ns_vol_exp_t,'pts_vol_explosive':ns_pts_vol_e,
                'sentinel_score_threshold':ns_sent_score,'sentinel_batch_size':ns_sent_batch,'sentinel_check_interval':ns_sent_interval,
                'cls_breakout_oi_min':ns_br_oi,'cls_breakout_vol_min':ns_br_vol,'cls_breakout_score_min':ns_br_sc,
                'cls_squeeze_fund_min':ns_sq_fd,'cls_squeeze_ob_min':ns_sq_ob,
                'journal_autocheck_on':ns_aj_on,'journal_autocheck_mins':ns_aj_mins,
                'orderflow_lookback':ns_of_lb,'pts_orderflow':ns_pts_of,'pts_liq_map':ns_pts_lm,
                'listing_alert_pts':ns_pts_lst,'onchain_whale_min':ns_oc_min,'pts_onchain_whale':ns_pts_oc,
                'social_enabled':ns_social_en,'social_reddit_weight':ns_social_wt,
                'social_min_mentions':ns_social_min,'social_buzz_threshold':ns_social_buzz,'apify_token':ns_apify,
                'backtest_min_score':ns_bt_min_sc,'backtest_days':ns_bt_days,
                'cmc_key':ns_cmc,'tg_token':ns_tg_tok,'tg_chat_id':ns_tg_cid,'discord_webhook':ns_dc_web,
                'okx_key':ns_okx_key,'okx_secret':ns_okx_sec,'okx_passphrase':ns_okx_pass,
                'gate_key':ns_gate_key,'gate_secret':ns_gate_sec,
                'auto_scan':q_auto,'auto_interval':q_auto_int,
            }
            save_settings(new_s)
            if ns_aj_force: st.session_state.journal_last_autocheck=0
            st.success("✅ Settings saved!"); st.balloons()
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: JOURNAL
# ═══════════════════════════════════════════════════════════════════════════
if nav=="📒 Journal":
    ensure_journal()
    col1,col2=st.columns([5,1])
    with col1: st.markdown('<div class="section-h">Trade Journal & Tracking Dashboard</div>',unsafe_allow_html=True)
    with col2:
        if st.button("🗑️ Clear"):
            if os.path.exists(JOURNAL_FILE): os.remove(JOURNAL_FILE)
            st.rerun()
    try: df_j=pd.read_csv(JOURNAL_FILE)
    except: df_j=pd.DataFrame()
    if df_j.empty:
        st.markdown('<div class="empty-st">📒 No signals logged yet</div>',unsafe_allow_html=True)
    else:
        df_j['ts']=pd.to_datetime(df_j['ts'],errors='coerce')
        now=datetime.now(); df24=df_j[df_j['ts']>=(now-timedelta(hours=24))]
        longs=len(df24[df24['type']=='LONG']); shorts=len(df24[df24['type']=='SHORT'])
        tps=len(df24[df24['status']=='TP']); sls=len(df24[df24['status']=='SL'])
        active=len(df24[df24['status']=='ACTIVE']); wr=(tps/(tps+sls)*100) if (tps+sls)>0 else 0
        untouched=len(df24[(df24['status']=='ACTIVE') & (df24.get('entry_touched',pd.Series(['0']*len(df24)))!='1')]) if 'entry_touched' in df24.columns else 0
        st.info(f"**Journal Logging Rules:** Classes: {S.get('j_filter_classes',['all'])} | Min score: {S.get('j_min_score',25)} | Required technicals: {S.get('j_require_technicals',[]) or 'None'} — Change in ⚙️ Settings")
        st.markdown(f"""<div class="stat-strip">
          <div><div class="ss-val">{len(df24)}</div><div class="ss-lbl">24h Total</div></div>
          <div><div class="ss-val">{longs}/{shorts}</div><div class="ss-lbl">Long/Short</div></div>
          <div><div class="ss-val" style="color:var(--amber);">{active}</div><div class="ss-lbl">Active</div></div>
          <div><div class="ss-val" style="color:#94a3b8;">{untouched}</div><div class="ss-lbl">Awaiting Entry</div></div>
          <div><div class="ss-val" style="color:var(--green);">{tps}</div><div class="ss-lbl">TP Hits</div></div>
          <div><div class="ss-val" style="color:var(--red);">{sls}</div><div class="ss-lbl">SL Hits</div></div>
          <div><div class="ss-val" style="color:var(--blue);">{wr:.1f}%</div><div class="ss-lbl">Win Rate</div></div>
        </div>""",unsafe_allow_html=True)
        st.caption("⚡ TP/SL tracking starts only after price touches the entry zone (within 0.5% of logged entry price)")
        fc1,fc2,fc3,fc4,fc5=st.columns(5)
        with fc1: jt=st.selectbox("Type",["ALL","LONG","SHORT"])
        with fc2: jc=st.selectbox("Class",["ALL","squeeze","breakout","whale_driven","early"])
        with fc3: js_stat=st.selectbox("Status",["ALL","ACTIVE","TP","SL"])
        with fc4: jx=st.selectbox("Exchange",["ALL","OKX","GATE","MEXC"])
        with fc5: js=st.selectbox("Sort",["ts","pump_score","symbol"])
        dv=df_j.copy()
        if jt!="ALL" and 'type' in dv.columns: dv=dv[dv['type']==jt]
        if jc!="ALL" and 'class' in dv.columns: dv=dv[dv['class']==jc]
        if js_stat!="ALL" and 'status' in dv.columns: dv=dv[dv['status']==js_stat]
        if jx!="ALL" and 'exchange' in dv.columns: dv=dv[dv['exchange']==jx]
        if js in dv.columns: dv=dv.sort_values(js,ascending=(js!='pump_score'))
        st.dataframe(dv,use_container_width=True,height=500)
        st.download_button("⬇️ Export CSV",dv.to_csv(index=False).encode(),
            file_name=f"apex_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: BACKTEST
# ═══════════════════════════════════════════════════════════════════════════
if nav=="📊 Backtest":
    st.markdown('<div class="section-h">Backtest — Replay Journal Signals Against Historical OHLCV</div>',unsafe_allow_html=True)
    st.info("📊 **How it works:** Select a min score + date range. For each journal signal, we fetch historical OHLCV from OKX (or Gate as fallback) and check if TP or SL was hit first after signal time.")
    if not os.path.exists(JOURNAL_FILE):
        st.warning("No journal file found. Run some scans first to populate the journal.")
        st.stop()
    try: df_bt=pd.read_csv(JOURNAL_FILE)
    except: df_bt=pd.DataFrame()
    if df_bt.empty:
        st.warning("Journal is empty — run scans first."); st.stop()

    bt1,bt2,bt3,bt4=st.columns(4)
    with bt1: bt_min_sc=st.slider("Min score",1,100,int(S.get('backtest_min_score',50)))
    with bt2: bt_days=st.slider("Days back",7,90,int(S.get('backtest_days',30)))
    with bt3: bt_cls=st.selectbox("Class filter",["ALL","squeeze","breakout","whale_driven","early"])
    with bt4: bt_type=st.selectbox("Type filter",["ALL","LONG","SHORT"])

    if st.button("▶️ RUN BACKTEST",use_container_width=True):
        df_bt['ts']=pd.to_datetime(df_bt['ts'],errors='coerce')
        cutoff=datetime.now()-timedelta(days=bt_days)
        mask=(df_bt['ts']>=cutoff) & (pd.to_numeric(df_bt['pump_score'],errors='coerce')>=bt_min_sc)
        if bt_cls!="ALL" and 'class' in df_bt.columns: mask=mask & (df_bt['class']==bt_cls)
        if bt_type!="ALL": mask=mask & (df_bt['type']==bt_type)
        df_sub=df_bt[mask].copy()
        if df_sub.empty:
            st.warning("No signals match the filters."); st.stop()
        st.info(f"Backtesting {len(df_sub)} signals...")
        results_bt=[]
        prog=st.progress(0); status_ph=st.empty()

        async def _bt_fetch(sym, exch, tf, ts_str, tp, sl, sig_type, entry_price):
            """Fetch OHLCV after signal time and determine TP/SL hit"""
            try:
                from datetime import datetime as _dt
                ts_dt=_dt.fromisoformat(str(ts_str)) if ts_str else None
                if ts_dt is None: return None
                ts_ms=int(ts_dt.timestamp()*1000)
                # Build ccxt exchange
                if exch=="OKX":
                    ex=ccxt.okx({'enableRateLimit':True,'rateLimit':100,'timeout':8000,'options':{'defaultType':'swap'}})
                    sym_ccxt=f"{sym}/USDT:USDT"
                elif exch=="GATE":
                    ex=ccxt.gateio({'enableRateLimit':True,'rateLimit':100,'timeout':8000,'options':{'defaultType':'swap'}})
                    sym_ccxt=f"{sym}/USDT:USDT"
                else:  # MEXC
                    ex=ccxt.mexc({'enableRateLimit':True,'rateLimit':100,'timeout':8000,'options':{'defaultType':'swap'}})
                    sym_ccxt=f"{sym}/USDT:USDT"
                await ex.load_markets()
                raw=await ex.fetch_ohlcv(sym_ccxt,tf,since=ts_ms,limit=200)
                await ex.close()
                if not raw or len(raw)<5: return None
                df_r=pd.DataFrame(raw,columns=['ts','open','high','low','close','volume'])
                # Simulate: find first candle after signal where price touches entry zone (within 1.5%)
                entry_touched_at=None
                for idx,row in df_r.iterrows():
                    if abs(float(row['close'])-float(entry_price))/float(entry_price)*100<=1.5:
                        entry_touched_at=idx; break
                if entry_touched_at is None: return {'result':'NOT_TRIGGERED','sym':sym,'score':0,'type':sig_type}
                # From entry touch, scan for TP or SL hit
                for idx2,row2 in df_r.iloc[entry_touched_at:].iterrows():
                    h=float(row2['high']); l=float(row2['low'])
                    if sig_type=="LONG":
                        if h>=float(tp): return {'result':'TP','sym':sym,'entry':entry_price,'tp':tp,'sl':sl,'type':sig_type,'hit_price':tp}
                        if l<=float(sl): return {'result':'SL','sym':sym,'entry':entry_price,'tp':tp,'sl':sl,'type':sig_type,'hit_price':sl}
                    else:
                        if l<=float(tp): return {'result':'TP','sym':sym,'entry':entry_price,'tp':tp,'sl':sl,'type':sig_type,'hit_price':tp}
                        if h>=float(sl): return {'result':'SL','sym':sym,'entry':entry_price,'tp':tp,'sl':sl,'type':sig_type,'hit_price':sl}
                return {'result':'OPEN','sym':sym,'entry':entry_price,'tp':tp,'sl':sl,'type':sig_type,'hit_price':0}
            except Exception as e: return {'result':'ERROR','sym':sym,'error':str(e)[:50]}

        total=len(df_sub)
        for i,(idx,row) in enumerate(df_sub.iterrows()):
            status_ph.text(f"Backtesting {i+1}/{total}: {row.get('symbol','?')}...")
            try:
                r=asyncio.run(_bt_fetch(
                    str(row.get('symbol','')),str(row.get('exchange','OKX')),"15m",
                    row.get('ts'),row.get('tp',0),row.get('sl',0),
                    row.get('type','LONG'),row.get('price',0)))
                if r:
                    r['score']=int(row.get('pump_score',0)); r['cls']=str(row.get('class',''))
                    r['ts']=str(row.get('ts','')); results_bt.append(r)
            except Exception as e: results_bt.append({'result':'ERROR','sym':str(row.get('symbol','?')),'error':str(e)[:50],'score':0,'cls':'','type':'','ts':''})
            prog.progress((i+1)/total)

        status_ph.empty(); prog.empty()
        if not results_bt:
            st.warning("No backtest results."); st.stop()

        df_res=pd.DataFrame(results_bt)
        tp_count=len(df_res[df_res['result']=='TP']); sl_count=len(df_res[df_res['result']=='SL'])
        open_count=len(df_res[df_res['result']=='OPEN']); nt_count=len(df_res[df_res['result']=='NOT_TRIGGERED'])
        err_count=len(df_res[df_res['result']=='ERROR'])
        total_closed=tp_count+sl_count; wr=(tp_count/total_closed*100) if total_closed>0 else 0

        # Calculate avg RR
        avg_rr=0
        rr_vals=[]
        for _,r in df_res.iterrows():
            if r.get('result')=='TP' and r.get('entry',0) and r.get('tp',0) and r.get('sl',0):
                try:
                    rr=abs(float(r['tp'])-float(r['entry']))/abs(float(r['entry'])-float(r['sl']))
                    rr_vals.append(rr)
                except: pass
        if rr_vals: avg_rr=sum(rr_vals)/len(rr_vals)

        st.markdown(f"""<div class="stat-strip">
          <div><div class="ss-val">{total}</div><div class="ss-lbl">Total Signals</div></div>
          <div><div class="ss-val" style="color:var(--green);">{tp_count}</div><div class="ss-lbl">TP Hits</div></div>
          <div><div class="ss-val" style="color:var(--red);">{sl_count}</div><div class="ss-lbl">SL Hits</div></div>
          <div><div class="ss-val" style="color:var(--blue);">{wr:.1f}%</div><div class="ss-lbl">Win Rate</div></div>
          <div><div class="ss-val" style="color:var(--amber);">{avg_rr:.2f}:1</div><div class="ss-lbl">Avg R:R (wins)</div></div>
          <div><div class="ss-val" style="color:var(--muted);">{open_count}</div><div class="ss-lbl">Still Open</div></div>
          <div><div class="ss-val" style="color:var(--muted);">{nt_count}</div><div class="ss-lbl">Not Triggered</div></div>
        </div>""",unsafe_allow_html=True)

        # Breakdown by class
        if 'cls' in df_res.columns:
            st.markdown("**Results by Class:**")
            cl_tab=df_res[df_res['result'].isin(['TP','SL'])].groupby('cls')['result'].value_counts().unstack(fill_value=0)
            if not cl_tab.empty: st.dataframe(cl_tab,use_container_width=True)

        st.markdown("**All Results:**")
        disp_cols=[c for c in ['ts','sym','type','cls','score','result','entry','tp','sl','hit_price','error'] if c in df_res.columns]
        st.dataframe(df_res[disp_cols],use_container_width=True,height=400)
        st.download_button("⬇️ Export Backtest CSV",df_res.to_csv(index=False).encode(),
            file_name=f"apex_backtest_{datetime.now().strftime('%Y%m%d')}.csv",mime="text/csv")
    st.stop()
# ─── AUTO-JOURNAL CHECK ──────────────────────────────────────────────────────
autocheck_journal_background(S)
check_daily_summary(S)

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: SCANNER
# ═══════════════════════════════════════════════════════════════════════════
eff_s=S.copy()
eff_s.update({'scan_depth':q_depth,'min_score':q_min,'btc_filter':q_btc,'require_momentum':q_mom,
              'auto_scan':q_auto,'auto_interval':q_auto_int,'scan_modes':selected_modes})

col_btn,_=st.columns([2,5])
with col_btn: do_scan=st.button("⚡  RUN PUMP/DUMP SCAN",use_container_width=True)
if eff_s.get('auto_scan'): do_scan=True; time.sleep(0.3)

if do_scan:
    try:
        screener=PrePumpScreener(
            cmc_key=eff_s.get('cmc_key',''),okx_key=eff_s.get('okx_key',''),
            okx_secret=eff_s.get('okx_secret',''),okx_passphrase=eff_s.get('okx_passphrase',''),
            gate_key=eff_s.get('gate_key',''),gate_secret=eff_s.get('gate_secret',''))
        raw_results,btc_t,btc_p,btc_r,scan_errs=asyncio.run(screener.run(eff_s))
        st.session_state.last_raw_count=len(raw_results)
        st.session_state.prev_results={f"{r['symbol']}_{r['type']}":r['pump_score'] for r in st.session_state.results}
        st.session_state.results=[r for r in raw_results if r['pump_score']>=eff_s['min_score']]
        st.session_state.last_scan=datetime.now().strftime('%H:%M:%S')
        st.session_state.scan_count+=1
        st.session_state.btc_price=btc_p; st.session_state.btc_trend=btc_t
        st.session_state.scan_errors=scan_errs

        # ── DUAL CONFIRM: mark coins flagged by both scanner + sentinel ───
        sentinel_syms={r['symbol'] for r in st.session_state.get('sentinel_results',[])}
        for r in st.session_state.results:
            prev=st.session_state.prev_results
            sym_key=f"{r['symbol']}_{r['type']}"
            r['is_new']=sym_key not in prev
            r['score_jump']=r['pump_score']-prev.get(sym_key,0) if prev.get(sym_key) else 0
            # DUAL CONFIRM logic: same coin in sentinel + scanner = score boost
            if r['symbol'] in sentinel_syms:
                r['dual_confirmed']=True
                r['pump_score']=min(100,r['pump_score']+10)
                if '🔥 DUAL CONFIRMED by Sentinel' not in r['reasons']:
                    r['reasons'].insert(0,'🔥 DUAL CONFIRMED by Sentinel — Scanner + Sentinel both flagged this coin (high conviction)')
            else:
                r['dual_confirmed']=False

        for r in st.session_state.results:
            ak=f"{r['symbol']}_{r['type']}_{datetime.now().hour}"
            lbl=pump_label(r['pump_score'],r['type'])
            score_bracket=r['pump_score']//15
            ak_recheck=f"{r['symbol']}_{r['type']}_{datetime.now().hour}_{score_bracket}"

            # ── JOURNAL LOGGING ──────────────────────────────────────────
            if ak not in st.session_state.logged_sigs:
                log_trade(r)  # log_trade handles all filter logic internally
                st.session_state.logged_sigs.add(ak)

            # ── NOTIFICATION CHECK ────────────────────────────────────────
            send_alert=False
            if r['pump_score']>=eff_s.get('alert_min_score',60):
                type_ok=(r['type']=='LONG' and eff_s.get('alert_longs',True)) or (r['type']=='SHORT' and eff_s.get('alert_shorts',True))
                cls_ok=((r['cls']=='squeeze' and eff_s.get('alert_squeeze',True)) or
                        (r['cls']=='breakout' and eff_s.get('alert_breakout',True)) or
                        (r['cls']=='whale_driven' and eff_s.get('alert_whale',True)) or
                        (r['cls']=='early' and eff_s.get('alert_early',False)) or
                        r['cls'] not in ('squeeze','breakout','whale_driven','early'))
                if type_ok and cls_ok: send_alert=True
            # Force alert for dual-confirmed high-conviction signals
            if r.get('dual_confirmed') and r['pump_score']>=60: send_alert=True

            if send_alert:
                if 'alerted_sigs' not in st.session_state: st.session_state.alerted_sigs=set()
                if ak_recheck not in st.session_state.alerted_sigs:
                    bd_r=r.get('signal_breakdown',{})
                    sentiment=r.get('sentiment',{})
                    def epip(v,lo=5,hi=12):
                        if v>=hi: return "🟩"
                        if v>=lo: return "🟨"
                        return "⬛"
                    pips_str=(f"OB {epip(bd_r.get('ob_imbalance',0),4,14)} | FD {epip(bd_r.get('funding',0)+bd_r.get('funding_hist',0),3,15)} | OI {epip(bd_r.get('oi_spike',0),5,14)}\n"
                              f"VOL {epip(bd_r.get('vol_surge',0),3,10)} | LQ {epip(bd_r.get('liq_cluster',0),4,12)} | WH {epip(bd_r.get('whale_wall',0),4,7)} | SENT {epip(bd_r.get('sentiment',0),8,20)}\n"
                              f"MTF {epip(max(0,bd_r.get('mtf',0)),6,12)} | FLOW {epip(bd_r.get('orderflow',0),6,12)} | LISTING {epip(bd_r.get('listing',0),6,25)}")
                    # ALL reasons — not truncated
                    reasons_str="\n".join([f"▸ {rsn}" for rsn in r['reasons']])
                    sent_line=f"\n📊 {sentiment.get('source','Exchange')} L/S: {sentiment['top_long_pct']:.0f}% / {sentiment['top_short_pct']:.0f}% | Taker: {sentiment['taker_buy_pct']:.0f}%" if sentiment.get('available') else ""
                    mom_line="\n✅ MOMENTUM CONFIRMED" if r.get('momentum_confirmed') else ""
                    dual_line="\n🔥🔥 DUAL CONFIRMED — Scanner + Sentinel — HIGH CONVICTION 🔥🔥" if r.get('dual_confirmed') else ""
                    warn_line="\n⚠️ WARNINGS:\n"+"\n".join(r.get('warnings',[])) if r.get('warnings') else ""
                    _p=float(r.get('price') or 0); _t=float(r.get('tp') or 0); _s_=float(r.get('sl') or 0)
                    _tp1=float(r.get('tp1') or _t); _tp2=float(r.get('tp2') or _t); _tp3=float(r.get('tp3') or _t)
                    _elo=float(r.get('entry_lo') or _p); _ehi=float(r.get('entry_hi') or _p)
                    rr_ratio=f"{abs(_t-_p)/abs(_p-_s_):.1f}:1" if abs(_p-_s_)>0 else "N/A"
                    oi_ch=r.get('oi_change_6h',0); oi_line=f"\n📈 OI 6h: {oi_ch:+.1f}%" if abs(oi_ch)>=5 else ""
                    sig_line="📗 <b>LONG</b>" if r['type']=='LONG' else "📕 <b>SHORT</b>"
                    rsi_arrow={"RISING":"↑","FALLING":"↓","FLAT":"→"}.get(r.get('rsi_direction',''),"")
                    # Telegram
                    if eff_s.get('tg_token') and eff_s.get('tg_chat_id'):
                        msg_tg=(f"{dual_line}\n" if dual_line else ""
                                +f'{sig_line} | <b>Score: {r["pump_score"]}/100</b> | {pump_label(r["pump_score"],r["type"])}\n'
                                +f'<b>═══ {r["symbol"]} — {r["cls"].upper()} ({r.get("exchange","MEXC")}) ═══</b>\n'
                                +f'RSI: {r.get("rsi",0):.1f}{rsi_arrow} | R:R: {rr_ratio}\n'
                                +f'📍 Entry Zone: ${_elo:.6f}–${_ehi:.6f}\n'
                                +f'🎯 TP1: ${_tp1:.6f} | TP2: ${_tp2:.6f} | TP3: ${_tp3:.6f}\n'
                                +f'🛑 SL: ${_s_:.6f}\n'
                                +f'{oi_line}{sent_line}{mom_line}{warn_line}\n\n'
                                +f'📊 Signals:\n{pips_str}\n\n'
                                +f'📝 All Reasons:\n{reasons_str}')
                        send_tg(eff_s['tg_token'],eff_s['tg_chat_id'],msg_tg)
                    # Discord
                    if eff_s.get('discord_webhook'):
                        dc_color=0x059669 if r['type']=='LONG' else 0xdc2626
                        sig_hdr=("📗 **LONG**" if r['type']=='LONG' else "📕 **SHORT**")+f" | Score: **{r['pump_score']}/100**"
                        stats_text=(f"{dual_line}\n{sig_hdr}\n"
                                    f"**Exchange:** {r.get('exchange','MEXC')} | **RSI:** {r.get('rsi',0):.1f}{rsi_arrow} | **R:R:** {rr_ratio}\n"
                                    f"📍 **Entry Zone:** `${_elo:.6f}`–`${_ehi:.6f}`\n"
                                    f"🎯 **TP1:** `${_tp1:.6f}` | **TP2:** `${_tp2:.6f}` | **TP3:** `${_tp3:.6f}`\n"
                                    f"🛑 **SL:** `${_s_:.6f}`"
                                    +(f"\n**OI 6h:** {oi_ch:+.1f}%" if abs(oi_ch)>=5 else "")
                                    +(f"\n{sentiment.get('source','')} L/S: {sentiment['top_long_pct']:.0f}% / {sentiment['top_short_pct']:.0f}% | Taker: {sentiment['taker_buy_pct']:.0f}%" if sentiment.get('available') else "")
                                    +(f"\n⚠️ "+' | '.join(r.get('warnings',[])) if r.get('warnings') else ""))
                        send_discord(eff_s['discord_webhook'],{
                            'title':f'{r["symbol"]} ({r["cls"].upper()}) — {r.get("exchange","MEXC")}',
                            'color':dc_color,
                            'description':stats_text,
                            'fields':[
                                {'name':'📊 Signal Breakdown','value':pips_str,'inline':False},
                                {'name':f'📝 All Reasons ({len(r["reasons"])})','value':reasons_str[:1024],'inline':False},
                                *([{'name':'📝 Reasons (continued)','value':reasons_str[1024:2048],'inline':False}] if len(reasons_str)>1024 else [])
                            ],
                            'footer':{'text':f'APEX Intelligence Terminal • {datetime.now(timezone.utc).strftime("%H:%M:%S")} UTC'}
                        })
                    st.session_state.alerted_sigs.add(ak_recheck)
                    _ch=[c for c,v in [('Discord',eff_s.get('discord_webhook')),('Telegram',eff_s.get('tg_token') and eff_s.get('tg_chat_id'))] if v]
                    _dual_note=' 🔥DUAL' if r.get('dual_confirmed') else ''
                    st.toast(f'📨{_dual_note} {r["symbol"]} → {"+".join(_ch) if _ch else "no channels configured"}',icon='📨' if _ch else '⚠️')
    except Exception as e:
        if any(x in str(e) for x in ["510","429","too frequent"]): st.error("🚦 Rate limited — wait 60s")
        else: st.error(f"Error: {e}")


# ─── SENTINEL RUNNER ─────────────────────────────────────────────────────────
if st.session_state.get('sentinel_active') and do_scan and st.session_state.scan_count>0:
    sent_ph=st.empty()
    sent_ph.markdown('<div style="background:#0f1117;border-radius:8px;padding:10px 18px;margin-bottom:10px;border:1px solid #7c3aed44;"><span style="color:#7c3aed;font-family:monospace;font-size:.65rem;font-weight:700;">🛰️ SENTINEL — TOP 100 SCANNING…</span></div>',unsafe_allow_html=True)
    try:
        try:
            import nest_asyncio as _na
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            try:
                _loop=asyncio.get_event_loop()
                if _loop.is_closed(): _loop=asyncio.new_event_loop(); asyncio.set_event_loop(_loop)
            except RuntimeError:
                _loop=asyncio.new_event_loop(); asyncio.set_event_loop(_loop)
            _na.apply(_loop)
        except: pass

        s_screener=PrePumpScreener(
            cmc_key=eff_s.get('cmc_key',''),okx_key=eff_s.get('okx_key',''),
            okx_secret=eff_s.get('okx_secret',''),okx_passphrase=eff_s.get('okx_passphrase',''),
            gate_key=eff_s.get('gate_key',''),gate_secret=eff_s.get('gate_secret',''))

        async def sentinel_tick(screener, s):
            btc_trend,btc_px,_=await screener.fetch_btc()
            await asyncio.gather(screener.okx.load_markets(),screener.mexc.load_markets(),screener.gate.load_markets(),return_exceptions=True)
            tk_okx,tk_mexc,tk_gate={},{},{}
            res_tk=await asyncio.gather(screener.okx.fetch_tickers(),screener.mexc.fetch_tickers(),screener.gate.fetch_tickers(),return_exceptions=True)
            if not isinstance(res_tk[0],Exception): tk_okx=res_tk[0]
            if not isinstance(res_tk[1],Exception): tk_mexc=res_tk[1]
            if not isinstance(res_tk[2],Exception): tk_gate=res_tk[2]
            all_swaps={}
            for sym,t in {**tk_mexc,**tk_gate,**tk_okx}.items():
                if sym.endswith(':USDT') and t.get('quoteVolume'):
                    exn='OKX' if sym in tk_okx else ('GATE' if sym in tk_gate else 'MEXC')
                    exo=screener.okx if exn=='OKX' else (screener.gate if exn=='GATE' else screener.mexc)
                    all_swaps[sym]={'vol':float(t['quoteVolume'] or 0),'exch_name':exn,'exch_obj':exo}
            sorted_swaps=sorted(all_swaps.items(),key=lambda x:x[1]['vol'],reverse=True)[:100]
            uni_size=len(sorted_swaps)
            checked=st.session_state.get('sentinel_total_checked',0)
            batch_sz=max(1,s.get('sentinel_batch_size',5))
            start=checked%max(1,uni_size); end=min(start+batch_sz,uni_size)
            coin_batch=sorted_swaps[start:end]
            new_sigs=[]
            for sym,data in coin_batch:
                try:
                    r=await screener.analyze(data['exch_name'],data['exch_obj'],sym,s,btc_trend)
                    if r and r['pump_score']>=s.get('sentinel_score_threshold',70): new_sigs.append(r)
                except: pass
            for exch in [screener.okx,screener.mexc,screener.gate]:
                try: await exch.close()
                except: pass
            return new_sigs,btc_px,uni_size

        new_sigs,s_btc,total_uni=asyncio.run(sentinel_tick(s_screener,eff_s))
        st.session_state.sentinel_total_checked=st.session_state.get('sentinel_total_checked',0)+eff_s.get('sentinel_batch_size',5)
        st.session_state.sentinel_last_check=datetime.now().strftime('%H:%M:%S')
        st.session_state.sentinel_universe_size=f"TOP {total_uni}"
        st.session_state.btc_price=s_btc

        if new_sigs:
            st.session_state.sentinel_signals_found=st.session_state.get('sentinel_signals_found',0)+len(new_sigs)
            if 'alerted_sigs' not in st.session_state: st.session_state.alerted_sigs=set()
            for r in new_sigs:
                st.toast(f"🛰️ {r['symbol']} — {r['pump_score']}/100 {r['type']}",icon="🚨")
                ak=f"{r['symbol']}_{r['type']}_{datetime.now().strftime('%Y%m%d%H')}_sent"
                if 'sentinel_results' not in st.session_state: st.session_state.sentinel_results=[]
                sent_syms={x['symbol'] for x in st.session_state.sentinel_results}
                if r['symbol'] not in sent_syms:
                    st.session_state.sentinel_results.insert(0,r)
                    # FIX: Sentinel signals go to journal too
                    log_trade(r)
                if ak not in st.session_state.alerted_sigs:
                    _p=float(r.get('price') or 0); _t=float(r.get('tp') or 0); _s_=float(r.get('sl') or 0)
                    rr_r=f"{abs(_t-_p)/abs(_p-_s_):.1f}:1" if abs(_p-_s_)>0 else "N/A"
                    # All reasons for sentinel alert too
                    rsns="\n".join([f"• {x}" for x in r['reasons']])
                    sig_line_sent="📗 LONG" if r['type']=='LONG' else "📕 SHORT"
                    _tp1=float(r.get('tp1') or _t); _tp2=float(r.get('tp2') or _t); _tp3=float(r.get('tp3') or _t)
                    _elo=float(r.get('entry_lo') or _p); _ehi=float(r.get('entry_hi') or _p)
                    if eff_s.get('tg_token') and eff_s.get('tg_chat_id'):
                        send_tg(eff_s['tg_token'],eff_s['tg_chat_id'],
                            f"<b>🛰️ [SENTINEL] {sig_line_sent} | Score: {r['pump_score']}/100</b>\n"
                            f"<b>{r['symbol']} — {r.get('cls','').upper()} | {r.get('exchange','')}</b>\n"
                            f"R:R {rr_r}\n"
                            f"📍 Entry Zone: ${_elo:.6f}–${_ehi:.6f}\n"
                            f"TP1: ${_tp1:.6f} | TP2: ${_tp2:.6f} | TP3: ${_tp3:.6f}\n"
                            f"🛑 SL: ${_s_:.6f}\n\n"
                            f"📝 All Reasons:\n{rsns}")
                    if eff_s.get('discord_webhook'):
                        dc_color_s=0x059669 if r['type']=='LONG' else 0xdc2626
                        send_discord(eff_s['discord_webhook'],{
                            "title":f"🛰️ SENTINEL: {r['symbol']} ({r['pump_score']}/100)",
                            "color":dc_color_s,
                            "description":(f"{'📗 **LONG**' if r['type']=='LONG' else '📕 **SHORT**'} | Score: **{r['pump_score']}/100**\n"
                                           f"**Exchange:** {r.get('exchange','')} | **Class:** {r.get('cls','').upper()} | **R:R:** {rr_r}\n"
                                           f"📍 Entry Zone: `${_elo:.6f}`–`${_ehi:.6f}`\n"
                                           f"TP1: `${_tp1:.6f}` | TP2: `${_tp2:.6f}` | TP3: `${_tp3:.6f}`\n"
                                           f"🛑 SL: `${_s_:.6f}`"),
                            "fields":[{'name':f'📝 All Reasons ({len(r["reasons"])})','value':"\n".join([f"▸ {x}" for x in r['reasons']])[:1024],'inline':False}],
                            "footer":{"text":f"APEX Sentinel — Top 100 | {datetime.now(timezone.utc).strftime('%H:%M UTC')}"}})
                    st.session_state.alerted_sigs.add(ak)

        chk_f=st.session_state.get('sentinel_total_checked',0)
        sig_f=st.session_state.get('sentinel_signals_found',0)
        pct_done=int((chk_f%max(1,total_uni))/max(1,total_uni)*100)
        sent_ph.markdown(
            f'<div style="background:#0f1117;border-radius:8px;padding:10px 18px;margin-bottom:10px;border:1px solid #7c3aed66;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;">'
            f'<span style="color:#7c3aed;font-family:monospace;font-size:.65rem;font-weight:700;">🛰️ SENTINEL LIVE — TOP {total_uni}</span>'
            f'<span style="font-family:monospace;font-size:.6rem;color:#9ca3af;">{chk_f} checked ({pct_done}% cycle)</span>'
            f'<span style="font-family:monospace;font-size:.62rem;color:#ff4444;font-weight:700;">{sig_f} signals found</span>'
            f'<span style="font-family:monospace;font-size:.58rem;color:#9ca3af;">Last: {st.session_state.get("sentinel_last_check","?")}</span>'
            f'</div><div style="height:3px;background:#1f2937;border-radius:2px;margin-top:6px;">'
            f'<div style="width:{pct_done}%;height:100%;background:#7c3aed;border-radius:2px;"></div></div></div>',
            unsafe_allow_html=True)
    except Exception as e:
        import traceback
        st.session_state.sentinel_active=False
        st.error(f"Sentinel error (auto-disabled): {e}")
        st.code(traceback.format_exc(),language="text")


# ─── RESULTS DISPLAY ─────────────────────────────────────────────────────────
results=st.session_state.results
errs=getattr(st.session_state,'scan_errors',[])
btc_t=getattr(st.session_state,'btc_trend',"NEUTRAL")

if not results:
    st.markdown('<div class="empty-st"><div style="font-size:2.5rem;opacity:.2;margin-bottom:12px;">🔥</div>Run a scan — or lower Min Score in sidebar</div>',unsafe_allow_html=True)
    if st.session_state.scan_count>0:
        with st.expander("🔍 Debug Info",expanded=True):
            raw=st.session_state.last_raw_count
            st.markdown(f"""**Scan #{st.session_state.scan_count}** at **{st.session_state.last_scan}**
| Check | Status |
|---|---|
| Raw coins found | **{raw}** |
| Min score filter | **{eff_s['min_score']}** — set to 1 to see everything |
| Min R:R | **{eff_s.get('min_rr',1.5)}** |
| Momentum required | **{'YES — turn off in sidebar' if eff_s.get('require_momentum') else 'NO'}** |
| BTC filter | **{btc_t}** — {"⚠️ BEARISH blocking LONGs!" if btc_t=='BEARISH' else "✅ OK"} |
| Errors | **{len(errs)}** out of {eff_s['scan_depth']} coins |""")
            if errs: st.code("\n".join(errs[:15]),language="text")
else:
    sq=[r for r in results if r['cls']=="squeeze"]
    br=[r for r in results if r['cls']=="breakout"]
    wh=[r for r in results if r['cls']=="whale_driven"]
    ea=[r for r in results if r['cls']=="early"]
    top=results[0]
    mom_count=sum(1 for r in results if r.get('momentum_confirmed'))
    dual_count=sum(1 for r in results if r.get('dual_confirmed'))
    penalty_count=sum(1 for r in results if r.get('warnings'))

    st.markdown(f"""<div class="stat-strip">
      <div><div class="ss-val" style="color:var(--red);">{len(sq)}</div><div class="ss-lbl">Squeeze</div></div>
      <div><div class="ss-val" style="color:var(--amber);">{len(br)}</div><div class="ss-lbl">Breakout</div></div>
      <div><div class="ss-val" style="color:var(--purple);">{len(wh)}</div><div class="ss-lbl">Whale Driven</div></div>
      <div><div class="ss-val" style="color:var(--blue);">{len(ea)}</div><div class="ss-lbl">Early</div></div>
      <div><div class="ss-val">{len(results)}</div><div class="ss-lbl">Total</div></div>
      <div><div class="ss-val">{st.session_state.last_raw_count}</div><div class="ss-lbl">Raw Scanned</div></div>
      <div><div class="ss-val" style="color:var(--green);">{mom_count}</div><div class="ss-lbl">Momentum Live</div></div>
      <div><div class="ss-val" style="color:#ff0000;font-weight:900;">{dual_count}</div><div class="ss-lbl">🔥 Dual Confirmed</div></div>
      <div><div class="ss-val" style="color:var(--red);">{penalty_count}</div><div class="ss-lbl">Penalised</div></div>
      <div><div class="ss-val" style="color:{pump_color(top['pump_score'])};">{top['symbol']}</div><div class="ss-lbl">Hottest ({top['pump_score']})</div></div>
    </div>""",unsafe_allow_html=True)

    # ── SENTINEL RESULTS (full cards) ─────────────────────────────────────
    sent_res=st.session_state.get('sentinel_results',[])
    if sent_res:
        st.markdown('<div style="background:linear-gradient(135deg,#1a0a2e,#0f1117);border:1px solid #7c3aed66;border-radius:10px;padding:10px 16px;margin-bottom:12px;"><span style="font-family:monospace;font-size:.65rem;font-weight:700;color:#a78bfa;">🛰️ SENTINEL LIVE — Full Signal Cards</span></div>',unsafe_allow_html=True)
        for _sr in sent_res[:10]:
            # Check if this sentinel signal is also in scanner results (dual confirm)
            scanner_syms={r['symbol'] for r in results}
            is_dual=_sr['symbol'] in scanner_syms
            render_card(_sr,_sr.get('pump_score',0)>=90,dual_confirmed=is_dual)
        if st.button("🗑️ Clear Sentinel Results",key="clr_sent"):
            st.session_state.sentinel_results=[]; st.rerun()
        st.markdown("---")

    tab_s,tab_b,tab_w,tab_e=st.tabs([
        f"🔴  SQUEEZE  ({len(sq)})",
        f"🟡  BREAKOUT  ({len(br)})",
        f"🐋  WHALE DRIVEN  ({len(wh)})",
        f"🔵  EARLY  ({len(ea)})"
    ])
    with tab_s:
        st.markdown('<div class="tab-desc"><b>About to Squeeze</b> — Funding extreme + OB heavily imbalanced. Trapped shorts/longs about to be liquidated. ⚠️ Cards with ⚠️ badges = late entry risk.</div>',unsafe_allow_html=True)
        [render_card(r,r.get('pump_score',0)>=90,r.get('dual_confirmed',False)) for r in sq] if sq else st.markdown('<div class="empty-st">⚡ No squeeze setups</div>',unsafe_allow_html=True)
    with tab_b:
        st.markdown('<div class="tab-desc"><b>Confirmed Breakout</b> — OI spiking with volume. New money entering. 🔷 FVG zones now used for SL placement.</div>',unsafe_allow_html=True)
        [render_card(r,r.get('pump_score',0)>=90,r.get('dual_confirmed',False)) for r in br] if br else st.markdown('<div class="empty-st">📈 No confirmed breakouts</div>',unsafe_allow_html=True)
    with tab_w:
        st.markdown('<div class="tab-desc"><b>Whale Driven</b> — Large wall + volume surge. Follow the whale.</div>',unsafe_allow_html=True)
        [render_card(r,r.get('pump_score',0)>=90,r.get('dual_confirmed',False)) for r in wh] if wh else st.markdown('<div class="empty-st">🐋 No whale setups</div>',unsafe_allow_html=True)
    with tab_e:
        st.markdown('<div class="tab-desc"><b>Early Signal</b> — Pre-pump alignment. Watchlist — often move into Squeeze/Breakout within 1–4h.</div>',unsafe_allow_html=True)
        [render_card(r,r.get('pump_score',0)>=90,r.get('dual_confirmed',False)) for r in ea] if ea else st.markdown('<div class="empty-st">📡 No early signals</div>',unsafe_allow_html=True)

if eff_s.get('auto_scan'):
    st.markdown(f'<div style="text-align:center;font-family:monospace;font-size:.62rem;color:#9ca3af;padding:16px 0;">🤖 Auto-Pilot Active — Next scan in {eff_s["auto_interval"]} minute(s)</div>',unsafe_allow_html=True)
    time.sleep(eff_s['auto_interval']*60)
    st.rerun()
