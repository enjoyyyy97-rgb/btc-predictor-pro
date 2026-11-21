import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="BTC/USDT PREDICTOR PRO",
    page_icon="₿",
    layout="wide",
)

MIN_SPREAD_PCT = 0.015
FLAT_SPREAD_PCT = 0.005
MIN_VOL_RATIO = 1.05


def to_float(val, default=0.0):
    if val is None or pd.isna(val):
        return default
    num = float(val)
    return default if (np.isnan(num) or np.isinf(num)) else num


@st.cache_data(ttl=3, show_spinner=False)
def fetch_klines(symbol="BTCUSDT", interval="5m", limit=350):
    endpoints = [
        "https://data-api.binance.vision/api/v3/klines",
        "https://api1.binance.com/api/v3/klines",
        "https://api2.binance.com/api/v3/klines",
        "https://api3.binance.com/api/v3/klines",
        "https://api4.binance.com/api/v3/klines",
        "https://api.binance.com/api/v3/klines",
    ]
    headers = {"User-Agent": "Mozilla/5.0"}
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    last_error = "sin respuesta"
    for url in endpoints:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=8)
            if resp.status_code == 451:
                last_error = f"HTTP 451 en {url}"
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 0:
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ]
                df = pd.DataFrame(data, columns=cols)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["close_time_ms"] = pd.to_numeric(df["close_time"], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close", "volume", "close_time_ms"])
                df = df[df["close"] > 0].reset_index(drop=True)
                if len(df) >= 35:
                    return df
        except Exception as exc:
            last_error = f"{url}: {exc}"

    raise RuntimeError(f"No se pudieron obtener velas de Binance ({last_error})")


def add_indicators(df):
    d = df.copy()
    d["EMA9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["close"].ewm(span=21, adjust=False).mean()

    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss > 0.0, 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss == 0.0)), 50.0)
    d["RSI"] = rsi

    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    d["MACD_hist"] = macd - macd_sig

    d["VOL_AVG"] = d["volume"].rolling(20, min_periods=1).mean()
    return d


def score_candle(candle, prev):
    close_p = to_float(candle["close"])
    ema9 = to_float(candle["EMA9"], close_p)
    ema21 = to_float(candle["EMA21"], close_p)
    vol = to_float(candle["volume"])
    vol_avg = to_float(candle["VOL_AVG"])
    rsi = to_float(candle["RSI"], 50.0)
    hist = to_float(candle["MACD_hist"])
    prev_hist = to_float(prev["MACD_hist"])

    vol_ratio = (vol / vol_avg) if vol_avg > 0 else 0.0
    spread_pct = (abs(ema9 - ema21) / close_p * 100.0) if close_p > 0 else 0.0

    is_flat = spread_pct < FLAT_SPREAD_PCT
    spread_ok = spread_pct >= MIN_SPREAD_PCT

    bull = [
        close_p > ema9 > ema21,
        spread_ok,
        54.0 <= rsi <= 78.0,
        hist > 0.0 and hist >= prev_hist * 0.85,
        vol_ratio >= MIN_VOL_RATIO,
    ]
    bear = [
        close_p < ema9 < ema21,
        spread_ok,
        22.0 <= rsi <= 46.0,
        hist < 0.0 and hist <= prev_hist * 0.85,
        vol_ratio >= MIN_VOL_RATIO,
    ]

    up_score = sum(int(x) for x in bull)
    down_score = sum(int(x) for x in bear)
    up_pct = int(round((up_score / 5.0) * 100))
    down_pct = int(round((down_score / 5.0) * 100))

    if up_score >= 4 and up_score > down_score:
        signal = "UP"
        reason = f"Señal alcista sólida ({up_score}/5 filtros · {up_pct}%) | Vol {vol_ratio:.2f}x | Spread {spread_pct:.3f}%."
    elif down_score >= 4 and down_score > up_score:
        signal = "DOWN"
        reason = f"Señal bajista sólida ({down_score}/5 filtros · {down_pct}%) | Vol {vol_ratio:.2f}x | Spread {spread_pct:.3f}%."
    elif not is_flat and up_score == 3 and down_score <= 1:
        signal = "IMPULSO ALCISTA (3/5)"
        reason = f"Presión alcista moderada ({up_score}/5 filtros · {up_pct}%) | Vol {vol_ratio:.2f}x | Spread {spread_pct:.3f}%."
    elif not is_flat and down_score == 3 and up_score <= 1:
        signal = "IMPULSO BAJISTA (3/5)"
        reason = f"Presión bajista moderada ({down_score}/5 filtros · {down_pct}%) | Vol {vol_ratio:.2f}x | Spread {spread_pct:.3f}%."
    else:
        signal = "LATERAL / NO OPERAR"
        if is_flat:
            reason = f"Mercado plano: EMA 9 y EMA 21 muy cercanas ({spread_pct:.3f}% < {FLAT_SPREAD_PCT:.3f}%)."
        else:
            reason = f"Sin ventaja clara ({up_score}/5 alcista vs {down_score}/5 bajista) | Vol {vol_ratio:.2f}x."

    return {
        "signal": signal,
        "reason": reason,
        "up_score": up_score,
        "down_score": down_score,
        "up_pct": up_pct,
        "down_pct": down_pct,
        "vol_ratio": vol_ratio,
        "rsi": rsi,
        "close": close_p,
    }


def radar_actual(curr, prev):
    c_close = to_float(curr["close"])
    c_ema9 = to_float(curr["EMA9"], c_close)
    p_ema9 = to_float(prev["EMA9"], c_ema9)
    c_rsi = to_float(curr["RSI"], 50.0)
    c_hist = to_float(curr["MACD_hist"])
    p_hist = to_float(prev["MACD_hist"])
    c_vol = to_float(curr["volume"])
    c_vavg = to_float(curr["VOL_AVG"])
    v_ratio = (c_vol / c_vavg) if c_vavg > 0 else 0.0

    bull = [
        c_close > c_ema9 and c_ema9 >= p_ema9,
        c_rsi >= 53.0,
        c_hist > p_hist,
        v_ratio >= 1.0,
    ]
    bear = [
        c_close < c_ema9 and c_ema9 <= p_ema9,
        c_rsi <= 47.0,
        c_hist < p_hist,
        v_ratio >= 1.0,
    ]

    up_s = sum(int(x) for x in bull)
    down_s = sum(int(x) for x in bear)
    return up_s, down_s, int(round(up_s / 4.0 * 100)), int(round(down_s / 4.0 * 100))


@st.cache_data(ttl=30, show_spinner=False)
def run_backtest(data, lookback=250):
    if data is None or len(data) < 40:
        return {"total": 0, "wins": 0, "losses": 0, "winrate": 0.0, "rows": []}

    start = max(25, len(data) - lookback - 2)
    end = len(data) - 2

    wins = 0
    losses = 0
    rows = []

    for i in range(start, end):
        info = score_candle(data.iloc[i], data.iloc[i - 1])
        sig = info["signal"]
        direction = None
        if sig in ("UP", "IMPULSO ALCISTA (3/5)"):
            direction = "UP"
        elif sig in ("DOWN", "IMPULSO BAJISTA (3/5)"):
            direction = "DOWN"

        if direction:
            nxt = data.iloc[i + 1]
            n_open = to_float(nxt["open"])
            n_close = to_float(nxt["close"])
            won = (direction == "UP" and n_close > n_open) or (direction == "DOWN" and n_close < n_open)
            if won:
                wins += 1
            else:
                losses += 1

            rows.append({
                "Señal": sig,
                "Resultado": "ACIERTO" if won else "FALLO",
                "Volumen": f"{info['vol_ratio']:.2f}x",
                "RSI": f"{info['rsi']:.1f}",
                "Cierre Vela": f"${info['close']:,.2f}",
            })

    total = wins + losses
    winrate = (wins / total * 100.0) if total > 0 else 0.0
    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "rows": list(reversed(rows[-10:])),
    }


st.title("₿ BTC/USDT PREDICTOR PRO")
st.caption("ANÁLISIS TÉCNICO EN VIVO · TEMPORALIDAD 5 MINUTOS")


@st.fragment(run_every="3s")
def dashboard():
    try:
        raw_df = fetch_klines("BTCUSDT", "5m", 350)
        data = add_indicators(raw_df)
    except Exception as exc:
        st.warning(f"Reconectando con Binance... ({exc})")
        return

    closed_candle = data.iloc[-2]
    prev_closed = data.iloc[-3]
    live_candle = data.iloc[-1]

    info = score_candle(closed_candle, prev_closed)
    early_up, early_down, early_up_pct, early_down_pct = radar_actual(live_candle, closed_candle)

    now_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    close_ms = int(to_float(live_candle["close_time_ms"], now_ms + 300_000))
    segs_left = max(1, min(300, int((close_ms - now_ms) // 1000)))
    countdown = f"{segs_left // 60:02d}:{segs_left % 60:02d}"

    st.subheader("PRECIO BTC / USDT")
    p1, p2 = st.columns(2)
    p1.metric("PRECIO EN VIVO", f"${to_float(live_candle['close']):,.2f} USDT")
    p2.metric("CIERRE DE VELA 5M", countdown)

    st.divider()
    st.subheader("DECISIÓN DEL SISTEMA · 5M")

    sig = info["signal"]
    if sig == "UP":
        st.success(f"🟢 {sig}")
    elif sig == "DOWN":
        st.error(f"🔴 {sig}")
    elif "ALCISTA" in sig:
        st.info(f"🔵 {sig}")
    elif "BAJISTA" in sig:
        st.info(f"🟠 {sig}")
    else:
        st.warning(f"🟡 {sig}")

    st.caption(info["reason"])

    c1, c2 = st.columns(2)
    c1.metric("PRESIÓN ALCISTA (CONFIRMADA)", f"{info['up_pct']}%", f"{info['up_score']}/5 filtros")
    c2.metric("PRESIÓN BAJISTA (CONFIRMADA)", f"{info['down_pct']}%", f"{info['down_score']}/5 filtros")

    st.markdown("### RADAR ANTICIPADO · 2 MINUTOS")
    if segs_left <= 120:
        if early_up >= 3 and early_up > early_down:
            st.success(f"🟢 PREALERTA SUBIDA ({early_up_pct}% · {early_up}/4)")
        elif early_down >= 3 and early_down > early_up:
            st.error(f"🔴 PREALERTA BAJADA ({early_down_pct}% · {early_down}/4)")
        elif early_up == 2:
            st.info(f"👀 VIGILANCIA ALCISTA ({early_up_pct}% · {early_up}/4)")
        elif early_down == 2:
            st.info(f"👀 VIGILANCIA BAJISTA ({early_down_pct}% · {early_down}/4)")
        else:
            st.warning("SIN PREALERTA: ESPERANDO CONFIRMACIÓN")
    else:
        st.info("⏳ El radar anticipado se activa en los últimos 2 minutos del cierre de vela de 5m.")

    st.markdown("### HISTORIAL DE SEÑALES · ÚLTIMAS 250 VELAS (5M)")
    bt = run_backtest(data, lookback=250)

    h1, h2, h3 = st.columns(3)
    h1.metric("% ACIERTO HISTÓRICO", f"{bt['winrate']:.1f}%")
    h2.metric("ACIERTOS / FALLOS", f"{bt['wins']} / {bt['losses']}")
    h3.metric("SEÑALES EMITIDAS", f"{bt['total']}")

    if bt["rows"]:
        st.dataframe(bt["rows"], width="stretch", hide_index=True)
    else:
        st.caption("Sin señales registradas en el tramo evaluado.")


dashboard()