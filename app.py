import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="BTC/USDT PANEL PRO",
    page_icon="₿",
    layout="wide",
)


def to_float(val, default=0.0):
    if val is None or pd.isna(val):
        return default
    num = float(val)
    return default if (np.isnan(num) or np.isinf(num)) else num


@st.cache_data(ttl=2, show_spinner=False)
def fetch_klines(symbol="BTCUSDT", interval="5m", limit=250):
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
            payload = resp.json()
            if isinstance(payload, list) and len(payload) > 0:
                cols = [
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore",
                ]
                df = pd.DataFrame(payload, columns=cols)
                for c in ["open", "high", "low", "close", "volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                df["close_time_ms"] = pd.to_numeric(df["close_time"], errors="coerce")
                df = df.dropna(subset=["open", "high", "low", "close", "volume", "close_time_ms"])
                df = df[df["close"] > 0].reset_index(drop=True)
                if len(df) >= 30:
                    return df
        except Exception as exc:
            last_error = f"{url}: {exc}"

    raise RuntimeError(f"Sin conexión con Binance ({last_error})")


def analyze_intracandle(df):
    d = df.copy()
    d["EMA9"] = d["close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["close"].ewm(span=21, adjust=False).mean()
    d["VOL_AVG"] = d["volume"].rolling(20, min_periods=1).mean()

    prev_closed = d.iloc[-2]
    live = d.iloc[-1]

    open_p = to_float(live["open"])
    high_p = to_float(live["high"], open_p)
    low_p = to_float(live["low"], open_p)
    close_p = to_float(live["close"], open_p)
    ema9 = to_float(live["EMA9"], close_p)
    ema21 = to_float(live["EMA21"], close_p)
    vol = to_float(live["volume"])
    vol_avg = to_float(live["VOL_AVG"], 1.0)

    diff_usd = close_p - open_p
    diff_pct = (diff_usd / open_p * 100.0) if open_p > 0 else 0.0
    candle_range = max(high_p - low_p, 1e-9)

    # Presión de cierre según posición dentro del rango high-low y cambio vs apertura
    pos_in_range = (close_p - low_p) / candle_range  # 0.0 (mínimo) a 1.0 (máximo)
    bull_pressure = int(np.clip(round(pos_in_range * 100.0), 0, 100))
    bear_pressure = 100 - bull_pressure

    # Retroceso intravela: cuánto ha retrocedido desde el máximo o mínimo de la vela
    pullback_up_pct = int(np.clip(round(((high_p - close_p) / candle_range) * 100.0), 0, 100))
    pullback_down_pct = int(np.clip(round(((close_p - low_p) / candle_range) * 100.0), 0, 100))

    move_pct = abs(diff_pct)
    vol_ratio = (vol / vol_avg) if vol_avg > 0 else 0.0
    spread_pct = (abs(ema9 - ema21) / close_p * 100.0) if close_p > 0 else 0.0

    # Decisión del sistema evitando atascos en lateral pero sin inventar señales sin desplazamiento
    if move_pct < 0.025 and spread_pct < 0.015:
        decision = "LATERAL · NO OPERAR"
        status_type = "warning"
        detail = (
            "Sin una dirección técnica y desplazamiento verificable en la vela actual; "
            "esperando mayor movimiento desde apertura."
        )
        bias = "NEUTRO"
    elif close_p > open_p and bull_pressure >= 65 and move_pct >= 0.05:
        decision = "UP · SESGO ALCISTA"
        status_type = "success"
        detail = f"Impulso alcista activo (+{diff_pct:.3f}%) con presión compradora del {bull_pressure}%."
        bias = "ALCISTA"
    elif close_p < open_p and bear_pressure >= 65 and move_pct >= 0.05:
        decision = "DOWN · SESGO BAJISTA"
        status_type = "error"
        detail = f"Impulso bajista activo ({diff_pct:.3f}%) con presión vendedora del {bear_pressure}%."
        bias = "BAJISTA"
    elif close_p < open_p and bear_pressure >= 55:
        decision = "POSIBLE SESGO BAJISTA"
        status_type = "info"
        detail = f"Presión bajista moderada ({bear_pressure}%) | Variación {diff_pct:+.3f}% | Vol {vol_ratio:.2f}x."
        bias = "POSIBLE SESGO BAJISTA"
    elif close_p > open_p and bull_pressure >= 55:
        decision = "POSIBLE SESGO ALCISTA"
        status_type = "info"
        detail = f"Presión alcista moderada ({bull_pressure}%) | Variación {diff_pct:+.3f}% | Vol {vol_ratio:.2f}x."
        bias = "POSIBLE SESGO ALCISTA"
    else:
        decision = "LATERAL · NO OPERAR"
        status_type = "warning"
        detail = "Lectura mixta o desplazamiento insuficiente para confirmar entrada."
        bias = "NEUTRO"

    # Preaviso temprano (barrido y recuperación de mecha/rango)
    swept_low = low_p < to_float(prev_closed["low"], low_p) and close_p > open_p
    swept_high = high_p > to_float(prev_closed["high"], high_p) and close_p < open_p

    if swept_low and bull_pressure >= 60:
        prealert = "🟢 PREALERTA ALCISTA: Barrido de mínimos con recuperación intravela."
    elif swept_high and bear_pressure >= 60:
        prealert = "🔴 PREALERTA BAJISTA: Barrido de máximos con rechazo intravela."
    else:
        prealert = "Sin prealerta temprana activa — aún no hay barrido y recuperación intravela con criterios suficientes."

    now_utc = pd.Timestamp.utcnow()
    close_ms = int(to_float(live["close_time_ms"], int(now_utc.timestamp() * 1000) + 300_000))
    now_ms = int(now_utc.timestamp() * 1000)
    secs_left = max(1, min(300, int((close_ms - now_ms) // 1000)))
    countdown = f"{secs_left // 60:02d}:{secs_left % 60:02d}"
    updated_str = now_utc.tz_convert("America/Caracas").strftime("%I:%M:%S %p")

    return {
        "live_price": close_p,
        "open_price": open_p,
        "diff_usd": diff_usd,
        "diff_pct": diff_pct,
        "countdown": countdown,
        "updated_str": updated_str,
        "decision": decision,
        "status_type": status_type,
        "detail": detail,
        "bull_pressure": bull_pressure,
        "bear_pressure": bear_pressure,
        "pullback_up_pct": pullback_up_pct,
        "pullback_down_pct": pullback_down_pct,
        "prealert": prealert,
        "bias": bias,
        "vol_ratio": vol_ratio,
        "ema9": ema9,
        "ema21": ema21,
    }


def render_timeframe_view(tf_label, interval):
    try:
        df = fetch_klines("BTCUSDT", interval=interval, limit=120)
        info = analyze_intracandle(df)
    except Exception as exc:
        st.warning(f"Reconectando ({tf_label})... {exc}")
        return

    sign = "+" if info["diff_usd"] >= 0 else ""
    st.metric(
        f"BTC / USDT ({tf_label})",
        f"${info['live_price']:,.2f}",
        f"{sign}${info['diff_usd']:,.2f} ({sign}{info['diff_pct']:.3f}%)",
    )
    st.write(f"**Estado:** {info['decision']}  |  **Volumen relativo:** {info['vol_ratio']:.2f}x")
    st.caption(info["detail"])


st.title("₿ BTC / USDT · PANEL PRO")
st.caption("LECTURA INTRAVELA EN TIEMPO REAL · SINCRONIZADO CON BINANCE")

tab_pro, tab_15m, tab_1h, tab_radar = st.tabs(["Panel Pro (5M)", "15M", "1H", "Radar Max"])


@st.fragment(run_every="2s")
def live_panel_pro():
    try:
        df_5m = fetch_klines("BTCUSDT", interval="5m", limit=250)
        info = analyze_intracandle(df_5m)
    except Exception as exc:
        st.warning(f"Reconectando con Binance... ({exc})")
        return

    sign = "+" if info["diff_usd"] >= 0 else ""
    is_new_candle = info["countdown"] >= "04:40"

    st.markdown("### PRECIO BTC / USDT")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("PRECIO EN VIVO", f"${info['live_price']:,.2f} USDT")
        st.caption(f"Apertura 5M: **${info['open_price']:,.2f}** · {sign}${info['diff_usd']:,.2f} ({sign}{info['diff_pct']:.3f}%)")
    with p2:
        st.metric("CIERRE VELA 5M", info["countdown"])
        st.caption(f"{'Vela nueva · ' if is_new_candle else ''}Actualizado {info['updated_str']}")

    st.divider()
    st.markdown("### DECISIÓN DEL SISTEMA · 5M")

    if info["status_type"] == "success":
        st.success(f"🟢 {info['decision']}")
    elif info["status_type"] == "error":
        st.error(f"🔴 {info['decision']}")
    elif info["status_type"] == "info":
        st.info(f"🔵 {info['decision']}")
    else:
        st.warning(f"🟡 {info['decision']}")

    st.caption(info["detail"])

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**PRESIÓN DE CIERRE**")
        st.caption("PRESIÓN DE CIERRE ALCISTA / BAJISTA")
        st.write(f"Alcista: **{info['bull_pressure']}%**")
        st.progress(info["bull_pressure"] / 100.0)
        st.write(f"Bajista: **{info['bear_pressure']}%**")
        st.progress(info["bear_pressure"] / 100.0)

    with col_right:
        st.markdown("**RETROCESO INTRAVELA**")
        st.caption("DESPLAZAMIENTO RESPECTO A EXTREMOS DE VELA")
        st.write(f"Al alza (margen hasta máximo): **{info['pullback_up_pct']}%**")
        st.progress(info["pullback_up_pct"] / 100.0)
        st.write(f"A la baja (margen desde mínimo): **{info['pullback_down_pct']}%**")
        st.progress(info["pullback_down_pct"] / 100.0)

    st.markdown("### PREAVISO 5M · ANTES DE CONFIRMAR")
    st.info(info["prealert"])

    st.markdown("### SESGO DIRECCIONAL")
    st.subheader(f"POSIBLE SESGO: {info['bias']}")


with tab_pro:
    live_panel_pro()

with tab_15m:
    render_timeframe_view("15M", "15m")

with tab_1h:
    render_timeframe_view("1H", "1h")

with tab_radar:
    st.subheader("RADAR MAX · RESUMEN MULTITEMPORAL")
    c_a, c_b, c_c = st.columns(3)
    with c_a:
        render_timeframe_view("5M", "5m")
    with c_b:
        render_timeframe_view("15M", "15m")
    with c_c:
        render_timeframe_view("1H", "1h")