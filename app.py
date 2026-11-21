import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="BTC/USDT PREDICTOR PRO",
    page_icon="₿",
    layout="wide",
)


@st.cache_data(ttl=3)
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

    last_error = None
    raw = None

    for url in endpoints:
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            if response.status_code == 451:
                last_error = f"451 en {url}"
                continue
            response.raise_for_status()
            raw = response.json()
            if isinstance(raw, list) and len(raw) > 0:
                break
        except Exception as exc:
            last_error = f"{url}: {exc}"
            continue

    if raw is None:
        raise RuntimeError(f"Sin acceso a velas desde la nube ({last_error})")

    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def analyze(df):
    df = df.copy()

    # Indicadores técnicos base
    df["EMA9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["close"].ewm(span=21, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_signal"]

    df["VOL_AVG"] = df["volume"].rolling(20).mean()

    # Vela cerrada para confirmación oficial (evita repintado)
    candle = df.iloc[-2]
    previous = df.iloc[-3]

    # Vela en curso para radar anticipado
    current = df.iloc[-1]
    prev_current = df.iloc[-2]

    rsi = candle["RSI"]
    vol_avg = candle["VOL_AVG"]
    volume_ratio = (
        float(candle["volume"] / vol_avg)
        if pd.notna(vol_avg) and vol_avg != 0
        else 0.0
    )

    # Filtro anti-lateral: separación mínima de 0.06% entre EMA 9 y EMA 21
    ema_spread_pct = (
        float(abs(candle["EMA9"] - candle["EMA21"]) / candle["close"] * 100)
        if pd.notna(candle["EMA9"]) and pd.notna(candle["EMA21"]) and candle["close"] != 0
        else 0.0
    )
    trend_active = ema_spread_pct >= 0.06

    bull_conds = [
        bool(candle["close"] > candle["EMA9"] > candle["EMA21"]),
        bool(trend_active),
        bool(pd.notna(rsi) and 56 <= rsi <= 78),
        bool(
            pd.notna(candle["MACD_hist"])
            and candle["MACD_hist"] > 0
            and pd.notna(previous["MACD_hist"])
            and candle["MACD_hist"] >= previous["MACD_hist"] * 0.85
        ),
        bool(volume_ratio >= 1.10),
    ]

    bear_conds = [
        bool(candle["close"] < candle["EMA9"] < candle["EMA21"]),
        bool(trend_active),
        bool(pd.notna(rsi) and 22 <= rsi <= 44),
        bool(
            pd.notna(candle["MACD_hist"])
            and candle["MACD_hist"] < 0
            and pd.notna(previous["MACD_hist"])
            and candle["MACD_hist"] <= previous["MACD_hist"] * 0.85
        ),
        bool(volume_ratio >= 1.10),
    ]

    up_score = sum(int(x) for x in bull_conds)
    down_score = sum(int(x) for x in bear_conds)

    up_pct = int(round((up_score / 5) * 100))
    down_pct = int(round((down_score / 5) * 100))

    if up_score >= 4 and up_score > down_score:
        signal = "UP"
        color = "green"
        explanation = (
            f"Alta confluencia alcista ({up_score}/5 filtros · {up_pct}% presión) "
            f"con volumen {volume_ratio:.2f}x."
        )
    elif down_score >= 4 and down_score > up_score:
        signal = "DOWN"
        color = "red"
        explanation = (
            f"Alta confluencia bajista ({down_score}/5 filtros · {down_pct}% presión) "
            f"con volumen {volume_ratio:.2f}x."
        )
    else:
        signal = "LATERAL / NO OPERAR"
        color = "gray"
        if not trend_active:
            explanation = "Filtro activo: EMA 9 y EMA 21 muy pegadas (mercado lateral)."
        elif volume_ratio < 1.10:
            explanation = f"Filtro activo: volumen insuficiente ({volume_ratio:.2f}x < 1.10x)."
        else:
            explanation = "Sin confluencia suficiente (se requieren al menos 4/5 filtros)."

    # Radar anticipado (vela abierta)
    curr_vol_avg = current["VOL_AVG"]
    curr_vol_ratio = (
        float(current["volume"] / curr_vol_avg)
        if pd.notna(curr_vol_avg) and curr_vol_avg != 0
        else 0.0
    )

    early_bull = [
        bool((current["close"] > current["EMA9"]) and (current["EMA9"] >= prev_current["EMA9"])),
        bool(pd.notna(current["RSI"]) and current["RSI"] >= 53),
        bool(
            pd.notna(current["MACD_hist"])
            and pd.notna(prev_current["MACD_hist"])
            and current["MACD_hist"] > prev_current["MACD_hist"]
        ),
        bool(curr_vol_ratio >= 1.0),
    ]

    early_bear = [
        bool((current["close"] < current["EMA9"]) and (current["EMA9"] <= prev_current["EMA9"])),
        bool(pd.notna(current["RSI"]) and current["RSI"] <= 47),
        bool(
            pd.notna(current["MACD_hist"])
            and pd.notna(prev_current["MACD_hist"])
            and current["MACD_hist"] < prev_current["MACD_hist"]
        ),
        bool(curr_vol_ratio >= 1.0),
    ]

    early_up_score = sum(int(x) for x in early_bull)
    early_down_score = sum(int(x) for x in early_bear)
    early_up_pct = int(round((early_up_score / 4) * 100))
    early_down_pct = int(round((early_down_score / 4) * 100))

    return (
        df,
        candle,
        rsi,
        volume_ratio,
        up_score,
        down_score,
        signal,
        color,
        explanation,
        early_up_score,
        early_down_score,
        up_pct,
        down_pct,
        early_up_pct,
        early_down_pct,
    )


@st.cache_data(ttl=30)
def backtest_signals(df, lookback=250):
    if df is None or len(df) < 40:
        return {
            "total": 0,
            "wins": 0,
            "losses": 0,
            "winrate": 0.0,
            "rows": [],
        }

    data = df.copy()
    data["EMA9"] = data["close"].ewm(span=9, adjust=False).mean()
    data["EMA21"] = data["close"].ewm(span=21, adjust=False).mean()

    delta = data["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    ema12 = data["close"].ewm(span=12, adjust=False).mean()
    ema26 = data["close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_hist"] = data["MACD"] - data["MACD_signal"]
    data["VOL_AVG"] = data["volume"].rolling(20).mean()

    start_idx = max(25, len(data) - lookback - 2)
    end_idx = len(data) - 2

    wins = 0
    losses = 0
    history = []

    for i in range(start_idx, end_idx):
        c = data.iloc[i]
        p = data.iloc[i - 1]
        nxt = data.iloc[i + 1]

        vol_ratio = (
            float(c["volume"] / c["VOL_AVG"])
            if pd.notna(c["VOL_AVG"]) and c["VOL_AVG"] != 0
            else 0.0
        )
        spread_pct = (
            float(abs(c["EMA9"] - c["EMA21"]) / c["close"] * 100)
            if pd.notna(c["EMA9"]) and pd.notna(c["EMA21"]) and c["close"] != 0
            else 0.0
        )
        trend_ok = spread_pct >= 0.06
        rsi_val = c["RSI"]

        bull = [
            bool(c["close"] > c["EMA9"] > c["EMA21"]),
            bool(trend_ok),
            bool(pd.notna(rsi_val) and 56 <= rsi_val <= 78),
            bool(
                pd.notna(c["MACD_hist"])
                and c["MACD_hist"] > 0
                and pd.notna(p["MACD_hist"])
                and c["MACD_hist"] >= p["MACD_hist"] * 0.85
            ),
            bool(vol_ratio >= 1.10),
        ]

        bear = [
            bool(c["close"] < c["EMA9"] < c["EMA21"]),
            bool(trend_ok),
            bool(pd.notna(rsi_val) and 22 <= rsi_val <= 44),
            bool(
                pd.notna(c["MACD_hist"])
                and c["MACD_hist"] < 0
                and pd.notna(p["MACD_hist"])
                and c["MACD_hist"] <= p["MACD_hist"] * 0.85
            ),
            bool(vol_ratio >= 1.10),
        ]

        up_s = sum(int(x) for x in bull)
        down_s = sum(int(x) for x in bear)

        sig = None
        if up_s >= 4 and up_s > down_s:
            sig = "UP"
        elif down_s >= 4 and down_s > up_s:
            sig = "DOWN"

        if sig in ("UP", "DOWN"):
            moved_up = bool(nxt["close"] > nxt["open"])
            moved_down = bool(nxt["close"] < nxt["open"])
            won = (sig == "UP" and moved_up) or (sig == "DOWN" and moved_down)

            if won:
                wins += 1
            else:
                losses += 1

            history.append({
                "Señal": sig,
                "Resultado": "ACIERTO" if won else "FALLO",
                "Volumen": f"{vol_ratio:.2f}x",
                "RSI": f"{rsi_val:.1f}" if pd.notna(rsi_val) else "-",
                "Cierre Vela": f"${c['close']:,.2f}",
            })

    total = wins + losses
    winrate = (wins / total * 100) if total > 0 else 0.0

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "winrate": winrate,
        "rows": list(reversed(history[-10:])),
    }


st.title("₿ BTC/USDT PREDICTOR PRO")
st.caption("ANÁLISIS TÉCNICO EN VIVO | TEMPORALIDAD 5 MINUTOS")


@st.fragment(run_every="3s")
def dashboard():
    try:
        raw_df = fetch_klines("BTCUSDT", "5m", 350)
    except Exception as exc:
        st.warning(f"Reconectando con Binance... ({exc})")
        return

    if raw_df is None or len(raw_df) < 35:
        st.info("Cargando datos históricos de velas...")
        return

    result = analyze(raw_df)
    (
        df,
        candle,
        rsi,
        volume_ratio,
        up_score,
        down_score,
        signal,
        color,
        explanation,
        early_up_score,
        early_down_score,
        up_pct,
        down_pct,
        early_up_pct,
        early_down_pct,
    ) = result

    ahora = int(time.time())
    segundos_restantes = 300 - (ahora % 300)
    mins = segundos_restantes // 60
    segs = segundos_restantes % 60
    countdown_str = f"{mins:02d}:{segs:02d}"

    live_price = float(df.iloc[-1]["close"])

    st.subheader("PRECIO BTC / USDT")
    p1, p2 = st.columns(2)
    with p1:
        st.metric("PRECIO EN VIVO", f"${live_price:,.2f} USDT")
    with p2:
        st.metric("CIERRE DE VELA 5M", countdown_str)

    st.divider()
    st.subheader("DECISIÓN DEL SISTEMA · 5M")

    if signal == "UP":
        st.success(f"🟢 {signal}")
    elif signal == "DOWN":
        st.error(f"🔴 {signal}")
    else:
        st.warning(f"🟡 {signal}")

    st.caption(explanation)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("PRESIÓN ALCISTA (CONFIRMADA)", f"{up_pct}%", f"{up_score}/5 filtros")
    with c2:
        st.metric("PRESIÓN BAJISTA (CONFIRMADA)", f"{down_pct}%", f"{down_score}/5 filtros")

    st.markdown("### RADAR ANTICIPADO · 2 MINUTOS")
    if 0 < segundos_restantes <= 120:
        if early_up_score >= 3 and early_up_score > early_down_score:
            st.success(f"🟢 PREALERTA SUBIDA ({early_up_pct}% · {early_up_score}/4)")
        elif early_down_score >= 3 and early_down_score > early_up_score:
            st.error(f"🔴 PREALERTA BAJADA ({early_down_pct}% · {early_down_score}/4)")
        elif early_up_score == 2:
            st.info(f"👀 VIGILANCIA ALCISTA ({early_up_pct}% · {early_up_score}/4)")
        elif early_down_score == 2:
            st.info(f"👀 VIGILANCIA BAJISTA ({early_down_pct}% · {early_down_score}/4)")
        else:
            st.warning("SIN PREALERTA: ESPERANDO CONFIRMACIÓN")
    else:
        st.info("⏳ El radar anticipado se activa en los últimos 2 minutos del cierre de vela de 5m.")

    st.markdown("### HISTORIAL DE SEÑALES · ÚLTIMAS 250 VELAS (5M)")
    bt = backtest_signals(raw_df, lookback=250)

    h1, h2, h3 = st.columns(3)
    with h1:
        st.metric("% ACIERTO HISTÓRICO", f"{bt['winrate']:.1f}%")
    with h2:
        st.metric("ACIERTOS / FALLOS", f"{bt['wins']} / {bt['losses']}")
    with h3:
        st.metric("SEÑALES EMITIDAS", f"{bt['total']}")

    if bt["rows"]:
        st.dataframe(bt["rows"], use_container_width=True, hide_index=True)
    else:
        st.caption("Aún no hubo señales confirmadas (4/5 filtros) en el tramo evaluado.")


dashboard()