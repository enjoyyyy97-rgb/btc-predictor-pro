"""
BOT BTC/USDT - Panel propio (versión independiente)
Corre en tu PC, sin cuentas de pago. Usa datos públicos de Kraken + Binance.

Uso:
  pip install -r requirements.txt
  python bot.py

Hace:
  1. Lee precio BTC/USDT en vivo (Binance público, sin API key)
  2. Calcula EMA 9/21, RSI 14, MACD, Fibonacci
  3. Genera señal: LONG / SHORT / NEUTRAL con confianza 0-100%
  4. Imprime en consola y guarda historial en señales.csv
  5. Genera panel.html actualizable para ver en el navegador
"""
import time
import csv
import json
import datetime as dt
import urllib.request
import urllib.error

SYMBOL = "BTCUSDT"
INTERVAL = "5m"      # cambia a 5m, 15m o 1h si quieres
LIMIT = 100
REFRESH_SEG = 3

# Kraken funciona en casi todos los países sin bloqueo. Binance como respaldo.
KRAKEN_INTERVAL = {"5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
BINANCE_KLINES = "https://api.binance.com/api/v3/klines?symbol={s}&interval={i}&limit={l}"

def get_json(url, timeout=15):
    req = urllib.request.Request(url, headers={"User-Agent": "btc-bot-pc/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def get_klines_binance():
    kl = get_json(BINANCE_KLINES.format(s=SYMBOL, i=INTERVAL, l=LIMIT))
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    lows = [float(k[3]) for k in kl]
    return closes, highs, lows, "Binance"

def get_klines_kraken():
    iv = KRAKEN_INTERVAL.get(INTERVAL, 5)
    j = get_json(f"https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval={iv}")
    key = [k for k in j["result"].keys() if k != "last"][0]
    velas = j["result"][key][-LIMIT:]
    closes = [float(v[4]) for v in velas]
    highs = [float(v[2]) for v in velas]
    lows = [float(v[3]) for v in velas]
    return closes, highs, lows, "Kraken"

def ema(values, period):
    k = 2 / (period + 1)
    out = []
    e = values[0]
    for v in values:
        e = v * k + e * (1 - k)
        out.append(e)
    return out

def rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i-1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        ag = (ag * (period-1) + gains[i]) / period
        al = (al * (period-1) + losses[i]) / period
    if al == 0:
        return 100.0
    rs = ag / al
    return 100 - (100 / (1 + rs))

def macd(closes, fast=12, slow=26, signal=9):
    ef = ema(closes, fast)
    es = ema(closes, slow)
    macd_line = [a - b for a, b in zip(ef, es)]
    sig = ema(macd_line, signal)
    hist = macd_line[-1] - sig[-1]
    return macd_line[-1], sig[-1], hist

def fibonacci(high, low):
    diff = high - low
    return {
        "0% (max)": high,
        "23.6%": high - 0.236 * diff,
        "38.2%": high - 0.382 * diff,
        "50%": high - 0.5 * diff,
        "61.8%": high - 0.618 * diff,
        "78.6%": high - 0.786 * diff,
        "100% (min)": low,
    }

def analizar():
    fuente = "?"
    try:
        closes, highs, lows, fuente = get_klines_kraken()
    except Exception:
        closes, highs, lows, fuente = get_klines_binance()
    price = closes[-1]

    e9 = ema(closes, 9)[-1]
    e21 = ema(closes, 21)[-1]
    r = rsi(closes)
    m, s, h = macd(closes)
    fib = fibonacci(max(highs[-50:]), min(lows[-50:]))

    # Scoring simple y explicable (no garantiza ganancias)
    score = 0
    motivos = []
    if price > e21:
        score += 20; motivos.append("Precio sobre EMA21 (+alcista)")
    else:
        score -= 20; motivos.append("Precio bajo EMA21 (-bajista)")
    if e9 > e21:
        score += 15; motivos.append("EMA9 sobre EMA21 (+)")
    else:
        score -= 15; motivos.append("EMA9 bajo EMA21 (-)")
    if r > 55:
        score += 20; motivos.append(f"RSI {r:.1f} fuerte (+)")
    elif r < 45:
        score -= 20; motivos.append(f"RSI {r:.1f} débil (-)")
    else:
        motivos.append(f"RSI {r:.1f} neutral")
    if h > 0:
        score += 15; motivos.append("MACD positivo (+)")
    else:
        score -= 15; motivos.append("MACD negativo (-)")

    # Fibonacci: distancia a niveles
    if price > fib["50%"]:
        score += 10; motivos.append("Sobre 50% Fib (+)")
    else:
        score -= 10; motivos.append("Bajo 50% Fib (-)")

    if score >= 30:
        senal, conf = "🟢 LONG / COMPRAR", min(95, 55 + score // 2)
    elif score <= -30:
        senal, conf = "🔴 SHORT / VENDER", min(95, 55 + abs(score) // 2)
    else:
        senal, conf = "⚪ NEUTRAL / ESPERAR", 50

    return {
        "fuente": fuente,
        "hora": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "precio": price,
        "ema9": e9, "ema21": e21, "rsi": r,
        "macd": m, "macd_signal": s, "macd_hist": h,
        "score": score, "senal": senal, "confianza": conf,
        "motivos": motivos, "fib": fib,
    }

def guardar_csv(d, path="señales.csv"):
    existe = False
    try:
        with open(path, "r", encoding="utf-8"):
            existe = True
    except FileNotFoundError:
        pass
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not existe:
            w.writerow(["hora", "precio", "senal", "confianza", "rsi", "ema9", "ema21", "score"])
        w.writerow([d["hora"], f'{d["precio"]:.2f}', d["senal"], d["confianza"],
                    f'{d["rsi"]:.1f}', f'{d["ema9"]:.2f}', f'{d["ema21"]:.2f}', d["score"]])

def main():
    print("=== BOT BTC/USDT propio - PC ===")
    print(f"Par: {SYMBOL} | Temporalidad: {INTERVAL} | Actualiza cada {REFRESH_SEG}s")
    print("Datos: Kraken/Binance públicos (sin claves). No opera solo, solo informa.")
    print("Pulsa CTRL+C para detener.\n")
    while True:
        try:
            d = analizar()
            print(f'[{d["hora"]}] ${d["precio"]:,.2f} ({d["fuente"]}) | {d["senal"]} ({d["confianza"]}%) | RSI {d["rsi"]:.1f} | score {d["score"]}')
            for m in d["motivos"]:
                print(f"   - {m}")
            print("   Fib:", " | ".join(f'{k} ${v:,.0f}' for k, v in d["fib"].items()))
            print()
            guardar_csv(d)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            print(f"Error de red (reintento en {REFRESH_SEG}s): {e}")
        except KeyboardInterrupt:
            print("\nBot detenido. Historial en señales.csv")
            break
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(REFRESH_SEG)

if __name__ == "__main__":
    main()
