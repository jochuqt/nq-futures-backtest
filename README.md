# NQ Futures Backtest

Motor de backtesting para estrategias intradía sobre futuros del Nasdaq-100 (NQ / MNQ),
construido desde cero en Python a partir de datos tick a tick (cada trade) de CME.

*Intraday backtesting engine for Nasdaq-100 futures (NQ/MNQ), built from raw CME
tick-by-tick trade data. Code and docs are in Spanish.*

## Estructura

```
nq-futures-backtest/
├── scripts/
│   ├── datos/          descarga desde Databento, velas, validación
│   ├── backtest/       motores de backtest
│   ├── indicadores/    perfil de volumen y VWAP desde trades
│   └── informes/       informes HTML interactivos
└── estrategias/
    ├── ORB/            Opening Range Breakout + plantilla documentada
    ├── Literatura/     ideas tomadas de papers académicos
    └── POC_VWAP/       rechazo / ruptura de la zona POC + VWAP
```

## Qué hace

1. **Descarga controlada de datos** (`scripts/datos/descargar.py`, `estimar_costos.py`): pide los
   trades de NQ a la API de Databento con controles de gasto: estima el costo antes, se niega si
   supera el crédito disponible, registra cada intento y comprueba que el archivo bajado tenga
   exactamente los trades esperados.
2. **Construcción de velas** (`construir_barras.py`, `obtener_contratos.py`): arma velas de
   1, 5 y 15 minutos desde los trades crudos (formato binario DBN, leído con numpy), con volumen
   y delta, y el nombre real de cada contrato (NQZ5, NQH6...).
3. **Validación de datos** (`validar.py`): huecos, feriados de NYSE, rolls de contrato,
   duplicados y consistencia entre trades y velas.
4. **Backtest** (`scripts/backtest/`): cada estrategia se describe en un archivo `.toml`
   (ver `estrategias/ORB/_plantilla_base.toml`, con todos los parámetros explicados).
   Simulación con velas de 1 minuto, sin mirar el futuro: lo que se decide al cierre de una
   vela se ejecuta en la apertura de la siguiente; si stop y objetivo caen en la misma vela,
   se asume el peor caso. Incluye comisiones, slippage y tamaño de posición por riesgo.
5. **Indicadores desde trades** (`scripts/indicadores/perfil_vwap.py`): perfil de volumen
   (POC, value area 70%) y VWAP con bandas, calculados "en desarrollo" (solo con los trades
   conocidos hasta cada minuto).
6. **Informes** (`scripts/informes/`): HTML interactivos con curva de saldo, mejores y peores
   trades, tablas por mes y comparación de variantes.

## Metodología (para evitar engañarse)

- **In-sample / out-of-sample**: los primeros 6 meses sirven para probar variantes; los
  6 meses siguientes están **bloqueados** y cada variante se puede correr ahí una sola vez
  (queda registrado con una huella de sus reglas). Los criterios para aprobar se escriben
  **antes** de correr el out-of-sample.
- Todas las horas en **hora de Nueva York** (America/New_York).
- Cada corrida guarda una copia de las reglas exactas usadas.

## Estrategias de ejemplo

Son ejemplos de investigación: **ninguna se presenta como rentable**. La mayoría no pasó la
prueba out-of-sample, que es justamente para lo que existe.

## Cómo usarlo

Todos los comandos se corren desde la carpeta raíz del proyecto.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

1. Crear una cuenta en [Databento](https://databento.com) y guardar la clave **solo** como
   variable de entorno `DATABENTO_API_KEY` (nunca dentro de un archivo del proyecto).
2. Copiar `config.example.json` a `config.json` y poner el crédito disponible.
3. Descargar y procesar un mes:

```bash
python -m scripts.datos.descargar --mes 2026-09 --solo-estimar
python -m scripts.datos.descargar --mes 2026-09
python -m scripts.datos.obtener_contratos
python -m scripts.datos.construir_barras
python -m scripts.datos.validar
```

4. Correr un backtest y ver los informes:

```bash
python -m scripts.backtest.backtest --estrategia ORB
python -m scripts.backtest.backtest_literatura
python -m scripts.backtest.backtest_poc_vwap
python -m scripts.informes.grafico_trades --variante 15M_Breakout_SL11
python -m scripts.informes.comparar --periodo in-sample
```

Las fechas de los periodos in-sample / out-of-sample están definidas en
`scripts/backtest/backtest.py`.

## Lo que NO incluye este repositorio

- **Datos de mercado**: son de Databento y su licencia no permite redistribuirlos.
  Cada usuario descarga los suyos.
- **Resultados** generados (contienen precios derivados de esos datos).
- Claves o configuración personal.

## Tecnologías

Python · pandas · numpy · plotly · API REST de Databento · formato binario DBN · zstandard

Desarrollado con ayuda de Claude Code (asistente de programación con IA).
