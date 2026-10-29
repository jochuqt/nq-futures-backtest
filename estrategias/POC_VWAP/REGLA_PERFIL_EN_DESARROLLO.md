# REGLA INVIOLABLE: perfil de volumen en desarrollo

Vale para esta estrategia y para cualquier estrategia que use perfil de volumen.

1. Al probar un trade, el perfil de volumen (POC, VAH, VAL, value area) se calcula
   **solo con los trades ocurridos ANTES del momento en que se ejecuta el trade**.
2. **Nunca** se usa como señal el perfil completo de la sesión (el que se conoce a las
   16:00), porque contiene información del futuro.
3. Si la señal se toma al cierre de una vela, el perfil incluye los trades hasta ese
   cierre y ninguno posterior.
4. El perfil completo de la sesión solo se usa para mirar gráficos después, nunca para
   decidir trades.
5. No hay excepciones: tampoco en pruebas rápidas ni "solo para tener una noción".

En el código: `perfil_en_desarrollo(trades_de_la_sesion, momento)` en
`scripts\indicadores\perfil_vwap.py` usa solo los trades con hora anterior a `momento`.
