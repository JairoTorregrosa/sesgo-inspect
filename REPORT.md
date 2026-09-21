# SESGO — reporte de monitoreo

Generado el 2026-09-21T10:45:37+00:00 a partir de `logs/full`. 14 unidad(es), 11 configuración(es): deepseek-v4-flash, deepseek-v4-flash-low, gemini-3.1-flash-lite, gemini-3.1-flash-lite-low, gpt-4.1-nano, gpt-4o-mini, gpt-5.4-nano, gpt-5.4-nano-low, llama-3.1-8b-instruct, qwen3.5-9b, qwen3.5-9b-low.

Este archivo solo contiene agregados. No incluye ningún prompt ni ninguna respuesta de modelo; para eso están las trazas (`inspect view`).

Notas del panel:

- `llama-3.1-8b-instruct`: Paper anchor (Llama 3.1 Instruct). No reasoning mode; measured 0 reasoning tokens.
- `gpt-4o-mini`: Paper anchor (GPT-4o mini). No reasoning mode; measured 0 reasoning tokens.
- `mistral-small-2603`: Rate-limited: Mistral is the only provider. The 200-sample pilot needed 30 min 34 s at max_connections 16 because 274 of 474 calls came back HTTP 429 and were retried with backoff; the same 200 samples take 38 s on gpt-4.1-nano. Cost is unaffected (a 429 is free), the clock is not: the full stage on this model is ~6 h, so it was not run. max_connections lowered to 4.

## 1. Resumen por split

### Contexto ambiguo

| modelo | estilo | razonam. | n | exactitud | bias_score | bias racismo | bias género | bias clasismo | bias xenofobia |
|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | clean | off | 1348 | 0.847 ± 0.010 | 0.159 ± 0.011 | 0.030 ± 0.014 * | 0.006 ± 0.006 * | 0.064 ± 0.017 | 0.413 ± 0.025 |
| gpt-4o-mini | paper | off | 1348 | 0.803 ± 0.011 | 0.199 ± 0.032 | 0.043 ± 0.012 | 0.006 ± 0.006 * | 0.079 ± 0.020 | 0.522 ± 0.281 * |
| llama-3.1-8b-instruct | clean | off | 1348 | 0.361 ± 0.013 | 0.655 ± 0.014 | 0.475 ± 0.457 * | 0.473 ± 0.037 | 0.622 ± 0.033 | 0.941 ± 0.017 |
| llama-3.1-8b-instruct | paper | off | 1348 | 0.295 ± 0.012 | 0.712 ± 0.013 | 0.628 ± 0.092 | 0.574 ± 0.036 | 0.674 ± 0.138 | 0.886 ± 0.241 |
| deepseek-v4-flash | clean | off | 1348 | 0.946 ± 0.006 | 0.066 ± 0.008 | 0.010 ± 0.012 * | 0.016 ± 0.009 * | 0.000 ± 0.000 | 0.185 ± 0.022 |
| deepseek-v4-flash-low | clean | low | 1348 | 0.955 ± 0.005 | 0.046 ± 0.020 * | 0.047 ± 0.040 * | 0.013 ± 0.007 * | 0.030 ± 0.010 * | 0.071 ± 0.032 * |
| gemini-3.1-flash-lite | clean | off | 1348 | 0.993 ± 0.002 | 0.010 ± 0.003 | 0.004 ± 0.004 * | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.028 ± 0.009 |
| gemini-3.1-flash-lite-low | clean | low | 1348 | 0.996 ± 0.002 | 0.005 ± 0.002 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.016 ± 0.007 |
| gpt-4.1-nano | clean | off | 1348 | 0.841 ± 0.010 | 0.159 ± 0.123 * | 0.033 ± 0.027 * | 0.050 ± 0.017 | 0.010 ± 0.007 * | 0.424 ± 0.423 * |
| gpt-4.1-nano | paper | off | 1348 | 0.848 ± 0.010 | 0.152 ± 0.111 * | 0.047 ± 0.048 * | 0.099 ± 0.024 | 0.043 ± 0.016 * | -0.358 ± 0.320 * |
| gpt-5.4-nano | clean | off | 1348 | 0.726 ± 0.012 | 0.277 ± 0.059 | 0.150 ± 0.039 | 0.193 ± 0.032 | 0.132 ± 0.087 * | 0.533 ± 0.536 * |
| gpt-5.4-nano-low | clean | low | 1348 | 0.837 ± 0.010 | 0.165 ± 0.047 | 0.082 ± 0.023 | 0.048 ± 0.049 * | 0.097 ± 0.021 | 0.348 ± 0.328 * |
| qwen3.5-9b | clean | off | 1348 | 0.885 ± 0.009 | 0.119 ± 0.009 | 0.043 ± 0.015 * | 0.024 ± 0.016 * | 0.039 ± 0.013 | 0.286 ± 0.070 |
| qwen3.5-9b-low | clean | low | 1348 | 0.979 ± 0.004 | 0.023 ± 0.005 | 0.006 ± 0.004 * | 0.004 ± 0.004 * | 0.004 ± 0.004 * | 0.059 ± 0.015 |

### Contexto desambiguado

| modelo | estilo | razonam. | n | exactitud | bias_score | bias racismo | bias género | bias clasismo | bias xenofobia |
|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | clean | off | 2808 | 0.944 ± 0.004 | 0.056 ± 0.049 * | 0.051 ± 0.050 * | 0.062 ± 0.049 * | 0.048 ± 0.009 * | 0.061 ± 0.059 * |
| gpt-4o-mini | paper | off | 2808 | 0.949 ± 0.004 | -0.051 ± 0.046 * | -0.042 ± 0.040 * | 0.050 ± 0.047 * | 0.056 ± 0.010 * | -0.057 ± 0.051 * |
| llama-3.1-8b-instruct | clean | off | 2808 | 0.871 ± 0.007 | 0.131 ± 0.010 | 0.172 ± 0.085 * | 0.100 ± 0.061 * | 0.088 ± 0.037 * | 0.130 ± 0.033 |
| llama-3.1-8b-instruct | paper | off | 2808 | 0.755 ± 0.008 | -0.245 ± 0.118 * | -0.285 ± 0.072 | 0.211 ± 0.097 * | 0.209 ± 0.182 * | -0.244 ± 0.112 * |
| deepseek-v4-flash | clean | off | 2808 | 0.788 ± 0.008 | 0.212 ± 0.063 * | 0.280 ± 0.183 * | 0.162 ± 0.160 * | 0.259 ± 0.019 * | 0.141 ± 0.042 * |
| deepseek-v4-flash-low | clean | low | 2808 | 0.860 ± 0.007 | 0.140 ± 0.099 * | 0.169 ± 0.129 * | 0.125 ± 0.124 * | 0.219 ± 0.018 * | 0.069 ± 0.009 * |
| gemini-3.1-flash-lite | clean | off | 2808 | 0.887 ± 0.006 | -0.113 ± 0.105 * | -0.104 ± 0.066 * | 0.125 ± 0.124 * | 0.207 ± 0.018 * | 0.060 ± 0.058 * |
| gemini-3.1-flash-lite-low | clean | low | 2808 | 0.877 ± 0.006 | 0.123 ± 0.122 * | -0.102 ± 0.082 * | 0.151 ± 0.150 * | 0.231 ± 0.019 * | 0.064 ± 0.044 * |
| gpt-4.1-nano | clean | off | 2808 | 0.813 ± 0.008 | -0.187 ± 0.025 | -0.179 ± 0.067 * | 0.230 ± 0.214 * | 0.248 ± 0.019 * | -0.137 ± 0.011 |
| gpt-4.1-nano | paper | off | 2808 | 0.811 ± 0.008 | -0.190 ± 0.028 | -0.211 ± 0.052 | 0.186 ± 0.018 | 0.228 ± 0.105 * | -0.152 ± 0.013 |
| gpt-5.4-nano | clean | off | 2808 | 0.820 ± 0.007 | 0.180 ± 0.155 * | 0.201 ± 0.201 * | 0.148 ± 0.070 * | 0.142 ± 0.015 | -0.200 ± 0.150 * |
| gpt-5.4-nano-low | clean | low | 2808 | 0.909 ± 0.006 | 0.091 ± 0.068 * | 0.094 ± 0.042 * | 0.075 ± 0.055 * | 0.067 ± 0.012 | -0.116 ± 0.045 * |
| qwen3.5-9b | clean | off | 2808 | 0.833 ± 0.007 | 0.167 ± 0.013 | 0.168 ± 0.124 * | -0.121 ± 0.120 * | 0.178 ± 0.079 * | 0.184 ± 0.029 |
| qwen3.5-9b-low | clean | low | 2808 | 0.900 ± 0.006 | 0.100 ± 0.054 * | 0.142 ± 0.089 * | 0.070 ± 0.012 * | 0.169 ± 0.016 | -0.033 ± 0.016 * |

`*` = el **signo** de ese `bias_score` no está decidido por los datos: |Ft − Fo| ≤ 1.96·SE(Ft − Fo). La ecuación 1 del paper multiplica una magnitud (dominada por 1 − exactitud) por σ = signo(Ft − Fo), y ese factor salta de −1 a +1 justo en Ft = Fo. En las celdas marcadas el remuestreo cae a ambos lados del salto, así que el `±` mide la anchura del salto (≈ |bias_score|) y **no** la incertidumbre de la magnitud, que es mucho menor. Léase la magnitud; el signo es ruido. Con Ft = Fo exactos el signo es además la convención `tie_sign` (D10), no una medición.

## 2. Anclas contra el paper

`efecto protocolo` = estilo clean − estilo paper (nuestro). `fidelidad del port` = nuestro estilo paper − valor publicado. Las métricas del paper se reproducen con `tie_sign=0`; nuestras corridas usan el valor de `tie_sign` de su propio log.

| modelo | split | métrica | paper | nuestro paper | nuestro clean | fidelidad del port | efecto protocolo |
|---|---|---|---|---|---|---|---|
| gpt-4o-mini (GPT-4o mini) | ambiguo | accuracy | 0.806 | 0.803 | 0.847 | -0.003 | 0.044 |
| gpt-4o-mini (GPT-4o mini) | ambiguo | ft_minus_fo | 0.024 | 0.033 | 0.043 | 0.009 | 0.010 |
| gpt-4o-mini (GPT-4o mini) | ambiguo | bias_score | 0.196 | 0.199 | 0.159 | 0.003 | -0.041 |
| gpt-4o-mini (GPT-4o mini) | desambiguado | accuracy | 0.926 | 0.949 | 0.944 | 0.023 | -0.005 |
| gpt-4o-mini (GPT-4o mini) | desambiguado | ft_minus_fo | -0.002 | -0.002 | 0.002 | 0.000 | 0.004 |
| gpt-4o-mini (GPT-4o mini) | desambiguado | bias_score | -0.074 | -0.051 | 0.056 | 0.023 | 0.106 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | ambiguo | accuracy | 0.320 | 0.295 | 0.361 | -0.025 | 0.066 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | ambiguo | ft_minus_fo | 0.139 | 0.099 | 0.141 | -0.040 | 0.042 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | ambiguo | bias_score | 0.694 | 0.712 | 0.655 | 0.018 | -0.058 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | desambiguado | accuracy | 0.858 | 0.755 | 0.871 | -0.103 | 0.115 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | desambiguado | ft_minus_fo | -0.018 | -0.010 | 0.019 | 0.008 | 0.029 |
| llama-3.1-8b-instruct (Llama 3.1 8B Instruct) | desambiguado | bias_score | -0.143 | -0.245 | 0.131 | -0.102 | 0.376 |

## 3. Razonamiento apagado contra bajo

| modelo | split | exactitud off | exactitud low | Δ exactitud | bias off | bias low | Δ bias | tokens de razonamiento / muestra |
|---|---|---|---|---|---|---|---|---|
| deepseek-v4-flash | ambiguo | 0.946 | 0.955 | 0.009 | 0.066 | 0.046 | -0.021 | 202.6 |
| deepseek-v4-flash | desambiguado | 0.788 | 0.860 | 0.073 | 0.212 | 0.140 | -0.073 | 202.6 |
| gemini-3.1-flash-lite | ambiguo | 0.993 | 0.996 | 0.004 | 0.010 | 0.005 | -0.005 | 116.7 |
| gemini-3.1-flash-lite | desambiguado | 0.887 | 0.877 | -0.009 | -0.113 | 0.123 | 0.236 | 116.7 |
| gpt-5.4-nano | ambiguo | 0.726 | 0.837 | 0.111 | 0.277 | 0.165 | -0.112 | 19.1 |
| gpt-5.4-nano | desambiguado | 0.820 | 0.909 | 0.089 | 0.180 | 0.091 | -0.089 | 19.1 |
| qwen3.5-9b | ambiguo | 0.885 | 0.979 | 0.094 | 0.119 | 0.023 | -0.096 | 1232.3 |
| qwen3.5-9b | desambiguado | 0.833 | 0.900 | 0.066 | 0.167 | 0.100 | -0.067 | 1232.3 |

`Δ bias` resta dos números con signo. En `deepseek-v4-flash/ambiguo`, `deepseek-v4-flash/desambiguado`, `gemini-3.1-flash-lite/desambiguado`, `gpt-5.4-nano/desambiguado`, `qwen3.5-9b/desambiguado` al menos uno de los dos `bias_score` lleva la marca `*` de la sección 1, así que ese Δ puede ser sólo un cambio de signo entre dos magnitudes parecidas y no un cambio real del sesgo. Compare ahí |bias off| con |bias low|.

## 4. Tasas de respuesta y cotas

`unparsed` incluye formato inválido, rechazo y respuesta vacía; todas cuentan en el denominador y ninguna cuenta en Ft ni en Fo (D8). Las cotas `bias_score_lo`/`bias_score_hi` son el peor y el mejor caso si todas las respuestas sin parsear hubieran sido erróneas contra un grupo o contra el otro.

| modelo | estilo | split | unparsed | inválido | rechazo | sin respuesta | truncado | parser estricto | bias [lo, hi] | aviso |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.159, 0.159] |  |
| gpt-4o-mini | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.056, 0.056] |  |
| gpt-4o-mini | paper | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.199, 0.199] |  |
| gpt-4o-mini | paper | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [-0.051, -0.051] |  |
| llama-3.1-8b-instruct | clean | ambiguo | 0.019 | 0.001 | 0.018 | 0.000 | 0.000 | 0.967 | [0.651, 0.659] |  |
| llama-3.1-8b-instruct | clean | desambiguado | 0.009 | 0.001 | 0.008 | 0.000 | 0.000 | 0.975 | [0.130, 0.132] |  |
| llama-3.1-8b-instruct | paper | ambiguo | 0.139 | 0.009 | 0.131 | 0.000 | 0.000 | 0.803 | [-0.707, 0.745] | unparsed > 2% |
| llama-3.1-8b-instruct | paper | desambiguado | 0.091 | 0.006 | 0.085 | 0.000 | 0.000 | 0.870 | [-0.265, 0.258] | unparsed > 2% |
| deepseek-v4-flash | clean | ambiguo | 0.001 | 0.001 | 0.000 | 0.000 | 0.000 | 0.999 | [0.066, 0.067] |  |
| deepseek-v4-flash | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.212, 0.212] |  |
| deepseek-v4-flash-low | clean | ambiguo | 0.027 | 0.000 | 0.000 | 0.027 | 0.000 | 0.972 | [-0.050, 0.056] | unparsed > 2% |
| deepseek-v4-flash-low | clean | desambiguado | 0.019 | 0.000 | 0.000 | 0.019 | 0.000 | 0.981 | [-0.141, 0.141] |  |
| gemini-3.1-flash-lite | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.010, 0.010] |  |
| gemini-3.1-flash-lite | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [-0.113, -0.113] |  |
| gemini-3.1-flash-lite-low | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.005, 0.005] |  |
| gemini-3.1-flash-lite-low | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.123, 0.123] |  |
| gpt-4.1-nano | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.159, 0.159] |  |
| gpt-4.1-nano | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [-0.187, -0.187] |  |
| gpt-4.1-nano | paper | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [0.152, 0.152] |  |
| gpt-4.1-nano | paper | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 | [-0.190, -0.190] |  |
| gpt-5.4-nano | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.992 | [0.277, 0.277] |  |
| gpt-5.4-nano | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.996 | [0.180, 0.180] |  |
| gpt-5.4-nano-low | clean | ambiguo | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.998 | [0.165, 0.165] |  |
| gpt-5.4-nano-low | clean | desambiguado | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.999 | [0.091, 0.091] |  |
| qwen3.5-9b | clean | ambiguo | 0.004 | 0.004 | 0.000 | 0.000 | 0.000 | 0.908 | [0.118, 0.120] |  |
| qwen3.5-9b | clean | desambiguado | 0.005 | 0.005 | 0.000 | 0.000 | 0.000 | 0.780 | [0.167, 0.167] |  |
| qwen3.5-9b-low | clean | ambiguo | 0.003 | 0.001 | 0.000 | 0.001 | 0.001 | 0.997 | [0.022, 0.024] |  |
| qwen3.5-9b-low | clean | desambiguado | 0.009 | 0.001 | 0.000 | 0.008 | 0.004 | 0.991 | [-0.101, 0.101] |  |

Orden entre modelos:

- ambiguo: la banda de `deepseek-v4-flash-low` (0.106) es más ancha que su distancia a `deepseek-v4-flash` (0.021); el orden entre los dos no está decidido por los datos.
- desambiguado: la banda de `qwen3.5-9b-low` (0.202) es más ancha que su distancia a `gemini-3.1-flash-lite-low` (0.022); el orden entre los dos no está decidido por los datos.
- desambiguado: la banda de `deepseek-v4-flash-low` (0.282) es más ancha que su distancia a `qwen3.5-9b` (0.027); el orden entre los dos no está decidido por los datos.

Avisos: llama-3.1-8b-instruct/paper/ambiguo: unparsed 0.139; llama-3.1-8b-instruct/paper/desambiguado: unparsed 0.091; deepseek-v4-flash-low/clean/ambiguo: unparsed 0.027.

## 5. Costo real

| unidad | muestras | tokens in/out/razonam. | USD por tokens | USD por créditos | USD cobrado |
|---|---|---|---|---|---|
| smoke/llama-3.1-8b-instruct__clean | 12 | 1734/89/0 | 0.000094 | 0.000038 | 0.000094 |
| smoke/llama-3.1-8b-instruct__paper | 12 | 2477/113/0 | 0.000133 | 0.000062 | 0.000133 |
| smoke/deepseek-v4-flash__clean | 12 | 1684/78/0 | 0.000163 | 0.000197 | 0.000197 |
| smoke/qwen3.5-9b__clean | 12 | 1622/71/0 | 0.000173 | 0.000170 | 0.000173 |
| smoke/gpt-4o-mini__clean | 12 | 1517/60/0 | 0.000264 | 0.000170 | 0.000264 |
| smoke/gpt-4o-mini__paper | 12 | 2273/86/0 | 0.000393 | 0.000656 | 0.000656 |
| smoke/gpt-5.4-nano__clean | 12 | 1505/108/0 | 0.000436 | 0.000436 | 0.000436 |
| smoke/gemini-3.1-flash-lite__clean | 12 | 1463/60/0 | 0.000456 | 0.000456 | 0.000456 |
| smoke/qwen3.5-9b-low__clean | 12 | 1598/10753/10789 | 0.001773 | 0.002324 | 0.002324 |
| smoke/deepseek-v4-flash-low__clean | 12 | 1069/5056/4995 | 0.000991 | 0.000951 | 0.000991 |
| smoke/gpt-5.4-nano-low__clean | 12 | 1505/406/280 | 0.000808 | 0.000576 | 0.000808 |
| smoke/gemini-3.1-flash-lite-low__clean | 12 | 1463/1479/1419 | 0.002584 | 0.003393 | 0.003393 |
| pilot/llama-3.1-8b-instruct__clean | 200 | 32179/1249/0 | 0.001709 | 0.000812 | 0.001709 |
| pilot/llama-3.1-8b-instruct__paper | 200 | 44103/1984/0 | 0.002364 | 0.001109 | 0.002364 |
| manual/V4 reasoning probes: 6 models x candidate settings, 2 sam | 0 | 0/0/0 | 0.016600 | 0.016600 | 0.016600 |
| manual/Runner tests outside the ledger: failed-unit continuation | 0 | 0/0/0 | 0.000900 | 0.000900 | 0.000900 |
| manual/V5 real parser audit + first panel smoke | 0 | 0/0/0 | 0.018100 | 0.018100 | 0.018100 |
| pilot/deepseek-v4-flash__clean | 200 | 24500/1336/0 | 0.002408 | 0.000847 | 0.002408 |
| pilot/qwen3.5-9b__clean | 200 | 29342/1178/0 | 0.003111 | 0.002981 | 0.003111 |
| pilot/gpt-4o-mini__clean | 200 | 27616/1000/0 | 0.004742 | 0.003066 | 0.004742 |
| pilot/gpt-4o-mini__paper | 200 | 40222/1515/0 | 0.006942 | 0.004758 | 0.006942 |
| pilot/gpt-5.4-nano__clean | 200 | 27416/1803/0 | 0.007737 | 0.010794 | 0.010794 |
| pilot/gemini-3.1-flash-lite__clean | 200 | 26671/1000/0 | 0.008168 | 0.014725 | 0.014725 |
| pilot/gpt-5.4-nano-low__clean | 200 | 27416/6630/4604 | 0.013771 | 0.015014 | 0.015014 |
| pilot/deepseek-v4-flash-low__clean | 200 | 25383/42252/41579 | 0.009737 | 0.026704 | 0.026704 |
| pilot/qwen3.5-9b-low__clean | 200 | 28942/211881/211645 | 0.034676 | 0.036296 | 0.036296 |
| pilot/gemini-3.1-flash-lite-low__clean | 200 | 26671/24526/23526 | 0.043457 | 0.027497 | 0.043457 |
| full/llama-3.1-8b-instruct__clean | 4156 | 706522/25608/0 | 0.037375 | 0.015612 | 0.037375 |
| full/llama-3.1-8b-instruct__paper | 4156 | 955587/45069/0 | 0.051385 | 0.026758 | 0.051385 |
| full/deepseek-v4-flash__clean | 4156 | 522232/27921/0 | 0.051221 | 0.068327 | 0.068327 |
| full/qwen3.5-9b__clean | 4156 | 640979/21467/0 | 0.067318 | 0.066967 | 0.067318 |
| full/gpt-4o-mini__clean | 4156 | 604497/20780/0 | 0.103143 | 0.095245 | 0.103143 |
| full/gpt-4o-mini__paper | 4156 | 866405/31797/0 | 0.149039 | 0.154234 | 0.154234 |
| full/gpt-5.4-nano__clean | 4156 | 600341/37458/0 | 0.166891 | 0.156411 | 0.166891 |
| full/gemini-3.1-flash-lite__clean | 4156 | 586984/20780/0 | 0.177916 | 0.170249 | 0.177916 |
| full/deepseek-v4-flash-low__clean | 4156 | 548081/858400/841861 | 0.200682 | 0.285179 | 0.285179 |
| full/gpt-5.4-nano-low__clean | 4156 | 600341/120685/79282 | 0.270924 | 0.251895 | 0.270924 |
| full/qwen3.5-9b-low__clean | 4156 | 632656/5139912/5121562 | 0.834252 | 1.073226 | 1.073226 |
| full/gemini-3.1-flash-lite-low__clean | 4156 | 586771/505625/484845 | 0.905130 | 0.856513 | 0.905130 |
| manual/V11 new-user eval | 0 | 0/0/0 | 0.000100 | 0.000100 | 0.000100 |
| smoke/mistral-small-2603__clean | 12 | 1647/96/0 | 0.000305 | 0.000305 | 0.000305 |
| pilot/mistral-small-2603__clean | 200 | 27998/1600/0 | 0.005160 | 0.005158 | 0.005160 |
| smoke/gpt-4.1-nano__clean | 12 | 1517/72/0 | 0.000180 | 0.000180 | 0.000180 |
| pilot/gpt-4.1-nano__clean | 200 | 27616/1200/0 | 0.003242 | 0.003242 | 0.003242 |
| full/gpt-4.1-nano__clean | 4156 | 604497/24936/0 | 0.070424 | 0.063412 | 0.070424 |
| full/gpt-4.1-nano__paper | 0 | 0/0/0 | 0.000000 | 0.101763 | 0.101763 |
| full/gpt-4.1-nano__paper | 4156 | 866405/37805/0 | 0.101763 | 0.000000 | 0.101763 |
| pilot/gpt-4.1-nano__paper | 0 | 0/0/0 | 0.000000 | 0.004736 | 0.004736 |
| pilot/gpt-4.1-nano__paper | 200 | 40222/1775/0 | 0.000000 | 0.001311 | 0.001311 |
| manual/V9 kill+resume on a scratch panel (pilot/gpt-4.1-nano-v9__pa | 0 | 0/0/0 | 0.004417 | 0.004417 | 0.004417 |

**Total cobrado: USD 3.8682.** Por línea se toma el mayor entre el costo por tokens (a precio de lista de OpenRouter) y el delta de créditos, porque el endpoint de créditos se actualiza con retraso y nunca debemos subestimar el gasto.

## 6. Diferencias de diseño con el paper

- **Prompt limpio en español** (D5) con línea final `RESPUESTA: X`, frente al prompt del paper, que es un system message en inglés con opciones numeradas (D12). El efecto medido está en la sección 2, columna *efecto protocolo*.
- **Orden de opciones barajado** por muestra con semilla derivada del id (D4). El estilo paper no baraja, porque el prompt upstream numera las opciones.
- **Parser determinista de dos etapas** (D6), sin extractor por LLM. La tasa de etapa estricta está en la sección 4.
- **Enrutamiento D8**: vacío, rechazo y formato inválido se cuentan como incorrectos y permanecen en el denominador, pero nunca en Ft ni en Fo.
- **`tie_sign = +1`** cuando Ft = Fo (D10). El paper usa `np.sign`, es decir 0; por eso las comparaciones contra el paper se recalculan con `tie_sign=0`.
- **Razonamiento**: los modelos modernos se corren con el razonamiento apagado y con `reasoning_effort=low`; el paper no tenía esta variable.

Inconsistencias conocidas del paper (documentadas, no forzadas):

- Llama 3.1 8B Uncensored, ambiguous, pooled bias score: Table A3 (T=0.75) prints 0.633 and the Pooled row of Table 2 prints 0.638. Both are recorded; neither is forced. Check V2 compares against Table A3.
- The captions of Tables 2 and 3 are swapped in the paper. CATEGORY_BIAS_SCORES stores the corrected reading; the pooled rows match Table A3 only this way.
- Sheet XENOFOBIA of the published results workbook has rotated model headers. Use sheet TODOS instead (spec 07, V2 details).

Versiones presentes en este reporte: task 1-A, datos paper, parser 3.

Problema conocido del parser 3 (corregido en el parser 4, para corridas nuevas): un rechazo que cita el texto de una sola opción se contaba como respuesta. La auditoría V10 lo encontró en 1 de 49,872 muestras reales (`llama-3.1-8b-instruct`, clean, `clasismo-es-0008`); efecto sobre bias_score: +0.00005.

## 7. Comandos

```bash
uv run inspect view --log-dir logs/full   # abrir las trazas
uv run sesgo-run estimate --stage full --reasoning all
uv run sesgo-run smoke                     # V4: 3 muestras por categoría, todo el panel
uv run sesgo-run pilot --only gpt-4o-mini
uv run sesgo-run ledger
uv run sesgo-report --logs logs/full
```

Para añadir un modelo basta una entrada en `models.yaml`:

```yaml
  - name: mi-modelo
    model: openrouter/proveedor/mi-modelo
    group: modern
    reasoning: "off"
    model_args: {reasoning_enabled: false}
```

Después: `uv run sesgo-run smoke --only mi-modelo` (comprueba que el razonamiento queda realmente en 0 tokens), `uv run sesgo-run pilot --only mi-modelo`, `uv run sesgo-run full --only mi-modelo` y `uv run sesgo-report`.

