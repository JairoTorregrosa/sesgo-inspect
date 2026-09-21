# sesgo

[![CI](https://github.com/JairoTorregrosa/sesgo-inspect/actions/workflows/ci.yml/badge.svg)](https://github.com/JairoTorregrosa/sesgo-inspect/actions/workflows/ci.yml)
[![Code license: Apache-2.0](https://img.shields.io/badge/code%20license-Apache--2.0-blue)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue)](pyproject.toml)
[![Inspect AI](https://img.shields.io/badge/built%20on-Inspect%20AI-blue)](https://inspect.aisi.org.uk)

Una implementación en [Inspect AI](https://inspect.aisi.org.uk) de
[SESGO](https://arxiv.org/abs/2509.03329), un benchmark en español de sesgo social en
contextos latinoamericanos. Es una implementación **independiente** y no oficial: no está
afiliada a las autoras del paper, a quienes mantienen Inspect AI ni al AI Security
Institute del Reino Unido, ni cuenta con su aval. [Read it in English](README.md).

## Resultado principal

**Entre los siete modelos evaluados con el prompt por defecto y el razonamiento apagado,
la exactitud en contexto ambiguo fue de 99.3 % a 36.1 %; la mayor parte del "sesgo"
medido fue error, no una preferencia direccional estable.**

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/headline-dark.svg">
  <img alt="Dos gráficas de puntos alineadas con las 14 corridas completas, ordenadas por exactitud en contexto ambiguo: exactitud a la izquierda y magnitud del bias score a la derecha, con intervalos bootstrap del 95 %." src="docs/figures/headline-light.svg">
</picture>

14 corridas completas (7 modelos con el prompt por defecto, 4 de ellos otra vez con
razonamiento bajo y 3 con el prompt del paper), 4,156 prompts cada una, una corrida por
configuración, evaluadas el 2026-09-21 a través de OpenRouter con temperatura 0.75 (la
del paper). Todo, incluidas las etapas de prueba y piloto, costó USD 3.87. Los intervalos
remuestrean ítems; no capturan la variación entre corridas ni entre proveedores. El panel
son modelos pequeños y baratos. Las gráficas están rotuladas en inglés.

## Qué mide SESGO

Cada uno de los 4,156 prompts da un contexto corto, una pregunta y tres opciones: el
grupo históricamente discriminado (`target`), otro grupo (`other`) y "no hay suficiente
información" (`unknown`). Hay cuatro categorías: racismo, género, clasismo y xenofobia.

Los contextos vienen en dos formas. Un contexto **ambiguo** no dice quién hizo qué, así
que `unknown` es la única respuesta correcta y cualquier otra significa que el modelo
llenó el vacío con una suposición. Un contexto **desambiguado** nombra a la persona, así
que una respuesta errónea pasa por encima de la evidencia. Ft y Fo son la fracción de
todas las respuestas de un split que son errores contra el grupo target y contra el otro
grupo; en contexto ambiguo todo error parseado es uno u otro. El bias score del paper
combina la tasa de error y esa diferencia:
signo(Ft − Fo) × √((1 − exactitud)² + (Ft − Fo)²).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/item-anatomy-dark.png">
  <img alt="Anatomía de un ítem de SESGO: un contexto ambiguo y uno desambiguado, los tres roles de respuesta y dónde se separan la exactitud y el sesgo direccional." src="docs/diagrams/item-anatomy-light.png" width="800">
</picture>

Paper: [Robles, Bernal, Raigoso y Dulce Rubio, arXiv:2509.03329](https://arxiv.org/abs/2509.03329).
Prompts: el repositorio de las autoras en el commit fijado que contiene los 4,156 prompts del paper,
[`mvrobles/SESGO@89b8a0e`](https://github.com/mvrobles/SESGO/tree/89b8a0ef69fd86d7f08b99e06f43d16f1c8f599f).

## Resultados

Contexto ambiguo, n = 1,348 por corrida; la última columna es el split desambiguado
(n = 2,808). "prompt del paper" es el system prompt en inglés de las autoras con
opciones numeradas; por defecto se usa un prompt en español con las opciones barajadas.
Todos los puntajes en contexto ambiguo de esta tabla son positivos, así que la columna
es también la magnitud.

| Modelo | Configuración | Exactitud | Bias score | Exactitud (desamb.) |
|---|---|---|---|---|
| gemini-3.1-flash-lite | razonamiento bajo | 0.996 | 0.005 | 0.877 |
| gemini-3.1-flash-lite | por defecto | 0.993 | 0.010 | 0.887 |
| qwen3.5-9b | razonamiento bajo | 0.979 | 0.023 | 0.900 |
| deepseek-v4-flash | razonamiento bajo | 0.955 | 0.046 | 0.860 |
| deepseek-v4-flash | por defecto | 0.946 | 0.066 | 0.788 |
| qwen3.5-9b | por defecto | 0.885 | 0.119 | 0.833 |
| gpt-4.1-nano | prompt del paper | 0.848 | 0.152 | 0.811 |
| gpt-4o-mini | por defecto | 0.847 | 0.159 | 0.944 |
| gpt-4.1-nano | por defecto | 0.841 | 0.159 | 0.813 |
| gpt-5.4-nano | razonamiento bajo | 0.837 | 0.165 | 0.909 |
| gpt-4o-mini | prompt del paper | 0.803 | 0.199 | 0.949 |
| gpt-5.4-nano | por defecto | 0.726 | 0.277 | 0.820 |
| llama-3.1-8b-instruct | por defecto | 0.361 | 0.655 | 0.871 |
| llama-3.1-8b-instruct | prompt del paper | 0.295 | 0.712 | 0.755 |

- **Dispersión entre modelos.** Con el prompt por defecto, los mismos 1,348 prompts
  ambiguos dan un bias score de 0.010 en gemini-3.1-flash-lite y de 0.655 en
  llama-3.1-8b-instruct.
- **Xenofobia tiene el mayor |bias score| de las cuatro categorías en cada una de las 14
  corridas**, de 0.016 (gemini-3.1-flash-lite, razonamiento bajo) a 0.941
  (llama-3.1-8b-instruct).
- **Las cuatro corridas con razonamiento bajo puntuaron menos que su par con razonamiento
  apagado** en contexto ambiguo, p. ej. qwen3.5-9b 0.119 → 0.023 y gpt-5.4-nano
  0.277 → 0.165. Una corrida de cada una.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/categories-dark.svg">
  <img alt="Mapa de calor de la magnitud del bias score en contexto ambiguo por corrida y categoría; la columna de xenofobia es la más oscura en todas las filas." src="docs/figures/categories-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/reasoning-dark.svg">
  <img alt="Gráfica de mancuernas que compara razonamiento apagado y bajo en cuatro modelos: en contexto ambiguo la exactitud sube y la magnitud del sesgo baja." src="docs/figures/reasoning-light.svg">
</picture>

**Contraste con el paper.** GPT-4o mini con el prompt del paper obtiene 0.803 de
exactitud y 0.199 de bias score en contexto ambiguo; la Tabla A3 del paper reporta 0.806
y 0.196.

Los agregados de cada corrida, split y categoría están en
[`results/history.jsonl`](results/history.jsonl) y
[`docs/figures/results.csv`](docs/figures/results.csv); el reporte completo es
[`REPORT.md`](REPORT.md).

**Trazas.** Cada prompt, respuesta y puntaje de nueve de estas corridas se puede explorar
en Inspect View en [huggingface.co/spaces/jairo/sesgo-inspect](https://huggingface.co/spaces/jairo/sesgo-inspect). Prueba el filtro
`metadata.category == "xenofobia" and metadata.context_condition == "ambig" and correct == 0`.

## Interpretación y límites

- **El signo es inestable.** El puntaje es discontinuo en Ft = Fo: un desbalance mínimo
  hacia cualquier lado da cerca de ±(1 − exactitud). `REPORT.md` marca cada celda donde
  |Ft − Fo| ≤ 1.96 errores estándar; ahí se lee la magnitud. Las gráficas muestran solo
  magnitudes.
- **La magnitud es sobre todo la tasa de error.** En todas las corridas de arriba
  domina el término de la tasa de error (gpt-4o-mini: 1 − exactitud = 0.153,
  Ft − Fo = 0.043, puntaje 0.159). Un puntaje alto dice que el modelo respondió cuando
  debía abstenerse, no que prefiera de forma consistente a un grupo.
- **Una nota en un benchmark no es la equidad de un modelo.** Son 4,156 prompts de opción
  múltiple, en un idioma y un formato.
- **Las APIs cambian.** Los modelos alojados por proveedores cambian sin aviso. Los
  números describen el 2026-09-21; los costos usan los precios de lista de OpenRouter de
  ese día.

## Cómo correrlo

Requiere Python 3.13, [uv](https://docs.astral.sh/uv/), `git`, `make` y, para modelos
reales, una clave de [OpenRouter](https://openrouter.ai/keys) con saldo. `make data`
descarga los prompts del repositorio de las autoras, que no declara licencia, y una
corrida los envía al proveedor: lee antes el [aviso sobre los datos](#datos-licencia-cita).

```bash
make setup && make data          # instala; descarga upstream en el commit fijado y convierte en local
make smoke                       # 10 muestras con un modelo simulado: gratis, prueba la tubería
cp .env.example .env             # luego escribe OPENROUTER_API_KEY=... en ese archivo

uv run sesgo-run estimate --stage full --only gpt-4o-mini   # tabla de costos; no llama a nadie
uv run sesgo-run full --only gpt-4o-mini --budget 1         # 2 estilos de prompt x 4,156 muestras, unos USD 0.30
uv run sesgo-run ledger                                     # lo que de verdad se gastó
uv run sesgo-report --logs logs/full                        # results/history.jsonl + REPORT.md
make view                                                   # trazas en Inspect View
```

Ninguna evaluación paga empieza sin una estimación. Una unidad es una configuración con
un estilo de prompt; la compuerta de presupuesto (`--budget`, en USD) se detiene antes de
la primera unidad que lo pasaría, con código de salida 3. Una corrida
interrumpida se reanuda donde quedó, sin muestras duplicadas, y su gasto queda
registrado. El ledger guarda el costo por tokens y el cambio en el saldo del proveedor, y
cobra el mayor.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/pipeline-dark.png">
  <img alt="Tubería: commit fijado de upstream, conversión local, task de Inspect, proveedor, logs reanudables, history.jsonl, reporte y figuras; la compuerta de presupuesto y el ledger controlan el paso de ejecución." src="docs/diagrams/pipeline-light.png" width="800">
</picture>

Para evaluar un modelo nuevo basta una entrada en [`models.yaml`](models.yaml). La
documentación detallada está en inglés: [uso y parámetros](docs/usage.md),
[monitoreo, presupuesto y ledger](docs/monitoring.md), [decisiones de diseño](SPEC.md) y
[specs](specs/).

## Verificación

- **Puntuación.** Las respuestas que las autoras publicaron para seis modelos, vueltas a
  puntuar con este paquete, coinciden en 82 de 84 valores de las tablas del paper. Los
  otros dos se explican por 23 filas que difieren entre dos de los archivos publicados.
- **Reporte.** 135 de 135 números de `REPORT.md` recalculados desde los logs crudos con
  un script que no importa el código del reporte.
- **Empaquetado.** Un clon limpio, siguiendo solo este README, termina en una evaluación
  real.
- **Reanudación.** Una corrida real interrumpida a la mitad y relanzada: conteo exacto de
  muestras, sin duplicados.
- **Gasto.** El ledger conciliado contra el saldo del proveedor.

Comandos y evidencia: [docs/verification.md](docs/verification.md).

## Datos, licencia, cita

**Aviso sobre los datos.** Los prompts de SESGO se obtienen del repositorio público de
las autoras en un commit fijado y se convierten localmente. El repositorio upstream no
declara licencia. Que algo esté disponible públicamente no otorga derecho a
redistribuirlo; este repositorio no redistribuye los prompts ni otorga derechos sobre
ellos. El visor de trazas enlazado arriba muestra prompts dentro de los logs de
evaluación; se retirará si las autoras lo piden. `external/`, `data/` y `logs/` están en `.gitignore` porque contienen prompts;
no los incluyas en nada que publiques. Solo se versionan agregados.

El código tiene licencia [Apache-2.0](LICENSE); ver [NOTICE](NOTICE). Si usas el
benchmark, cita el paper; [`CITATION.cff`](CITATION.cff) trae las dos entradas.

```bibtex
@article{robles2025sesgo,
  title   = {SESGO: Spanish Evaluation of Stereotypical Generative Outputs},
  author  = {Robles, Melissa and Bernal, Catalina and Raigoso, Denniss and Dulce Rubio, Mateo},
  journal = {arXiv preprint arXiv:2509.03329},
  year    = {2025}
}
```
