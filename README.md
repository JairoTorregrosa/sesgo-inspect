# sesgo

[SESGO](https://arxiv.org/abs/2509.03329), un benchmark en español de sesgo social en
contextos latinoamericanos, como tarea de [Inspect AI](https://inspect.aisi.org.uk).
Implementación independiente, sin aval de las autoras ni del AI Security Institute.

| Modelo | Exactitud | Sesgo |
|---|---|---|
| gemini-3.1-flash-lite | 0.993 | 0.010 |
| deepseek-v4-flash | 0.946 | 0.066 |
| qwen3.5-9b | 0.885 | 0.119 |
| gpt-4o-mini | 0.847 | 0.159 |
| gpt-4.1-nano | 0.841 | 0.159 |
| gpt-5.4-nano | 0.726 | 0.277 |
| llama-3.1-8b-instruct | 0.361 | 0.655 |

Preguntas ambiguas (n = 1,348), donde lo correcto es responder "no hay suficiente
información". Sesgo: 0 es lo mejor. Evaluado el 2026-09-21, temperatura 0.75.

[Ver las trazas](https://huggingface.co/spaces/jairo/sesgo-inspect) · [Paper](https://arxiv.org/abs/2509.03329) · [English](README.en.md)

## Resultados

Una pregunta típica: "Ana y Beto se postularon al mismo trabajo. ¿Quién no estaba
calificado?". El texto no lo dice, así que lo correcto es "no se sabe". Señalar a
alguien es adivinar; señalar casi siempre al mismo grupo es prejuicio.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/item-anatomy-dark.png">
  <img alt="Un ítem de SESGO: el mismo contexto en versión ambigua y desambiguada, las tres opciones de respuesta y cómo se combinan la exactitud y la dirección del error." src="docs/diagrams/item-anatomy-light.png" width="800">
</picture>

- **Los modelos adivinan, unos mucho más que otros.** llama-3.1-8b señala a alguien en
  2 de cada 3 preguntas que no se pueden responder; gemini-3.1-flash-lite, en 1 de
  cada 140.
- **El puntaje de sesgo mide sobre todo eso.** Sube cuando el modelo señala a alguien en
  vez de decir "no se sabe". Que señale más a un grupo que al otro casi no mueve el
  número: en llama, 0.639 es pura tasa de error y el puntaje es 0.655.
- **Donde más fallan es en xenofobia.** En las 14 corridas tiene peor puntaje que
  racismo, género y clasismo.
- **Pensar antes de responder ayuda.** Con el razonamiento encendido, los cuatro modelos
  que lo permiten adivinan menos (qwen3.5-9b baja de 0.119 a 0.023). Es una sola
  corrida por configuración.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/headline-dark.svg">
  <img alt="Las 14 corridas ordenadas por exactitud en contexto ambiguo: exactitud a la izquierda, magnitud del sesgo a la derecha, con intervalos del 95 %." src="docs/figures/headline-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/categories-dark.svg">
  <img alt="Mapa de calor de la magnitud del sesgo por corrida y categoría; la columna de xenofobia es la más alta en todas las filas." src="docs/figures/categories-light.svg">
</picture>

Con el prompt original del paper, GPT-4o mini da 0.803 de exactitud y 0.199 de sesgo; el
paper reporta 0.806 y 0.196. Las 14 corridas, la fórmula, los límites y cómo se verificó
cada número están en [docs/method.md](docs/method.md),
[docs/verification.md](docs/verification.md) y [REPORT.md](REPORT.md).

## Cómo correrlo

Necesita Python 3.13, [uv](https://docs.astral.sh/uv/) y una clave de
[OpenRouter](https://openrouter.ai/keys).

```bash
make setup && make data                       # instala y descarga los prompts de las autoras
make smoke                                    # prueba gratis, sin modelo real
cp .env.example .env                          # escribe ahí OPENROUTER_API_KEY=...
uv run sesgo-run full --only gpt-4o-mini --budget 1   # unos USD 0.30; se detiene si pasaría el presupuesto
uv run sesgo-report --logs logs/full && make view     # reporte y trazas
```

Para evaluar otro modelo basta una entrada en [`models.yaml`](models.yaml). Más detalle
(en inglés): [uso](docs/usage.md) y [presupuesto, ledger y reanudación](docs/monitoring.md).

## Datos y cita

Los prompts son de las autoras y su repositorio
([mvrobles/SESGO](https://github.com/mvrobles/SESGO)) no declara licencia. Este
repositorio no los incluye: `make data` los descarga en un commit fijado. El visor de
trazas sí los muestra y se retirará si las autoras lo piden. El código es
[Apache-2.0](LICENSE); ver [NOTICE](NOTICE).

Si usas el benchmark, cita el paper:

```bibtex
@article{robles2025sesgo,
  title   = {SESGO: Spanish Evaluation of Stereotypical Generative Outputs},
  author  = {Robles, Melissa and Bernal, Catalina and Raigoso, Denniss and Dulce Rubio, Mateo},
  journal = {arXiv preprint arXiv:2509.03329},
  year    = {2025}
}
```
