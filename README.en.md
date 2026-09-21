# sesgo

[SESGO](https://arxiv.org/abs/2509.03329), a Spanish-language benchmark of social bias in
Latin American contexts, as an [Inspect AI](https://inspect.aisi.org.uk) task.
Independent implementation, not endorsed by the authors or by the AI Security Institute.

| Model | Accuracy | Bias |
|---|---|---|
| gemini-3.1-flash-lite | 0.993 | 0.010 |
| deepseek-v4-flash | 0.946 | 0.066 |
| qwen3.5-9b | 0.885 | 0.119 |
| gpt-4o-mini | 0.847 | 0.159 |
| gpt-4.1-nano | 0.841 | 0.159 |
| gpt-5.4-nano | 0.726 | 0.277 |
| llama-3.1-8b-instruct | 0.361 | 0.655 |

Ambiguous questions (n = 1,348), where the correct answer is "not enough information".
Bias: 0 is best. Evaluated on 2026-09-21, temperature 0.75.

[Browse the traces](https://huggingface.co/spaces/jairo/sesgo-inspect) · [Paper](https://arxiv.org/abs/2509.03329) · [Español](README.md)

## Results

A typical question: "Ana and Beto applied for the same job. Who was not qualified?"
The text does not say, so the right answer is "can't tell". Naming someone is a guess;
naming the same group almost every time is prejudice.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/diagrams/item-anatomy-dark.png">
  <img alt="One SESGO item: the same context in its ambiguous and disambiguated form, the three answer options, and how accuracy and the direction of error combine." src="docs/diagrams/item-anatomy-light.png" width="800">
</picture>

- **Models guess, some far more than others.** llama-3.1-8b names someone in 2 out of
  3 questions that cannot be answered; gemini-3.1-flash-lite, in 1 out of 140.
- **The bias score mostly measures that.** It rises when a model names someone instead
  of saying "can't tell". Naming one group more than the other barely moves the number:
  for llama, 0.639 is plain error rate and the score is 0.655.
- **Xenophobia is where they fail most.** In all 14 runs it scores worse than racism,
  gender and classism.
- **Thinking before answering helps.** With reasoning switched on, the four models that
  allow it guess less (qwen3.5-9b drops from 0.119 to 0.023). One run per configuration.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/headline-dark.svg">
  <img alt="The 14 runs ordered by ambiguous-context accuracy: accuracy on the left, bias magnitude on the right, with 95% intervals." src="docs/figures/headline-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/categories-dark.svg">
  <img alt="Heatmap of bias magnitude by run and category; the xenophobia column is the highest in every row." src="docs/figures/categories-light.svg">
</picture>

With the paper's original prompt, GPT-4o mini scores 0.803 accuracy and 0.199 bias; the
paper reports 0.806 and 0.196. All 14 runs, the formula, the limits and how each number
was verified are in [docs/method.md](docs/method.md),
[docs/verification.md](docs/verification.md) and [REPORT.md](REPORT.md) (Spanish).

## Run it

Needs Python 3.13, [uv](https://docs.astral.sh/uv/) and an
[OpenRouter](https://openrouter.ai/keys) key.

```bash
make setup && make data                       # install and download the authors' prompts
make smoke                                    # free dry run, no real model
cp .env.example .env                          # write OPENROUTER_API_KEY=... into it
uv run sesgo-run full --only gpt-4o-mini --budget 1   # about USD 0.30; stops if it would pass the budget
uv run sesgo-report --logs logs/full && make view     # report and traces
```

To evaluate another model, add one entry to [`models.yaml`](models.yaml). More:
[usage](docs/usage.md) and [budget, ledger and resume](docs/monitoring.md).

## Data and citation

The prompts belong to the authors, and their repository
([mvrobles/SESGO](https://github.com/mvrobles/SESGO)) declares no license. This
repository does not include them: `make data` downloads them at a pinned commit. The
trace viewer does show them and will be taken down at the authors' request. The code is
[Apache-2.0](LICENSE); see [NOTICE](NOTICE).

If you use the benchmark, cite the paper:

```bibtex
@article{robles2025sesgo,
  title   = {SESGO: Spanish Evaluation of Stereotypical Generative Outputs},
  author  = {Robles, Melissa and Bernal, Catalina and Raigoso, Denniss and Dulce Rubio, Mateo},
  journal = {arXiv preprint arXiv:2509.03329},
  year    = {2025}
}
```
