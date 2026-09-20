# Лабораторна робота 1

Мінімальний baseline: підготовка даних, поділ 60/20/20, масштабування ознак, навчання невеликої нейромережі та оцінювання на validation і test.

```powershell
uv sync
uv run python train.py --data ..\creditcard.csv --out results
```

Результати зберігаються в `results/metrics.json` і `results/predictions.npz`.

