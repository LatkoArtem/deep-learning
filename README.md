# Лабораторні роботи 1 і 2

У репозиторії дві роботи:

- `lab1` — ручний backpropagation для Iris із перевіркою PyTorch і чисельною похідною;
- `lab2` — підбір порога для Credit Card Fraud Detection за вартістю `C = 10 * FN + FP`.

Для запуску потрібен `creditcard.csv` з колонками `V1 ... V28`, `Amount` і `Class`.
Файл даних не додається до репозиторію.

```powershell
cd lab1
uv sync
uv run python backprop.py
uv run python backprop.py --wrong-gradient

cd ..\lab2
uv sync
uv run python lab2_starter.py --data ..\creditcard.csv --out results
uv run python analyze_thresholds.py --saved results
```

Репозиторій: https://github.com/LatkoArtem/deep-learning

```powershell
git remote -v
git push -u origin main
```
