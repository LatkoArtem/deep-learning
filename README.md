# Лабораторні роботи 1 і 2

У репозиторії дві незалежні роботи:

- `lab1` — базове навчання моделі для виявлення шахрайських транзакцій;
- `lab2` — підбір порога за вартістю помилок `C = 10 * FN + FP`.

Для запуску потрібен `creditcard.csv` з колонками `V1 ... V28`, `Amount` і `Class`.
Файл даних не додається до репозиторію.

```powershell
cd lab1
uv sync
uv run python train.py --data ..\creditcard.csv --out results

cd ..\lab2
uv sync
uv run python lab2_starter.py --data ..\creditcard.csv --out results
uv run python analyze_thresholds.py --saved results
```

Після створення репозиторію на GitHub залишиться виконати:

```powershell
git remote add origin https://github.com/<username>/<repository>.git
git push -u origin main
```

