# Лабораторна робота 2

Модель навчається на Credit Card Fraud Detection. У модель входять тільки `V1 ... V28` і `Amount`; `Time` та `Class` не входять. `Amount` додатково зберігається для аналізу помилок.

## Запуск

Поклади локальний `creditcard.csv` у корінь репозиторію або передай інший шлях:

```powershell
cd lab2
uv sync
uv run python lab2_starter.py --data ..\creditcard.csv --out results
uv run python analyze_thresholds.py --saved results
```

Перша команда навчає модель один раз і зберігає ваги, індекси поділу, мітки та оцінки. Друга працює тільки зі збереженими прогнозами, без CSV і без повторного навчання.

Поділ є стратифікованим 60/20/20 з `random_state=0`, scaler навчається тільки на train. Модель: `Linear(29,32) -> ReLU -> Linear(32,1)`, `BCEWithLogitsLoss`, Adam `lr=1e-3`, 12 епох, batch 1024, CPU, `float32`.

## Поріг

Для `s >= t` вручну обчислюються TP, FP, FN, TN і `C(t) = 10 * FN + FP`. Кандидати - усі унікальні validation scores і `+inf`. Пошук виконується після сортування score та через кумулятивні суми, тому має складність `O(n log n)`. За однакової вартості вибирається більший поріг.

Скрипт зберігає:

- `validation_pr.png` і `validation_roc.png`;
- `validation_threshold_candidates.csv` з повною таблицею кандидатів;
- `validation_threshold_comparison.csv` для `0.5`, `t*` і `+inf`;
- `toy_sets_summary.csv` і `toy_sets_check.csv` для точних наборів A/B/C з умови;
- `test_errors.csv` з першими п'ятьма FP і FN за номером рядка CSV;
- `threshold_report.json` з підсумком validation і test.

Для набору A перевіряються рівні оцінки `0.7`, для B - відсутність позитивних міток, для C - нічия вартості на порогах `0.8` і `0.5`; у разі нічиєї вибирається `0.8`.

Датасет: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Файл `creditcard.csv` до GitHub не додається.
