# Лабораторна робота 2

`lab2_starter.py` навчає модель один раз і зберігає прогнози. `analyze_thresholds.py` виконує решту роботи:

- рахує TP, FP, FN, TN і `C = 10 * FN + FP`;
- будує PR і ROC;
- шукає поріг сортуванням та кумулятивними сумами;
- звіряє результат із повним перебором на наборах A, B і C;
- фіксує поріг на validation та оцінює test;
- зберігає приклади FP і FN.

```powershell
uv sync
uv run python lab2_starter.py --data ..\creditcard.csv --out results
uv run python analyze_thresholds.py --saved results
```

Підбір порога виконується тільки на validation. Test використовується лише після фіксації порога.

