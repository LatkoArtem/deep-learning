# Deep Learning Labs

Репозиторій містить виконання лабораторних робіт з курсу **Deep Learning**.

## Структура робіт

- **[`lab1`](./lab1)** — Ручна реалізація алгоритму Backpropagation (Iris dataset), верифікація результатів з PyTorch та чисельним градієнтом.
- **[`lab2`](./lab2)** — Двійкова класифікація незбалансованих даних (Credit Card Fraud Detection), підбір оптимального порогу класифікації за матрицею вартості.
- **[`lab3`](./lab3)** — Сім запусків ResNet-18 на Intel Image Classification, перевірки BatchNorm, розклад ефектів трансферного навчання та тестова оцінка.
- **[`lab4`](./lab4)** — Експорт checkpoint з ЛР3 у ONNX, порівняння FP32 та статичної INT8-квантизації на CPU.

> Детальний опис завдання, архітектури та інструкції до запуску знаходяться у `README.md` відповідної папки кожної роботи.

## Початок після клонування

```powershell
git clone https://github.com/LatkoArtem/deep-learning.git
cd deep-learning
```

Для ЛР3 і ЛР4 потрібен Intel Image Classification з [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). Датасет не зберігається в репозиторії; завантаж і розпакуй його в `data/intel-image-classification`. Контрольні шляхи й команди запуску наведені в [інструкції ЛР3](./lab3/README.md#підготовка-після-клонування) та [інструкції ЛР4](./lab4/README.md#підготовка-після-клонування).

## Запуск та оточення

Для керування залежностями використовується [`uv`](https://github.com/astral-sh/uv). Перейдіть у теку потрібної лабораторної роботи:

```powershell
cd lab1  # або lab2 / lab3 / lab4
uv sync
```

Запуск Jupyter Notebook:

```powershell
uv run --with jupyter --with ipykernel jupyter notebook
```
