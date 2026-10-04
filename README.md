# Deep Learning Labs

Репозиторій містить виконання лабораторних робіт з курсу **Deep Learning**.

## Структура робіт

- **[`lab1`](./lab1)** — Ручна реалізація алгоритму Backpropagation (Iris dataset), верифікація результатів з PyTorch та чисельним градієнтом.
- **[`lab2`](./lab2)** — Двійкова класифікація незбалансованих даних (Credit Card Fraud Detection), підбір оптимального порогу класифікації за матрицею вартості.

> Детальний опис завдання, архітектури та інструкції до запуску знаходяться у `README.md` відповідної папки кожної роботи.

## Запуск та оточення

Для керування залежностями використовується [`uv`](https://github.com/astral-sh/uv). Перейдіть у теку потрібної лабораторної роботи:

```powershell
cd lab1  # або cd lab2
uv sync
```

Запуск Jupyter Notebook:

```powershell
uv run --with jupyter --with ipykernel jupyter notebook
```
