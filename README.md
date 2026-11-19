# Week 5 — Deep Q-Networks на CartPole

Лабораторная работа: собственная реализация DQN и то же самое на Stable-Baselines3, плюс все упражнения.

| Файл | Что это |
|---|---|
| `Report_DQN_Week5.docx` | Отчёт для сдачи |
| `ANSWERS.md` | Те же ответы в Markdown |
| `dqn_lab.py` | DQN вручную (PyTorch) |
| `dqn_lab_sb3.py` | DQN через Stable-Baselines3 |
| `experiments.py` | Все упражнения: 12 конфигураций × 3 сида |
| `experiments.log`, `results.json` | Вывод и сырые результаты экспериментов |
| `*.png` | Графики |
| `dqn_cartpole.zip` | Сохранённая модель SB3 |

## Запуск

```bash
pip install "gymnasium>=1.0" numpy matplotlib torch stable-baselines3
python dqn_lab.py
python dqn_lab_sb3.py
python experiments.py   # ~2 минуты на 10 ядрах CPU
```
