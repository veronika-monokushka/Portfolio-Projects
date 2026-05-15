import numpy as np
import pandas as pd
import os
import time
import random
import pickle
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from datetime import datetime, timedelta

from functions import *
from functions_testing import *

# Отключаем информационные логи
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'



# ==============================================================================
# НАСТРОЙКИ ЭКСПЕРИМЕНТА
# ==============================================================================
#проверить что веса разные
#поменять 100 * 50
#сравнить результаты с 70*50
LAMBDA = [0.25]
EPOCHS = 300
NUM_SAMPLES = 70
NUM_RUNS = 70

neurons = 4
XLA = False
SEED_data = 127354
BATCH_SIZE = 8

# ==============================================================================
# НАСТРОЙКА ФУНКЦИЙ ПОТЕРЬ
# ==============================================================================
loss_functions_by_lambda = {
    0.25: {
        'Welsch': WelschLoss(beta=1.5),
        'Huber': keras.losses.Huber(delta=0.4),
        'Andrews': AndrewsLoss(beta=2.3),
        'Cauchy': CauchyLoss(beta=1.1),
        'Categorical CrossEntropy': 'categorical_crossentropy',
        'Log-Cosh': 'log_cosh'
    }
}

loss_functions_by_lambda = { # epochs = 300
    0.25: {
        'Welsch': WelschLoss(beta=2.5),
        #'Huber': keras.losses.Huber(delta=0.3),
        #'Andrews': AndrewsLoss(beta=5.0),
        'Cauchy': CauchyLoss(beta=2.7),
        'Categorical CrossEntropy': 'categorical_crossentropy',
        'Log-Cosh': 'log_cosh'
    }
}
# Welsch
# batch = 16 - 4 час. - 0.646
# batch = 1 - 12 час. - 0.775
# batch = 8 -  4 час. - 0.691
# batch = 4 -  5.5 час. - 0.738 - компромисс
# Cauchy
# batch = 16 - 4 час. - 0.643

# Huber
# batch = 16 - 4 час. - 0.629
# batch = 4 -  6 час. - 0.695
# ==============================================================================
# ЗАГРУЗКА ДАННЫХ
# ==============================================================================
iris = load_iris()
X = iris.data
y = iris.target
num_classes = len(np.unique(y))
y_one_hot = tf.keras.utils.to_categorical(y, num_classes)
print(f"\n\nДатасет: Iris, образцов: {X.shape[0]}, признаков: {X.shape[1]}, классов: {num_classes}")
print("Генерация зашумленных данных...")
X_noise_dict = generate_noisy_datasets(X, LAMBDA, NUM_SAMPLES, random_seed=SEED_data)
print("=" * 80)

# ==============================================================================
# ПРОВЕРКА И ЗАГРУЗКА ЧЕКПОИНТА (для отображения прогресса)
# ==============================================================================
checkpoint_file = 'experiment_checkpoint.pkl'
completed_info = {}

if os.path.exists(checkpoint_file):
    print(f"\n🔍 Найден чекпоинт: {checkpoint_file}")
    try:
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            results_saved = checkpoint.get('results', {})
            if results_saved:
                print("📊 Сохранённый прогресс:")
                for lam_val, loss_dict in results_saved.items():
                    for loss_name, metrics in loss_dict.items():
                        acc_vals = metrics.get('accuracy', {}).get('all_values', [])
                        if acc_vals:
                            # Определяем количество завершённых генераций
                            completed_gen = len(acc_vals) // NUM_RUNS if NUM_RUNS > 0 else 0
                            completed_info[(lam_val, loss_name)] = completed_gen
                            print(f"  📍 {loss_name} (λ={lam_val}): {completed_gen}/{NUM_SAMPLES} генераций завершено")
    except Exception as e:
        print(f"  ⚠️ Ошибка чтения чекпоинта: {e}")

    response = input("\nВозобновить с последнего сохранения? (y/n): ")
    if response.lower() != 'y':
        backup_name = f"backup_{time.strftime('%Y.%m.%d_%H%M%S')}.pkl"
        os.rename(checkpoint_file, backup_name)
        print(f"Старый чекпоинт сохранён как {backup_name}")
        completed_info = {}
else:
    print("\n📁 Чекпоинт не найден. Эксперимент начнётся с начала.")



# ==============================================================================
# ЗАПУСК ЭКСПЕРИМЕНТА
# ==============================================================================
print("\n" + "=" * 100)
print("\nЗАПУСК ЭКСПЕРИМЕНТА")
print(f"Количество классов: {num_classes}")
print(f"Уровни загрязнения λ: {list(loss_functions_by_lambda.keys())}")
print(f"NUM_SAMPLES: {NUM_SAMPLES}")
print(f"NUM_RUNS: {NUM_RUNS}")
print(f"Эпохи: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Нейроны: {neurons}")
total_sessions = len(loss_functions_by_lambda) * len(loss_functions_by_lambda[0.25]) * NUM_SAMPLES * NUM_RUNS
print(f"Всего обучающих сессий: {total_sessions:,}")
print("⚠️ Для прерывания нажмите Ctrl+C — результаты будут сохранены!")



# ==============================================================================
# ЗАПУСК
# ==============================================================================
try:
    modifed_files, system_messages = Entry_Point_Compare_loss_functions(
        loss_functions_dict_by_lambda=loss_functions_by_lambda,
        X_by_lambda_dict=X_noise_dict,
        y=y_one_hot,
        num_classes=num_classes,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        neurons_hidden=neurons,
        n_runs=NUM_RUNS,
        base_random_state=SEED_data,
        xla=XLA
    )

    print("\n✅ Результаты сохранены в:")
    for f in modifed_files:
        print(modifed_files)
    if system_messages:
        print(f"Сообщения системы: {system_messages}")

except KeyboardInterrupt:
    print("\n\n" + "=" * 100)
    print("⚠️ ЭКСПЕРИМЕНТ ПРЕРВАН ПОЛЬЗОВАТЕЛЕМ")

print("\nЗАВЕРШЕНИЕ")
