import numpy as np
import pandas as pd
import os
import time
import random
import signal
import sys
import atexit
import pickle  # для сохранения чекпоинтов

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from sklearn.datasets import load_iris
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.losses import Loss

def reset_weights(model):
    """Сбрасывает веса модели к начальным значениям"""
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.assign(layer.kernel_initializer(shape=layer.kernel.shape, dtype=layer.kernel.dtype))
        if hasattr(layer, 'bias_initializer') and layer.bias is not None:
            layer.bias.assign(layer.bias_initializer(shape=layer.bias.shape, dtype=layer.bias.dtype))

# LOSS FUNCTIONS ===================

class AndrewsLoss(Loss):
    def __init__(self, beta=1.0, reduction='sum_over_batch_size', name='andrews_loss'):
        """
        Инициализация функции потерь Эндрюса.

        Args:
            beta: Параметр масштаба. Контролирует точку перехода (аналог delta в Huber).
            reduction: Тип редукции ('sum_over_batch_size', 'sum', None).
            name: Имя функции потерь.
        """
        super().__init__(reduction=reduction, name=name)
        self.beta = beta

    def call(self, y_true, y_pred):
        # Вычисляем ошибку (разницу между предсказанием и истиной)
        error = y_pred - y_true

        # Порог: π * beta
        threshold = tf.constant(np.pi) * self.beta

        # Условие: |error| < threshold
        mask = tf.abs(error) < threshold

        # Loss где |error| < threshold: beta * (1 - cos(error / beta))
        loss_inside = self.beta * (1.0 - tf.cos(error / self.beta))

        # Loss где |error| >= threshold: 2 * beta (константа, обрезка)
        loss_outside = 2.0 * self.beta

        # Объединяем с помощью tf.where (векторизованная операция)
        loss = tf.where(mask, loss_inside, loss_outside)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'beta': float(self.beta),
        })
        return config

class CauchyLoss(Loss):
    def __init__(self, beta=1.0, reduction='sum_over_batch_size', name='cauchy_loss'):
        """
        Инициализация функции потерь Коши.

        Args:
            beta: Параметр масштаба. Контролирует робастность.
            reduction: Тип редукции ('sum_over_batch_size', 'sum', None).
            name: Имя функции потерь.
        """
        super().__init__(reduction=reduction, name=name)
        self.beta = beta

    def call(self, y_true, y_pred):
        # Вычисляем ошибку (разницу между предсказанием и истиной)
        error = y_pred - y_true

        # Формула Коши: log(1 + 0.5 * (error / beta)^2)
        # Используем tf.ops для поддержки градиентов
        loss = tf.math.log(1.0 + 0.5 * tf.square(error / self.beta))

        return loss

    def get_config(self):
        # Необходимо для сериализации модели (сохранения/загрузки)
        config = super().get_config()
        config.update({
            'beta': float(self.beta),
        })
        return config

    class WelschLoss(Loss):
        def __init__(self, beta=1.0, reduction='sum_over_batch_size', name='welsch_loss'):
            """
            Инициализация функции потерь Уэлша.

            Args:
                beta: Параметр масштаба. Контролирует робастность (аналог delta в Huber).
                reduction: Тип редукции ('sum_over_batch_size', 'sum', None).
                name: Имя функции потерь.
            """
            super().__init__(reduction=reduction, name=name)
            self.beta = beta

        def call(self, y_true, y_pred):
            error = y_pred - y_true

            # Формула Уэлша: 1 - exp(-0.5 * (error / beta)^2)
            loss = 1.0 - tf.exp(-0.5 * tf.square(error / self.beta))

            return loss

        def get_config(self):
            # Необходимо для сериализации модели (сохранения/загрузки)
            config = super().get_config()
            config.update({
                'beta': float(self.beta),
            })
            return config


        def print_summary_table_with_params(results, total_epochs, total_models, elapsed_time):
            """Выводит таблицу с лучшими параметрами для каждой функции потерь"""
            print("\n" + "=" * 100)
            print("СВОДНАЯ ТАБЛИЦА ЛУЧШИХ РЕЗУЛЬТАТОВ")
            print("=" * 100)

            data = []
            for name, res in results.items():
                param_display = f"{res['param_name']}={res['best_param']:.2f}" if res[
                                                                                      'best_param'] is not None else "N/A"
                data.append({
                    'Функция потерь': name,
                    'Лучший параметр': param_display,
                    'Best Val Acc': f"{res['best_val_acc']:.4f}",
                    'Best Train Acc': f"{res['best_epoch_train_acc']:.4f}",
                    'Gap (T-V)': f"{res['best_epoch_train_acc'] - res['best_val_acc']:.4f}",
                    'Эпоха': res['best_epoch_num'],
                    'Final Val Acc': f"{res['final_val_acc']:.4f}"
                })

            df = pd.DataFrame(data)
            print(df.to_string(index=False))

            print("\n" + "=" * 100)
            print("ОБЩАЯ СТАТИСТИКА ЭКСПЕРИМЕНТА")
            print("=" * 100)
            print(f"Всего функций потерь протестировано: {len(results)}")
            print(f"Всего моделей обучено: {total_models}")
            print(f"Всего эпох обучения (суммарно): {total_epochs}")
            print(f"Затраченное время: {elapsed_time:.2f} сек. ({elapsed_time / 60:.2f} мин.)")
            print(f"Среднее время на одну модель: {elapsed_time / total_models:.2f} сек.")
            print("=" * 100)

            # Определение победителя
            best_loss_name = max(results, key=lambda x: results[x]['best_val_acc'])
            print(f"\n🏆 ПОБЕДИТЕЛЬ: {best_loss_name}")
            print(f"   Параметр: {results[best_loss_name]['param_name']}={results[best_loss_name]['best_param']}")
            print(f"   Точность на валидации: {results[best_loss_name]['best_val_acc']:.4f}")
            print("=" * 100)


        def optimize_loss_functions(
                loss_functions_config,
                X_train, y_train,
                X_test, y_test,
                num_classes,
                epochs=50,
                batch_size=16,
                neurons_hidden=10,
                random_state=42,
                verbose=1
        ):
            """
            Ищет оптимальные параметры для каждой функции потерь по лучшему значению на валидации.

            Параметры:
            ----------
            loss_functions_config : dict
                Словарь {название: {'loss': функция_потерь, 'params': список_значений}}.
                Пример: {
                    'Huber': {'loss': keras.losses.Huber, 'params': [0.5, 1.0, 2.0]},
                    'CrossEntropy': {'loss': 'categorical_crossentropy', 'params': [None]}
                }
            X_train, y_train : np.array
                Обучающие данные и метки (One-Hot).
            X_test, y_test : np.array
                Тестовые данные и метки.
            num_classes : int
                Количество классов.
            epochs : int
                Количество эпох для каждого запуска.
            batch_size : int
                Размер батча.
            neurons_hidden : int
                Количество нейронов в скрытом слое.
            random_state : int
                Зерно для воспроизводимости.
            verbose : int
                Уровень детализации вывода (0=тихо, 1=нормально, 2=подробно).

            Возвращает:
            -----------
            results : dict
                Словарь с лучшими результатами для каждой функции потерь.
            """

            results = {}
            total_epochs_trained = 0
            total_models_trained = 0

            print("=" * 100)
            print("НАЧАЛО ПОИСКА ОПТИМАЛЬНЫХ ПАРАМЕТРОВ ФУНКЦИЙ ПОТЕРЬ")
            print("=" * 100)
            print(f"Общее количество эпох на одну конфигурацию: {epochs}")
            print(f"Размер батча: {batch_size}")
            print("=" * 100)

            start_time = time.time()

            for loss_name, config in loss_functions_config.items():
                loss_class = config['loss']
                param_values = config['params']
                param_name = config.get('param_name', 'param')  # Имя параметра (beta, delta и т.д.)

                print(f"\n{'=' * 100}")
                print(f">>> Оптимизация: {loss_name}")
                print(f"{'=' * 100}")
                print(f"Диапазон параметра {param_name}: {param_values}")
                print("-" * 100)

                best_result = None
                best_param_value = None
                param_history = []

                for param_value in param_values:
                    # Сброс сессии для чистоты эксперимента
                    tf.keras.backend.clear_session()
                    np.random.seed(random_state)
                    tf.random.set_seed(random_state)

                    # Формирование функции потерь
                    if param_value is None:
                        # Для функций без параметров (например, CrossEntropy)
                        loss_func = loss_class
                        display_param = "N/A"
                    else:
                        # Для функций с параметрами (Huber, Andrews и т.д.)
                        if isinstance(loss_class, str):
                            loss_func = loss_class  # Строка, например 'categorical_crossentropy'
                        else:
                            loss_func = loss_class(**{param_name: param_value})
                        display_param = f"{param_value:.2f}" if isinstance(param_value, float) else param_value

                    if verbose >= 1:
                        print(f"  Тест параметра {param_name}={display_param}...", end=" ")

                    # Построение модели
                    model = keras.Sequential([
                        layers.Input(shape=(X_train.shape[1],)),
                        layers.Dense(neurons_hidden, activation='sigmoid'),
                        layers.Dense(num_classes, activation='softmax')
                    ])

                    model.compile(
                        optimizer='adam',
                        loss=loss_func,
                        metrics=['accuracy']
                    )

                    # Обучение
                    history = model.fit(
                        X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(X_test, y_test),
                        shuffle=True,
                        verbose=0
                    )

                    total_epochs_trained += epochs
                    total_models_trained += 1

                    # Анализ результатов
                    val_accuracies = history.history['val_accuracy']
                    train_accuracies = history.history['accuracy']
                    best_epoch_idx = np.argmax(val_accuracies)
                    best_val_acc = val_accuracies[best_epoch_idx]

                    current_result = {
                        'history': history,
                        'model': model,
                        'param_value': param_value,
                        'final_train_loss': history.history['loss'][-1],
                        'final_val_loss': history.history['val_loss'][-1],
                        'final_train_acc': history.history['accuracy'][-1],
                        'final_val_acc': history.history['val_accuracy'][-1],
                        'best_val_acc': best_val_acc,
                        'best_epoch_train_acc': train_accuracies[best_epoch_idx],
                        'best_epoch_num': best_epoch_idx + 1
                    }

                    param_history.append(current_result)

                    if verbose >= 1:
                        print(f"Best Val Acc: {best_val_acc:.4f} (Эпоха {best_epoch_idx + 1})")

                    # Обновление лучшего результата
                    if best_result is None or best_val_acc > best_result['best_val_acc']:
                        best_result = current_result
                        best_param_value = param_value

                # Сохранение лучшего результата для этой функции потерь
                results[loss_name] = {
                    'best_param': best_param_value,
                    'param_name': param_name,
                    'all_params_results': param_history,
                    **best_result  # Распаковываем лучшие метрики
                }

                print(f"\n  ✓ ЛУЧШИЙ РЕЗУЛЬТАТ для {loss_name}:")
                print(f"    Параметр {param_name} = {best_param_value if best_param_value is not None else 'N/A'}")
                print(f"    Лучшая точность на валидации: {best_result['best_val_acc']:.4f}")
                print(
                    f"    Эпоха: {best_result['best_epoch_num']}, Train Acc: {best_result['best_epoch_train_acc']:.4f}")

            elapsed_time = time.time() - start_time

            # Вывод сводной таблицы
            print_summary_table_with_params(results, total_epochs_trained, total_models_trained, elapsed_time)

            return results


class WelschLoss(Loss):
    def __init__(self, beta=1.0, reduction='sum_over_batch_size', name='welsch_loss'):
        """
        Инициализация функции потерь Уэлша.

        Args:
            beta: Параметр масштаба. Контролирует робастность (аналог delta в Huber).
            reduction: Тип редукции ('sum_over_batch_size', 'sum', None).
            name: Имя функции потерь.
        """
        super().__init__(reduction=reduction, name=name)
        self.beta = beta

    def call(self, y_true, y_pred):
        error = y_pred - y_true

        # Формула Уэлша: 1 - exp(-0.5 * (error / beta)^2)
        loss = 1.0 - tf.exp(-0.5 * tf.square(error / self.beta))

        return loss

    def get_config(self):
        # Необходимо для сериализации модели (сохранения/загрузки)
        config = super().get_config()
        config.update({
            'beta': float(self.beta),
        })
        return config


# DATA ===============================

def generate_noisy_datasets(X, lambda_values, num_samples, random_seed=42):
    """
    Генерирует зашумленные выборки исходных данных с различным распределнием шумов для разных λ.

    Параметры:
    ----------
    X : np.array
        Исходные данные (чистые)
    lambda_values : list
        Список значений λ для генерации
    num_samples : int
        Количество зашумленных выборок для каждого λ
    random_seed : int
        Начальное значение для генератора случайных чисел

    Возвращает:
    -----------
    noisy_datasets : dict
        Словарь {λ: np.array формы (num_samples, n_samples, n_features)}
    """
    # Настройка уровней шума
    noise_config = {
        2: (30, 120),   # Признак 3: (ρ_low, ρ_high)
        3: (40, 150),   # Признак 4: (ρ_low, ρ_high)
    }

    print(f"Исходные данные: {X.shape[0]} образцов, {X.shape[1]} признака")
    print()

    # Создание инжектора шума
    noise_injector = NoiseInjector(noise_levels=noise_config)
    noise_injector.fit(X)

    # Словарь для хранения зашумленных данных
    noisy_datasets = {}

    for lam_value in lambda_values:
        # Список для хранения всех генераций
        X_noisy_list = []

        for i in range(num_samples):
            X_noisy = noise_injector.transform(X, lambda_param=lam_value, random_state=random_seed + i)
            X_noisy_list.append(X_noisy)

        # Сохраняем как numpy массив: (num_samples, n_samples, n_features)
        noisy_datasets[lam_value] = np.array(X_noisy_list)

    print(f"Размер массива для каждого λ: {noisy_datasets[lam_value].shape}")

    return noisy_datasets

class NoiseInjector:
    """
    Класс для добавления шума к признакам 3 и 4 по формулам смеси распределений.

    Формулы:
    --------
    (2.4) ρ_ij = (σ_ij / c) * 100%  →  σ_ij = (ρ_ij / 100) * c
    (3.1) x̃_mi = x_mi + ε_mi
    (3.2) F_i(x) = (1-λ)F₁(x, 0, σ_i1) + λF₂(x, 0, σ_i2)
    """

    def __init__(self, noise_levels):
        """
        Инициализация инжектора шума.

        Параметры:
        ----------
        noise_levels : dict
            Уровни шума ρ (%) для каждого зашумляемого признака.
            Формат: {индекс_признака: (ρ_low, ρ_high)}
            Пример: {2: (30, 120), 3: (40, 150)}  # для признаков 3 и 4
        """
        self.noise_levels = noise_levels
        self.c_squared = None
        self.c = None
        self.is_fitted = False

    def fit(self, X):
        """
        Вычисление дисперсии чистой выборки (c²) по формуле (2.4).

        Параметры:
        ----------
        X : np.array
            Чистые данные (без шума).
        """

        # Вычисляем дисперсию по всем признакам (или можно по конкретным)
        self.c_squared = np.var(X)
        self.c = np.sqrt(self.c_squared)
        self.is_fitted = True

        return self

    def _get_noise_sigma(self, feature_idx, component=1):
        """
        Вычисление σ_ij по формуле (2.4): σ_ij = (ρ_ij / 100) * c

        Параметры:
        ----------
        feature_idx : int
            Индекс признака (0-based).
        component : int
            Компонента смеси (1 или 2).
            component=1: σ_i1 (основное распределение)
            component=2: σ_i2 (засоряющее распределение)
        """
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit() для вычисления c")

        rho_low, rho_high = self.noise_levels[feature_idx]

        if component == 1:
            sigma = (rho_low / 100.0) * self.c
        else:
            sigma = (rho_high / 100.0) * self.c

        return sigma

    def transform(self, X, lambda_param, random_state=None):
        """
        Добавление шума к признакам 3 и 4 по формуле смеси (3.2).

        Параметры:
        ----------
        X : np.array
            Исходные данные (чистые).
        lambda_param : float
            Параметр смеси λ ∈ [0, 1] - доля засоряющих наблюдений.
            λ = 0: только чистый шум (σ_i1)
            λ = 1: только загрязняющий шум (σ_i2)

        Возвращает:
        -----------
        X_noisy : np.array
            Данные с добавленным шумом.
        """
        if not self.is_fitted:
            raise ValueError("Сначала вызовите fit() для вычисления c")

        if not 0 <= lambda_param <= 1:
          raise ValueError(f"lambda_param должен быть в [0, 1], получено {lambda_param}")

        rng = np.random.RandomState(random_state)

        X_noisy = X.copy().astype(np.float64)
        n_samples = X.shape[0]

        # Добавляем шум только к признакам 3 и 4 (индексы 2 и 3)
        for feature_idx in self.noise_levels.keys():
            mixture_mask = rng.random(n_samples) < lambda_param

            sigma_1 = self._get_noise_sigma(feature_idx, component=1)
            noise_1 = rng.normal(loc=0, scale=sigma_1, size=n_samples)

            sigma_2 = self._get_noise_sigma(feature_idx, component=2)
            noise_2 = rng.normal(loc=0, scale=sigma_2, size=n_samples)

            noise = np.where(mixture_mask, noise_2, noise_1)

            X_noisy[:, feature_idx] = X[:, feature_idx] + noise

        return X_noisy

    def fit_transform(self, X, lambda_param, random_state=None):
        """Удобный метод: fit + transform в одном вызове"""
        self.fit(X)
        return self.transform(X, lambda_param, random_state=random_state)
