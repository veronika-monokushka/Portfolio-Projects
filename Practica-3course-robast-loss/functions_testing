import signal
import sys
import atexit
import pickle
import time
import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers



def signal_handler(sig, frame):
    """Обработчик прерывания — поднимает исключение"""
    print("\n\n" + "=" * 100)
    print("⚠️ ПРЕРЫВАНИЕ ОБНАРУЖЕНО (Ctrl+C)")
    print("Останавливаю текущее обучение и сохраняю результаты...")
    print("=" * 100)

    # Поднимаем исключение, которое можно перехватить
    raise KeyboardInterrupt()
# Регистрируем обработчик
#signal.signal(signal.SIGINT, signal_handler)


# ВСПОМОГАТЕЛЬНЫЕ

def create_tf_dataset(X, y, batch_size, shuffle=True):
    """
    Создаёт эффективный tf.data.Dataset с prefetch для ускорения обучения.

    Параметры:
    ----------
    X : np.array
        Признаки формы (n_samples, n_features)
    y : np.array
        Метки (one-hot) формы (n_samples, num_classes)
    batch_size : int
        Размер батча
    shuffle : bool
        Перемешивать ли данные при каждой эпохе

    Возвращает:
    ----------
    dataset : tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)  # Ключевое ускорение!
    return dataset


# ==============================================================================
# ФУНКЦИИ ДЛЯ ЗАГРУЗКИ ЧЕКПОИНТОВ И ИТОГОВЫХ МЕТРИК
# ==============================================================================
def save_checkpoint(metrics_tuple,
                    lam_value, loss_name, gen_idx, epochs,
                    checkpoint_dir="interrupted_functions"):
    """
    Сохраняет чекпоинт при прерывании эксперимента.
    Создаёт ДВА файла:
    1) .npy файл с 4 массивами метрик
    2) .pkl файл с метаинформацией (lam_value, loss_name, gen_idx)

    Параметры:
    ----------
    metrics_tuple : tuple
        Кортеж из 4 списков: (all_accuracies, all_precisions, all_recalls, all_f1s)
    checkpoint_dir : str
        Папка для сохранения чекпоинтов
    lam_value : float
        Значение λ (0.25, 0.40...)
    loss_name : str
        Имя функции потерь ('Huber', 'Andrews'...)
    gen_idx : int
        Номер генерации, на которой произошло прерывание
    """

    # Создаём папку, если её нет
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Формируем имена файлов
    metrics_filename = os.path.join(checkpoint_dir, f"checkpoint_metrics_{loss_name}_{lam_value}_{epochs}.npy")
    info_filename = os.path.join(checkpoint_dir, f"checkpoint_info_{loss_name}_{lam_value}_{epochs}.pkl")

    # ==========================================================================
    # ФАЙЛ 1: 4 массива метрик (сохраняем как numpy .npy)
    # ==========================================================================
    all_accuracies, all_precisions, all_recalls, all_f1s = metrics_tuple

    # Преобразуем списки в numpy массивы
    acc_array = np.array(all_accuracies, dtype=np.float32)
    prec_array = np.array(all_precisions, dtype=np.float32)
    rec_array = np.array(all_recalls, dtype=np.float32)
    f1_array = np.array(all_f1s, dtype=np.float32)

    # Сохраняем все 4 массива в один .npy файл
    np.savez_compressed(metrics_filename,
                        accuracies=acc_array,
                        precisions=prec_array,
                        recalls=rec_array,
                        f1s=f1_array)

    print(f"📊 Метрики сохранены: {metrics_filename}")
    print(f"   Accuracy: {len(acc_array)} значений, F1: {len(f1_array)} значений")

    # ==========================================================================
    # ФАЙЛ 2: Метаинформация (lam_value, loss_name, gen_idx)
    # ==========================================================================
    info_data = {
        'lam_value': lam_value,
        'loss_name': loss_name,
        'gen_idx': gen_idx,
        'epochs': epochs,
        'num_accuracies': len(all_accuracies),
        'num_precisions': len(all_precisions),
        'num_recalls': len(all_recalls),
        'num_f1s': len(all_f1s),
        'timestamp': datetime.now().isoformat(),
        'file_metrics': metrics_filename
    }

    with open(info_filename, 'wb') as f:
        pickle.dump(info_data, f)

    print(f"📁 Информация сохранена: {info_filename}")
    print(f"   λ={lam_value}, loss={loss_name}, gen_idx={gen_idx}")

    return metrics_filename, info_filename


def load_checkpoint(checkpoint_dir="interrupted_functions"):
    """
    Загружает все чекпоинты из папки.
    Возвращает список словарей с данными.
    """
    import glob

    checkpoints = []

    # Ищем все info файлы
    info_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_info_*.pkl"))

    for info_file in info_files:
        with open(info_file, 'rb') as f:
            info = pickle.load(f)

        # Загружаем соответствующий metrics файл
        metrics_file = info.get('file_metrics')
        if metrics_file and os.path.exists(metrics_file):
            data = np.load(metrics_file)
            info['accuracies'] = data['accuracies'].tolist()
            info['precisions'] = data['precisions'].tolist()
            info['recalls'] = data['recalls'].tolist()
            info['f1s'] = data['f1s'].tolist()

        checkpoints.append(info)

    return checkpoints


def save_one_loss_function(metrics, lam_value, loss_name,
                           epochs, num_samples, num_runs, batch_size,
                           results_folder="experiment_results"):
    """
    Дозаписывает результаты одной функции потерь в CSV файл.

    Параметры:
    ----------
    results : dict
        Словарь с результатами для текущей λ и функции потерь
    lam_value : float
        Значение λ (0.25, 0.40...)
    loss_name : str
        Имя функции потерь ('Huber', 'Andrews'...)
    epochs : int
        Количество эпох
    num_samples : int
        Количество зашумленных выборок (NUM_SAMPLES)
    num_runs : int
        Количество запусков на выборку (NUM_RUNS)
    batch_size : int
        Размер батча
    results_folder : str
        Папка для сохранения результатов
    """

    # Создаём папку, если её нет
    os.makedirs(results_folder, exist_ok=True)

    # Имя файла для конкретной λ
    filename = os.path.join(results_folder, f"results_{lam_value}.csv")

    # Формируем строку с результатами
    row = {
        'loss_function': loss_name,
        'epochs': epochs,
        'num_samples': num_samples,
        'num_runs': num_runs,
        'batch_size': batch_size,
        'accuracy_mean': metrics['accuracy']['mean'],
        'accuracy_std': metrics['accuracy']['std'],
        'accuracy_min': metrics['accuracy']['min'],
        'accuracy_max': metrics['accuracy']['max'],
        'precision_mean': metrics['precision']['mean'],
        'precision_std': metrics['precision']['std'],
        'precision_min': metrics['precision']['min'],
        'precision_max': metrics['precision']['max'],
        'recall_mean': metrics['recall']['mean'],
        'recall_std': metrics['recall']['std'],
        'recall_min': metrics['recall']['min'],
        'recall_max': metrics['recall']['max'],
        'f1_mean': metrics['f1']['mean'],
        'f1_std': metrics['f1']['std'],
        'f1_min': metrics['f1']['min'],
        'f1_max': metrics['f1']['max']
    }

    # Создаём DataFrame из одной строки
    df_row = pd.DataFrame([row])

    # Проверяем, существует ли файл
    file_exists = os.path.exists(filename)

    # Дозаписываем в файл (header только если файла нет)
    df_row.to_csv(filename,
                  mode='a',  # append mode (дозапись)
                  header=not file_exists,  # header только если файла нет
                  index=False)

    print(f"    ✅ {loss_name} λ={lam_value} -> {filename}")
    return filename


class MetricsLogger:
    """Логирует метрики в бинарные файлы с дозаписью в конец (append)"""

    def __init__(self, name_id, log_dir="logs"):
        self.log_dir = log_dir
        self.name_id = name_id
        os.makedirs(log_dir, exist_ok=True)

        # Файлы для каждой метрики
        self.files = {
            'accuracy': open(f"{log_dir}/accuracy_{name_id}.bin", 'ab'),
            'precision': open(f"{log_dir}/precision_{name_id}.bin", 'ab'),
            'recall': open(f"{log_dir}/recall_{name_id}.bin", 'ab'),
            'f1': open(f"{log_dir}/f1_{name_id}.bin", 'ab')
        }

    def log(self, metrics):
        """Добавляет метрики (без перезаписи всего файла)"""
        for key, f in self.files.items():
            arr = np.array(metrics[key], dtype=np.float32)
            arr.tofile(f)  # Дозаписывает в конец файла!

            # 2. Принудительная запись на диск
            f.flush()
            os.fsync(f.fileno())

    def read_all(self, metric_name):
        """Читает все сохранённые метрики"""
        filename = f"{self.log_dir}/{metric_name}_{self.name_id}.bin"
        if not os.path.exists(filename):
            print(f"Файл {filename} не найден")
            return np.array([])

        with open(filename, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
        return

    def close(self):
        for f in self.files.values():
            f.close()

    def delete(self):
        """Удаляет ВСЕ файлы, созданные этим логгером"""
        # Сначала закрываем файлы (если открыты)
        #self.close()

        # Ищем и удаляем все файлы, связанные с этим name_id
        deleted_files = []
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            filename = f"{self.log_dir}/{metric_name}_{self.name_id}.bin"
            if os.path.exists(filename):
                os.remove(filename)
                deleted_files.append(filename)

        # Удаляем папку, если она пуста (опционально)
        try:
            os.rmdir(self.log_dir)  # Удаляет папку ТОЛЬКО если она пуста
            deleted_files.append(f"{self.log_dir}/ (папка удалена)")
        except OSError:
            pass  # Папка не пуста или не существует

        # Выводим информацию об удалении
        if deleted_files:
            print(f"\n🗑️ Удалено {len(deleted_files)} файлов logs")
        else:
            print("⚠️ Нет файлов для удаления")

    def delete_metric(self, metric_name):
        """Удаляет только один файл метрики"""
        filename = f"{self.log_dir}/{metric_name}_{self.name_id}.bin"
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🗑️ Удалён: {filename}")
            return True
        else:
            print(f"⚠️ Файл не найден: {filename}")
            return False



# ==============================================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ==============================================================================

def train_multiple_runs(X_train, y_train, X_test, y_test,
                        loss_func, num_classes, epochs, batch_size,
                        neurons_hidden, n_runs=50, random_state=42, xla=False):
    """
    Обучает модель с возможностью прерывания между запусками.
    """

    best_metrics = {'accuracy': [0]*n_runs, 'precision': [0]*n_runs, 'recall': [0]*n_runs, 'f1': [0]*n_runs}

    # Создаём Dataset один раз на все запуски
    train_dataset = create_tf_dataset(X_train, y_train, batch_size, shuffle=True)
    val_dataset = create_tf_dataset(X_test, y_test, batch_size, shuffle=True)

    for run in range(n_runs):
        run_seed = random_state + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)

        # Создаём новую модель
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(neurons_hidden, activation='sigmoid',
                         kernel_initializer='glorot_uniform'),
            layers.Dense(num_classes, activation='softmax',
                         kernel_initializer='glorot_uniform')
        ])

        model.compile(
            optimizer='adam',
            loss=loss_func,
            metrics=['accuracy'],
            jit_compile=xla
        )

        model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            verbose=0
        )

        # Получаем предсказания
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

        best_metrics['accuracy'][run] = accuracy
        best_metrics['precision'][run] = precision
        best_metrics['recall'][run] = recall
        best_metrics['f1'][run] = f1

        #if run == 0 or run+1 == 25:
            #print(f"    Запуск {run + 1}/{n_runs}, Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return best_metrics


def test_cycles_by_all_parametrs(
        loss_functions_dict_by_lambda,
        X_by_lambda_dict,y,
        NUM_SAMPLES,
        num_classes,
        epochs,
        batch_size,
        neurons_hidden,
        n_runs,
        base_random_state,
        xla=False
):
    modifed_files = []
    system_messages = ""
    for lam_value, loss_functions_dict in loss_functions_dict_by_lambda.items():
        print(f"\n{'-' * 10}")
        print(f"λ = {lam_value}")
        print(f"{'-' * 10}\n")

        X_to_lambda = X_by_lambda_dict[lam_value]
        start_time = 0

        for loss_name, loss_func in loss_functions_dict.items():
            start_time = datetime.now() + timedelta(hours=4)
            print(f"\n>>> Тестируем: {loss_name}")
            print(f"    Текущее время: {start_time.strftime('%Y-%m-%d %H:%M')}\n")
            print("-" * 70)

            all_accuracies = []
            all_precisions = []
            all_recalls = []
            all_f1s = []

            logger = MetricsLogger(f'{lam_value}_{loss_name}_{epochs}', log_dir="logs")
            try:
                for gen_idx in range(NUM_SAMPLES):
                    X_train_gen, X_test_gen, y_train_gen, y_test_gen = train_test_split(
                        X_to_lambda[gen_idx], y,
                        test_size=0.2,
                        stratify=y,
                        random_state=None,
                        shuffle=True
                    )

                    best_metrics = train_multiple_runs(
                        X_train_gen, y_train_gen,
                        X_test_gen, y_test_gen,
                        loss_func, num_classes,
                        epochs, batch_size, neurons_hidden,
                        n_runs=n_runs,
                        random_state=base_random_state + gen_idx * n_runs, xla=xla
                    )

                    all_accuracies.extend(best_metrics['accuracy'])
                    all_precisions.extend(best_metrics['precision'])
                    all_recalls.extend(best_metrics['recall'])
                    all_f1s.extend(best_metrics['f1'])


                    if (gen_idx + 1) % 10 == 0:
                        print(f"  Генерация {gen_idx + 1}/{NUM_SAMPLES}")
                        logger.log({
                            'accuracy': best_metrics['accuracy'],
                            'precision': best_metrics['precision'],
                            'recall': best_metrics['recall'],
                            'f1':best_metrics['f1']
                        })


            except KeyboardInterrupt:
                print(f"\n  ⚠️ Прерывание на генерации {gen_idx + 1}")

                save_checkpoint((all_accuracies, all_precisions, all_recalls, all_f1s),
                                lam_value=lam_value, loss_name=loss_name, gen_idx=gen_idx, epochs=epochs)
                logger.close()

                print("Завершение программы...")
                sys.exit(0)

            except MemoryError:
                # Нехватка памяти
                print(f"\n  ⚠️ ОШИБКА ПАМЯТИ (MemoryError) на генерации {gen_idx + 1}")
                print("   Попробуйте уменьшить NUM_RUNS или NUM_SAMPLES")

                save_checkpoint((all_accuracies, all_precisions, all_recalls, all_f1s),
                                lam_value=lam_value, loss_name=loss_name, gen_idx=gen_idx, epochs=epochs)
                logger.close()
                print("Завершение программы...")
                sys.exit(1)

            except Exception as e:
                # Любая другая Python-ошибка
                print(f"\n  ⚠️ ОШИБКА на генерации {gen_idx + 1}: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()  # Печатаем полный стек ошибки

                save_checkpoint((all_accuracies, all_precisions, all_recalls, all_f1s),
                                lam_value=lam_value, loss_name=loss_name, gen_idx=gen_idx, epochs=epochs)
                logger.close()
                print("Завершение программы...")
                sys.exit(1)

            except:  # Ловит всё остальное (включая некоторые C++ ошибки, но не все)
                print(f"\n  ⚠️ НЕИЗВЕСТНАЯ КРИТИЧЕСКАЯ ОШИБКА на генерации {gen_idx + 1}")
                import traceback
                traceback.print_exc()

                save_checkpoint((all_accuracies, all_precisions, all_recalls, all_f1s),
                                lam_value=lam_value, loss_name=loss_name, gen_idx=gen_idx, epochs=epochs)
                logger.close()
                print("Завершение программы...")
                sys.exit(1)

            logger.close()
            # Сохраняем статистику для этой функции потерь
            if all_accuracies:
                results = {
                    'accuracy': {
                        'mean': np.mean(all_accuracies),
                        'std': np.std(all_accuracies),
                        'min': np.min(all_accuracies),
                        'max': np.max(all_accuracies),
                        'all_values': all_accuracies
                    },
                    'precision': {
                        'mean': np.mean(all_precisions),
                        'std': np.std(all_precisions),
                        'min': np.min(all_precisions),
                        'max': np.max(all_precisions),
                        'all_values': all_precisions
                    },
                    'recall': {
                        'mean': np.mean(all_recalls),
                        'std': np.std(all_recalls),
                        'min': np.min(all_recalls),
                        'max': np.max(all_recalls),
                        'all_values': all_recalls
                    },
                    'f1': {
                        'mean': np.mean(all_f1s),
                        'std': np.std(all_f1s),
                        'min': np.min(all_f1s),
                        'max': np.max(all_f1s),
                        'all_values': all_f1s
                    }
                }

                print(
                    f"\n  ✅ {loss_name}: Acc={results['accuracy']['mean']:.4f}")

                try:
                    filename = save_one_loss_function(results, lam_value, loss_name,
                                       epochs, NUM_SAMPLES, n_runs, batch_size)
                    end_time = datetime.now() + timedelta(hours=4)
                    duration = end_time - start_time
                    print(f"Текущее время: {end_time.strftime('%Y-%m-%d %H:%M')}")
                    print(f'Время выполнения {duration}\n'.split('.')[0])


                    modifed_files.append(filename)
                    logger.delete()

                except FileNotFoundError as e:
                    print(f"   ⚠️ Папка не найдена: {e}")
                    system_messages += f"\n{e}"

                except OSError as e:
                    # Ошибка операционной системы (диск полон, и т.д.)
                    print(f"   ⚠️ Ошибка ОС при сохранении: {e}")
                    system_messages += f"\n{e}"

                except Exception as e:
                    print(f"   ⚠️ Неожиданная ошибка при сохранении: {type(e).__name__}: {e}")
                    system_messages += f"\n{e}"

    return modifed_files, system_messages


def Entry_Point_Compare_loss_functions(
        loss_functions_dict_by_lambda,
        X_by_lambda_dict, y,
        num_classes,
        epochs,
        batch_size,
        neurons_hidden,
        n_runs,
        base_random_state=42,
        xla=False
):
    """
    Сравнивает функции потерь с возможностью прерывания.
    """

    results = {}
    NUM_SAMPLES = X_by_lambda_dict[list(X_by_lambda_dict.keys())[0]].shape[0]


    print("=" * 100, '\n')
    print("START TESTING...")
    print(f"Параметры: Эпохи={epochs}, Батч={batch_size}, Нейроны={neurons_hidden}")
    print(f"Всего сессий на функцию: {NUM_SAMPLES * n_runs}")

    results_files = test_cycles_by_all_parametrs(
        loss_functions_dict_by_lambda,
        X_by_lambda_dict, y,
        NUM_SAMPLES,
        num_classes,
        epochs,
        batch_size,
        neurons_hidden,
        n_runs,
        base_random_state, xla
    )

    return results_files



"""
    Правила чтобы начать с чекпоинта:
            1. Должна существовать та лямбда и loss функция на которой прерывание произошло
            2. По порядку в словаре до неё лямбды и функции не требуют тестирования
"""
