import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Визначення функцій для моделювання похибок ---

def generate_collimation_error(num_samples):
    """
    Генерує колімаційну похибку (кутові секунди) у заданому несиметричному діапазоні.
    В даному випадку, від +12" до +18".
    """
    return np.random.uniform(12, 18, num_samples) / 3600 * np.pi / 180 # Переведення в радіани

def generate_zero_point_error(num_samples):
    """
    Генерує похибку місця нуля (кутові секунди) у заданому несиметричному діапазоні.
    В даному випадку, від -18" до -12".
    """
    return np.random.uniform(-18, -12, num_samples) / 3600 * np.pi / 180 # Переведення в радіани

def generate_distance_errors(num_samples):
    """
    Генерує адитивну (мм) та мультиплікативну (ppm) похибки відстані.
    Адитивна у несиметричному діапазоні, в даному випадку, від +2 до +3 мм.
    Мультиплікативну лишаємо симетричною.
    """
    additive_error = np.random.uniform(2, 3, num_samples) / 1000 # Переведення в метри
    multiplicative_error_ppm = np.random.uniform(-3, 3, num_samples)
    return additive_error, multiplicative_error_ppm / 1_000_000

# --- 2. Функція для симуляції одного вимірювання з розділеним шумом ---

def simulate_measurement(
    true_horizontal_angle,
    true_zenith_angle, # випадкове значення
    true_distance,     # випадкове значення
    collimation_error_val,
    zero_point_error_val,
    additive_dist_error_val,
    multiplicative_dist_error_val,
    random_noise_std_angle=1,   # Стандартне відхилення випадкового шуму (кутові секунди)
    random_noise_std_distance=0.5 # Стандартне відхилення випадкового шуму (мм)
):
    # Додавання випадкового шуму (окремо для кожного кута)
    random_noise_h_angle = np.random.normal(0, random_noise_std_angle / 3600 * np.pi / 180)
    random_noise_z_angle = np.random.normal(0, random_noise_std_angle / 3600 * np.pi / 180)
    random_noise_distance = np.random.normal(0, random_noise_std_distance / 1000)

    # Застосування систематичних похибок
    measured_horizontal_angle = true_horizontal_angle + collimation_error_val / np.sin(true_zenith_angle) + random_noise_h_angle
    measured_zenith_angle = true_zenith_angle + zero_point_error_val + random_noise_z_angle
    measured_distance = true_distance * (1 + multiplicative_dist_error_val) + additive_dist_error_val + random_noise_distance

    return measured_horizontal_angle, measured_zenith_angle, measured_distance

# --- 3. Основний цикл Монте-Карло ---

def run_monte_carlo_simulation(num_samples):
    results = []

    # Визначення діапазонів для "істинних" значень
    min_true_dist = 10    # метрів
    max_true_dist = 1000  # метрів
    min_true_zenith_angle_deg = 30 # градусів
    max_true_zenith_angle_deg = 150 # градусів

    for meas in range(num_samples):
        # Генеруємо випадкові "істинні" значення для поточної ітерації
        true_h_angle = np.random.uniform(0, 2 * np.pi) # Горизонтальний кут від 0 до 360 градусів (у радіанах)
        true_z_angle_deg = np.random.uniform(min_true_zenith_angle_deg, max_true_zenith_angle_deg)
        true_z_angle_rad = true_z_angle_deg * np.pi / 180 # Переведення в радіани
        true_dist = np.random.uniform(min_true_dist, max_true_dist)

        # Генерація похибок приладу для поточної ітерації
        coll_err = generate_collimation_error(1)[0]
        zero_err = generate_zero_point_error(1)[0]
        add_dist_err, mult_dist_err = generate_distance_errors(1)
        add_dist_err = add_dist_err[0]
        mult_dist_err = mult_dist_err[0]

        # Симуляція вимірювання
        measured_h, measured_z, measured_d = simulate_measurement(
            true_h_angle, true_z_angle_rad, true_dist, # Використовуємо випадкові істинні значення
            coll_err, zero_err, add_dist_err, mult_dist_err
        )

        # Розрахунок похибок
        h_error = (measured_h - true_h_angle) * 180 / np.pi * 3600 # в секундах
        z_error = (measured_z - true_z_angle_rad) * 180 / np.pi * 3600 # в секундах
        d_error = (measured_d - true_dist) * 1000 # в міліметрах

        results.append({
            'true_distance_m': true_dist,  # Додаємо істинні значення для аналізу
            'true_zenith_angle_deg': true_z_angle_deg,
            'collimation_error_sec': coll_err * 180 / np.pi * 3600,
            'zero_point_error_sec': zero_err * 180 / np.pi * 3600,
            'additive_distance_error_mm': add_dist_err * 1000,
            'multiplicative_distance_error_ppm': mult_dist_err * 1_000_000,
            'horizontal_angle_error_sec': h_error,
            'zenith_angle_error_sec': z_error,
            'distance_error_mm': d_error
        })

    return pd.DataFrame(results)

# --- 4. Виконання симуляції та аналіз ---

if __name__ == "__main__":
    num_samples = 5000 # Задаємо кількість вибірок
    df_results = run_monte_carlo_simulation(num_samples)

    print("Перші 5 рядків результатів:")
    print(df_results.head())

    print("\nОписова статистика похибок вимірювань:")
    # Включаємо true_distance і true_zenith_angle в опис
    print(df_results[['horizontal_angle_error_sec', 'zenith_angle_error_sec', 'distance_error_mm',
                      'true_distance_m', 'true_zenith_angle_deg']].describe())

    # --- 5. Візуалізація результатів ---

    # Гістограми розподілу похибок вимірювань
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.histplot(df_results['horizontal_angle_error_sec'], kde=True)
    plt.title('Розподіл похибок горизонтального кута (сек)')
    plt.xlabel('Похибка (сек)')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 2)
    sns.histplot(df_results['zenith_angle_error_sec'], kde=True)
    plt.title('Розподіл похибок зенітного кута (сек)')
    plt.xlabel('Похибка (сек)')
    plt.ylabel('Частота')

    plt.subplot(1, 3, 3)
    sns.histplot(df_results['distance_error_mm'], kde=True)
    plt.title('Розподіл похибок відстані (мм)')
    plt.xlabel('Похибка (мм)')
    plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()

    # Вплив похибок приладу на кутові виміри
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='collimation_error_sec', y='horizontal_angle_error_sec', data=df_results,
                    hue='true_zenith_angle_deg', size='true_zenith_angle_deg', sizes=(20, 200), alpha=0.5,
                    palette='viridis')
    plt.title('Вплив колімаційної похибки на гор. кут (залежить від Зен. Кута)')
    plt.xlabel('Колімаційна похибка (сек)')
    plt.ylabel('Похибка гор. кута (сек)')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='zero_point_error_sec', y='zenith_angle_error_sec', data=df_results, alpha=0.5)
    plt.title('Вплив похибки місця нуля на зен. кут')
    plt.xlabel('Похибка місця нуля (сек)')
    plt.ylabel('Похибка зен. кута (сек)')

    plt.tight_layout()
    plt.show()

    # Вплив похибок приладу на виміри відстані
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='additive_distance_error_mm', y='distance_error_mm', data=df_results, alpha=0.5)
    plt.title('Вплив адитивної похибки на відстань')
    plt.xlabel('Адитивна похибка (мм)')
    plt.ylabel('Похибка відстані (мм)')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='multiplicative_distance_error_ppm', y='distance_error_mm', data=df_results,
                    hue='true_distance_m', size='true_distance_m', sizes=(20, 200), alpha=0.5,
                    palette='plasma')
    plt.title('Вплив мультиплікативної похибки на відстань (залежить від D)')
    plt.xlabel('Мультиплікативна похибка (ppm)')
    plt.ylabel('Похибка відстані (мм)')

    plt.tight_layout()
    plt.show()

    # Залежності похибок вимірів від істинних значень
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='true_zenith_angle_deg', y='horizontal_angle_error_sec', data=df_results, alpha=0.5)
    plt.title('Похибка гор. кута від Зенітного кута (через колімацію)')
    plt.xlabel('Істинний Зенітний кут (град)')
    plt.ylabel('Похибка гор. кута (сек)')

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='true_distance_m', y='distance_error_mm', data=df_results, alpha=0.5)
    plt.title('Похибка відстані від Істинної відстані (через мультиплікативну)')
    plt.xlabel('Істинна відстань (м)')
    plt.ylabel('Похибка відстані (мм)')

    plt.tight_layout()
    plt.show()

    # Кореляційна матриця
    new_column_names = {
        'true_distance_m': 'Істинна відстань (м)',
        'true_zenith_angle_deg': 'Істинний Зен. Кут (град)',
        'collimation_error_sec': 'Колімаційна похибка (сек)',
        'zero_point_error_sec': 'Похибка місця нуля (сек)',
        'additive_distance_error_mm': 'Адитивна похибка D (мм)',
        'multiplicative_distance_error_ppm': 'Мультиплікативна похибка D (ppm)',
        'horizontal_angle_error_sec': 'Похибка гор. кута (сек)',
        'zenith_angle_error_sec': 'Похибка зен. кута (сек)',
        'distance_error_mm': 'Похибка відстані (мм)'
    }
    df_renamed = df_results.rename(columns=new_column_names)

    plt.figure(figsize=(14, 12))
    sns.heatmap(df_renamed.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Кореляційна матриця похибок вимірювань та істинних значень', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()