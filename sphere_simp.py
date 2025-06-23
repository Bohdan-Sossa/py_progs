import numpy as np
from scipy.optimize import least_squares
import argparse

def load_points_from_file_simplified(filepath):
    """
    Завантажує координати точок з текстового файлу.
    Припускається, що файл коректний та безпомилковий.
    """
    points_data = []
    with open(filepath, 'r') as f:
        for line_content in f:
            line = line_content.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            coords = [float(parts[0]), float(parts[1]), float(parts[2])]
            points_data.append(coords)
    return np.array(points_data)

def sphere_residuals(center_coords, points, radius):
    """
    Обчислює залишки (відхилення від радіуса) для методу найменших квадратів.
    """
    xc, yc, zc = center_coords
    distances = np.sqrt(
        (points[:, 0] - xc)**2 +
        (points[:, 1] - yc)**2 +
        (points[:, 2] - zc)**2
    )
    return distances - radius

def find_sphere_center(points_array, known_radius):
    """
    Знаходить координати центру сфери методом найменших квадратів.
    """
    # Початкове припущення для центру - середнє арифметичне координат точок
    initial_center_guess = np.mean(points_array, axis=0)

    # Використання least_squares для мінімізації залишків
    result = least_squares(
        sphere_residuals,
        initial_center_guess,
        args=(points_array, known_radius),
    )

    return tuple(result.x)
    
def main():
    parser = argparse.ArgumentParser(description="Розрахунок центра сфери з відсіюванням викидів.")
    parser.add_argument("point_file", help="Текстовий файл з координатами точок (x y z на рядок).")
    parser.add_argument("-r", "--radius", type=float, default=0.07267, help="Відомий радіус сфери.")
    parser.add_argument("-i", "--iterations", type=int, default=20, help="Максимальна кількість ітерацій відсіювання (за замовчуванням 20).")
    parser.add_argument("-t", "--threshold_multiplier", type=float, default=3.0,
                        help="Множник для середнього абсолютного відхилення при визначенні порогу відсіювання (за замовчуванням 3.0).")
    args = parser.parse_args()

    known_radius = args.radius
    max_iterations = args.iterations
    threshold_multiplier = args.threshold_multiplier

    # --- Завантаження точок ---
    all_points = load_points_from_file_simplified(args.point_file)
    total_points_loaded = all_points.shape[0]

    # --- Ітераційне відсіювання ---
    current_points = all_points
    num_iterations_performed = 0
    
    # Початковий центр для обробки випадку, якщо перша ітерація відсіє всі точки
    current_center = find_sphere_center(current_points, known_radius)
    if current_center is None: # На випадок, якщо початкові точки занадто погані
        print("Помилка: Не вдалося розрахувати початковий центр сфери. Перевірте вхідні дані.")
        return

    # Маска для відстеження відсіяних точок з початкового набору
    initial_indices = np.arange(total_points_loaded)
    current_indices = initial_indices.copy() # Індекси точок, які зараз враховуються

    for i in range(max_iterations):
        num_iterations_performed = i + 1 # Кількість виконаних ітерацій
        
        if current_points.shape[0] < 3:
            # print("Залишилося менше 3 точок для розрахунку. Зупинка відсіювання.") # Дебаг
            break

        center_candidate = find_sphere_center(current_points, known_radius)
        if center_candidate is None:
            # print(f"Не вдалося розрахувати центр сфери на ітерації {i+1}. Зупинка відсіювання.") # Дебаг
            break # Зупиняємо, зберігаючи останній успішний центр

        current_center = center_candidate

        residuals = sphere_residuals(current_center, current_points, known_radius)
        abs_residuals = np.abs(residuals)
        
        mean_abs_residual = np.mean(abs_residuals)
        
        rejection_threshold = threshold_multiplier * mean_abs_residual
        
        is_not_outlier_mask_current_iter = abs_residuals <= rejection_threshold
        
        next_points = current_points[is_not_outlier_mask_current_iter]
        next_indices = current_indices[is_not_outlier_mask_current_iter]

        num_points_rejected_this_iteration = current_points.shape[0] - next_points.shape[0]

        if num_points_rejected_this_iteration == 0:
            # print("Не відсіяно жодної точки на цій ітерації. Процес відсіювання завершено.") # Дебаг
            break

        current_points = next_points
        current_indices = next_indices
        
    # --- Фінальні результати ---
    num_included_points = current_points.shape[0]
    num_rejected_points = total_points_loaded - num_included_points

    final_residuals = sphere_residuals(current_center, current_points, known_radius)
    final_abs_residuals = np.abs(final_residuals)

    final_mean_abs_residual = np.mean(final_abs_residuals)
    final_max_abs_residual = np.max(final_abs_residuals)


    print("\n--- Результати розрахунку сфери ---")
    print(f"Розрахований центр сфери: X={current_center[0]:.6f}, Y={current_center[1]:.6f}, Z={current_center[2]:.6f}")
    print(f"Кількість виконаних ітерацій відсіювання: {num_iterations_performed}")
    print(f"Кількість врахованих точок: {num_included_points}")
    print(f"Кількість відсіяних точок: {num_rejected_points}")
    print(f"Фінальне середнє абсолютне відхилення: {final_mean_abs_residual:.6f}")
    print(f"Фінальне максимальне абсолютне відхилення: {final_max_abs_residual:.6f}")


if __name__ == "__main__":
    main()
