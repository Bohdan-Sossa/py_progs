import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# --- 1. Функції для генерації істинних координат та вимірювань ---

def generate_true_coordinates(
    room_length,
    room_width,
    room_height,
    num_points_per_surface=500
):
    """
    Генерує "істинні" координати приладу та точок вимірювання
    в заданій системі координат приміщення.
    """
    dist_from_wall_1 = np.random.uniform(3, 4)
    dist_from_wall_2 = np.random.uniform(3, 4)
    instrument_height = np.random.uniform(1, 2)

    instrument_x = dist_from_wall_1
    instrument_y = dist_from_wall_2
    instrument_z = instrument_height

    instrument_coords = np.array([instrument_x, instrument_y, instrument_z])

    all_true_points = []

    # Стіна X=0
    points_wall_x0 = np.array([
        np.full(num_points_per_surface, 0.0),
        np.random.uniform(0, room_width, num_points_per_surface),
        np.random.uniform(0, room_height, num_points_per_surface)
    ]).T
    all_true_points.extend([list(p) + ['Wall_X0'] for p in points_wall_x0])

    # Стіна X=room_length
    points_wall_x_max = np.array([
        np.full(num_points_per_surface, room_length),
        np.random.uniform(0, room_width, num_points_per_surface),
        np.random.uniform(0, room_height, num_points_per_surface)
    ]).T
    all_true_points.extend([list(p) + ['Wall_X_max'] for p in points_wall_x_max])

    # Стіна Y=0
    points_wall_y0 = np.array([
        np.random.uniform(0, room_length, num_points_per_surface),
        np.full(num_points_per_surface, 0.0),
        np.random.uniform(0, room_height, num_points_per_surface)
    ]).T
    all_true_points.extend([list(p) + ['Wall_Y0'] for p in points_wall_y0])

    # Стіна Y=room_width
    points_wall_y_max = np.array([
        np.random.uniform(0, room_length, num_points_per_surface),
        np.full(num_points_per_surface, room_width),
        np.random.uniform(0, room_height, num_points_per_surface)
    ]).T
    all_true_points.extend([list(p) + ['Wall_Y_max'] for p in points_wall_y_max])

    # Стеля (Z=room_height)
    points_ceiling = np.array([
        np.random.uniform(0, room_length, num_points_per_surface),
        np.random.uniform(0, room_width, num_points_per_surface),
        np.full(num_points_per_surface, room_height)
    ]).T
    all_true_points.extend([list(p) + ['Ceiling'] for p in points_ceiling])

    # Підлога (Z=0)
    points_floor = np.array([
        np.random.uniform(0, room_length, num_points_per_surface),
        np.random.uniform(0, room_width, num_points_per_surface),
        np.full(num_points_per_surface, 0.0)
    ]).T
    all_true_points.extend([list(p) + ['Floor'] for p in points_floor])

    true_points_df = pd.DataFrame(all_true_points, columns=['X', 'Y', 'Z', 'Surface'])


    return instrument_coords, true_points_df

def visualize_room_and_points(instrument_coords, true_points_df,
                               room_length, room_width, room_height):
    """
    Візуалізує приміщення, розташування приладу та "істинні" точки.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Візуалізація приміщення (каркас)
    verts = [
        (0, 0, 0),
        (0, room_length, 0),
        (room_width, room_length, 0),
        (room_width, 0, 0),

        (0, 0, room_height),
        (0, room_length, room_height),
        (room_width, room_length, room_height),
        (room_width, 0, room_height)
    ]
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]

    for edge in edges:
        x, y, z = zip(*[verts[edge[0]], verts[edge[1]]])
        ax.plot(x, y, z, color='gray', linestyle='--')

    # Візуалізація приладу
    ax.scatter(instrument_coords[0], instrument_coords[1], instrument_coords[2],
               color='red', marker='^', s=200, label='Instrument')

    # Візуалізація точок за поверхнями
    surfaces = true_points_df['Surface'].unique()
    colors_map = plt.get_cmap('tab10', len(surfaces))

    for i, surface in enumerate(surfaces):
        subset = true_points_df[true_points_df['Surface'] == surface]
        ax.scatter(subset['Y'], subset['X'], subset['Z'], color=colors_map(i), label=surface, s=10, alpha=0.6)

    ax.set_xlabel('Y (East, m)')
    ax.set_ylabel('X (North, m)')
    ax.set_zlabel('Z (Elevation, m)')
    ax.set_title('True Coordinates of Instrument and Points in Room')
    ax.legend()
    ax.grid(True)
    ax.set_box_aspect([room_length, room_width, room_height])
    plt.show()

def calculate_true_measurements(instrument_coords, true_points_df):
    """
    Розраховує "істинні" горизонтальні кути (дирекційні кути),
    зенітні відстані та віддалі від приладу до кожної точки.
    """
    X_inst, Y_inst, Z_inst = instrument_coords
    points_with_measurements = true_points_df.copy()

    delta_X = points_with_measurements['X'] - X_inst
    delta_Y = points_with_measurements['Y'] - Y_inst
    delta_Z = points_with_measurements['Z'] - Z_inst

    horizontal_distance = np.sqrt(delta_X**2 + delta_Y**2)
    points_with_measurements['true_distance_m'] = np.sqrt(horizontal_distance**2 + delta_Z**2)

    true_horizontal_angle = np.arctan2(delta_Y, delta_X)
    points_with_measurements['true_horizontal_angle_rad'] = np.where(
        true_horizontal_angle < 0,
        true_horizontal_angle + 2 * np.pi,
        true_horizontal_angle
    )

    # Забезпечуємо, що horizontal_distance не дорівнює нулю для arctan2
    # Це необхідно, якщо точка знаходиться безпосередньо над/під приладом
    epsilon_dist = 1e-9
    safe_horizontal_distance = np.maximum(horizontal_distance, epsilon_dist)

    points_with_measurements['true_zenith_angle_rad'] = np.arctan2(
        safe_horizontal_distance,
        delta_Z
    )

    return points_with_measurements

# --- 2. Функції для моделювання систематичних похибок ---

def arcsec_to_rad(arcsec):
    return arcsec / 3600 * np.pi / 180

def generate_collimation_error(num_samples):
    return arcsec_to_rad(np.random.uniform(12, 18, num_samples))

def generate_zero_point_error(num_samples):
    return arcsec_to_rad(np.random.uniform(-18, -12, num_samples))

def generate_additive_distance_error(num_samples):
    return np.random.uniform(2, 3, num_samples) / 1000.0

def generate_multiplicative_distance_error(num_samples):
    return np.random.uniform(-3, 3, num_samples)

def generate_tilt_horizontal_axis_error(num_samples):
    return arcsec_to_rad(np.random.uniform(25, 30, num_samples))

def generate_angular_eccentricity_param():
    return arcsec_to_rad(np.random.uniform(15, 18))

def apply_systematic_errors(true_measurements_df):
    """
    Застосовуємо змодельовані систематичні похибки до "істинних" вимірювань.
    """
    num_samples = len(true_measurements_df)

    collimation_error = generate_collimation_error(num_samples)
    zero_point_error = generate_zero_point_error(num_samples)
    additive_dist_error = generate_additive_distance_error(num_samples)
    multiplicative_dist_error_ppm = generate_multiplicative_distance_error(num_samples)
    tilt_horiz_axis_error = generate_tilt_horizontal_axis_error(num_samples)

    e_max_angular = generate_angular_eccentricity_param()
    angular_eccentricity_effect = e_max_angular * np.sin(true_measurements_df['true_horizontal_angle_rad'])

    df_with_errors = true_measurements_df.copy()
    df_with_errors['collimation_error_rad'] = collimation_error
    df_with_errors['zero_point_error_rad'] = zero_point_error
    df_with_errors['additive_distance_error_m'] = additive_dist_error
    df_with_errors['multiplicative_distance_error_ppm'] = multiplicative_dist_error_ppm
    df_with_errors['tilt_horizontal_axis_error_rad'] = tilt_horiz_axis_error
    df_with_errors['angular_eccentricity_param_rad'] = e_max_angular
    df_with_errors['angular_eccentricity_effect_rad'] = angular_eccentricity_effect

    epsilon = 1e-6
    safe_zenith_angle = np.clip(df_with_errors['true_zenith_angle_rad'], epsilon, np.pi - epsilon)
    horizontal_angle_error_from_tilt_axis = tilt_horiz_axis_error / np.tan(safe_zenith_angle)

    df_with_errors['measured_horizontal_angle_rad'] = (
    df_with_errors['true_horizontal_angle_rad'] +
    collimation_error / np.sin(df_with_errors['true_zenith_angle_rad']) +
    horizontal_angle_error_from_tilt_axis +
    angular_eccentricity_effect
)

    distance_error_additive = additive_dist_error
    distance_error_multiplicative = df_with_errors['true_distance_m'] * (multiplicative_dist_error_ppm / 1_000_000.0)

    df_with_errors['measured_distance_m'] = (
        df_with_errors['true_distance_m'] +
        distance_error_additive +
        distance_error_multiplicative
    )

    df_with_errors['measured_zenith_angle_rad'] = (
        df_with_errors['true_zenith_angle_rad'] +
        zero_point_error
    )

    # Нормалізація кутів
    df_with_errors['measured_horizontal_angle_rad'] = np.where(
        df_with_errors['measured_horizontal_angle_rad'] < 0,
        df_with_errors['measured_horizontal_angle_rad'] + 2 * np.pi,
        df_with_errors['measured_horizontal_angle_rad']
    )
    df_with_errors['measured_horizontal_angle_rad'] = np.where(
        df_with_errors['measured_horizontal_angle_rad'] >= 2 * np.pi,
        df_with_errors['measured_horizontal_angle_rad'] - 2 * np.pi,
        df_with_errors['measured_horizontal_angle_rad']
    )
    df_with_errors['measured_zenith_angle_rad'] = np.clip(
        df_with_errors['measured_zenith_angle_rad'], 0, np.pi
    )

    return df_with_errors

# --- 3. Функція для розрахунку "виміряних" координат ---

def calculate_measured_coordinates(instrument_coords, df_with_errors):
    """
    Розраховує "виміряні" координати точок на основі "виміряних"
    кутів та відстаней від приладу.
    """
    X_inst, Y_inst, Z_inst = instrument_coords
    df_measured_coords = df_with_errors.copy()

    measured_horizontal_projection = df_measured_coords['measured_distance_m'] * np.sin(df_measured_coords['measured_zenith_angle_rad'])

    delta_X_measured = measured_horizontal_projection * np.cos(df_measured_coords['measured_horizontal_angle_rad'])
    delta_Y_measured = measured_horizontal_projection * np.sin(df_measured_coords['measured_horizontal_angle_rad'])
    delta_Z_measured = df_measured_coords['measured_distance_m'] * np.cos(df_measured_coords['measured_zenith_angle_rad'])

    df_measured_coords['measured_X'] = X_inst + delta_X_measured
    df_measured_coords['measured_Y'] = Y_inst + delta_Y_measured
    df_measured_coords['measured_Z'] = Z_inst + delta_Z_measured

    return df_measured_coords

# --- 4. Функції для аналізу та візуалізації похибок ---

def analyze_and_visualize_errors(df_final):
    """
    Проводить аналіз та візуалізацію похибок в планових (X, Y)
    та висотних (Z) координатах.
    """
    print("\n--- Аналіз загальних похибок ---")

    # 1. Загальна статистика похибок в координатах
    print("\nОписова статистика похибок в координатах (measured - true):")
    print(df_final[['error_X', 'error_Y', 'error_Z']].describe())

    # 2. Обчислення RMSE
    rmse_x = np.sqrt(np.mean(df_final['error_X']**2))
    rmse_y = np.sqrt(np.mean(df_final['error_Y']**2))
    rmse_z = np.sqrt(np.mean(df_final['error_Z']**2))
    rmse_xy = np.sqrt(rmse_x**2 + rmse_y**2)
    rmse_3d = np.sqrt(rmse_x**2 + rmse_y**2 + rmse_z**2)

    print(f"\nRMSE по X: {rmse_x:.4f} м")
    print(f"RMSE по Y: {rmse_y:.4f} м")
    print(f"RMSE планових координат (XY): {rmse_xy:.4f} м")
    print(f"RMSE по Z: {rmse_z:.4f} м")
    print(f"RMSE 3D координат: {rmse_3d:.4f} м")


    # --- Візуалізація планових похибок (X, Y) ---
    plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    sns.scatterplot(x='error_X', y='error_Y', data=df_final, alpha=0.5, s=10)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Distribution of planar errors (Error_Y vs Error_X)')
    plt.xlabel('Error in X (m)')
    plt.ylabel('Error in Y (m)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal') # Щоб осі були в однаковому масштабі

    plt.subplot(1, 2, 2)
    # 2D KDE plot
    sns.kdeplot(x='error_X', y='error_Y', data=df_final, fill=True, cmap='viridis', levels=10)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Planar error density distribution')
    plt.xlabel('Error in X (m)')
    plt.ylabel('Error in Y (m)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # Гістограми розподілу окремих компонент планових похибок
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_final['error_X'], kde=True, bins=50, color='skyblue')
    plt.title('Distribution of error in X')
    plt.xlabel('Error in X (m)')
    plt.ylabel('Number of points')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.histplot(df_final['error_Y'], kde=True, bins=50, color='lightcoral')
    plt.title('Distribution of error in Y')
    plt.xlabel('Error in Y (m)')
    plt.ylabel('Number of points')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # --- Візуалізація висотних похибок (Z) ---
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df_final['error_Z'], kde=True, bins=50, color='lightgreen')
    plt.title('Distribution of error in Z')
    plt.xlabel('Error in Z (m)')
    plt.ylabel('Number of points')
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(1, 2, 2)
    sns.scatterplot(x='true_distance_m', y='error_Z', data=df_final, alpha=0.5, s=10)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Dependence of Z error on distance')
    plt.xlabel('True distance (m)')
    plt.ylabel('Error in Z (m)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Додаткова візуалізація: похибка Z від істинного Z
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Z', y='error_Z', data=df_final, alpha=0.5, s=10)
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
    plt.title('Dependence of Z error on true Z')
    plt.xlabel('True Z (m)')
    plt.ylabel('Error in Z (m)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()


def analyze_individual_error_impact(df_final):
    """
    Проводить аналіз впливу кожної систематичної похибки на координати X, Y, Z
    шляхом побудови графіків залежності та кореляційних матриць.
    """
    print("\n--- Аналіз впливу окремих систематичних похибок на координати ---")

    # Словник для зручного відображення назв стовпців на графіках
    error_component_labels = {
        'collimation_error_rad': 'Collimation error (arcsec)',
        'zero_point_error_rad': 'Zero point error (arcsec)',
        'additive_distance_error_m': 'Additive distance error (mm)',
        'multiplicative_distance_error_ppm': 'Multiplicative distance error (ppm)',
        'tilt_horizontal_axis_error_rad': 'Tilt of horizontal axis (arcsec)',
        'angular_eccentricity_effect_rad': 'Angular eccentricity effect (arcsec)'
    }

    # Масштабування для відображення на графіках (радіани -> кут. сек, метри -> мм)
    df_plot = df_final.copy()
    # Конвертація радіанів в кутові секунди
    df_plot['collimation_error_rad'] = df_plot['collimation_error_rad'] * 3600 * 180 / np.pi
    df_plot['zero_point_error_rad'] = df_plot['zero_point_error_rad'] * 3600 * 180 / np.pi
    df_plot['tilt_horizontal_axis_error_rad'] = df_plot['tilt_horizontal_axis_error_rad'] * 3600 * 180 / np.pi
    df_plot['angular_eccentricity_effect_rad'] = df_plot['angular_eccentricity_effect_rad'] * 3600 * 180 / np.pi
    # Конвертація метрів в міліметри
    df_plot['additive_distance_error_m'] = df_plot['additive_distance_error_m'] * 1000


    coordinate_errors = ['error_X', 'error_Y', 'error_Z']
    systematic_errors = [
        'collimation_error_rad',
        'zero_point_error_rad',
        'additive_distance_error_m',
        'multiplicative_distance_error_ppm',
        'tilt_horizontal_axis_error_rad',
        'angular_eccentricity_effect_rad'
    ]

    # --- Побудова графіків залежності ---
    print("\nПобудова графіків залежності похибок координат від систематичних похибок...")
    for sys_err_col in systematic_errors:
        plt.figure(figsize=(18, 5))
        for i, coord_err_col in enumerate(coordinate_errors):
            plt.subplot(1, 3, i + 1)
            sns.scatterplot(x=sys_err_col, y=coord_err_col, data=df_plot, alpha=0.5, s=10)
            plt.title(f'{error_component_labels[sys_err_col]} vs {coord_err_col}')
            plt.xlabel(error_component_labels[sys_err_col])
            plt.ylabel(f'{coord_err_col} (м)')
            plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.show()

    # --- Створення кореляційних матриць ---
    print("\nСтворення кореляційних матриць...")

    # Об'єднуємо всі релевантні стовпці для кореляції
    correlation_columns = systematic_errors + coordinate_errors
    df_correlation = df_plot[correlation_columns] # Використовуємо df_plot для перетворених одиниць

    # Перейменування стовпців для кореляційної матриці для кращої читабельності
    correlation_labels = {
        'collimation_error_rad': 'Collimation (arcsec)',
        'zero_point_error_rad': 'Zero point (arcsec)',
        'additive_distance_error_m': 'Additive D (mm)',
        'multiplicative_distance_error_ppm': 'Multiplicative D (ppm)',
        'tilt_horizontal_axis_error_rad': 'Tilt of axis (arcsec)',
        'angular_eccentricity_effect_rad': 'Eccentricity effect (arcsec)',
        'error_X': 'Error in X (m)',
        'error_Y': 'Error in Y (m)',
        'error_Z': 'Error in Z (m)'
    }
    df_correlation_renamed = df_correlation.rename(columns=correlation_labels)

    # Загальна кореляційна матриця
    plt.figure(figsize=(14, 12))
    sns.heatmap(df_correlation_renamed.corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation matrix: Systematic errors vs Coordinate errors', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Окремі кореляційні матриці для планових (X,Y) та висотних (Z)
    print("\nКореляційна матриця для планових похибок та їх джерел:")
    # zero_point_error_rad зазвичай не впливає на планові координати, але ми спробуємо
    planar_sources = [
        'collimation_error_rad',
        'additive_distance_error_m',
        'multiplicative_distance_error_ppm',
        'tilt_horizontal_axis_error_rad',
        'angular_eccentricity_effect_rad',
        'zero_point_error_rad'
    ]
    planar_corr_cols_renamed = [correlation_labels[col] for col in planar_sources] + ['Error in X (m)', 'Error in Y (m)']

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_correlation_renamed[planar_corr_cols_renamed].corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation: Systematic errors (plan) vs Errors X, Y', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\nКореляційна матриця для висотних похибок та їх джерел:")
    # collimation_error_rad, angular_eccentricity_effect_rad зазвичай не впливають на висоту
    vertical_sources = [
        'zero_point_error_rad',
        'additive_distance_error_m',
        'multiplicative_distance_error_ppm',
        'tilt_horizontal_axis_error_rad' # Нахил осі може впливати на Z через тангенс зенітного кута
    ]
    vertical_corr_cols_renamed = [correlation_labels[col] for col in vertical_sources] + ['Error in Z (m)']

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_correlation_renamed[vertical_corr_cols_renamed].corr(method='pearson'), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation: Systematic errors (vertical) vs Error Z', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


# --- Основний блок виконання ---
if __name__ == "__main__":
    room_length = 30    # Довжина кімнати (м)
    room_width = 50     # Ширина кімнати (м)
    room_height = 20    # Висота кімнати (м)

    instrument_coords, true_points_df = generate_true_coordinates(room_length, room_width, room_height)
    true_measurements_df = calculate_true_measurements(instrument_coords, true_points_df)

    print("Координати приладу (X, Y, Z):", instrument_coords)
    print("\nКількість згенерованих точок:", len(true_points_df))

    df_with_systematic_errors = apply_systematic_errors(true_measurements_df)
    df_final = calculate_measured_coordinates(instrument_coords, df_with_systematic_errors)

    df_final['error_X'] = df_final['measured_X'] - df_final['X']
    df_final['error_Y'] = df_final['measured_Y'] - df_final['Y']
    df_final['error_Z'] = df_final['measured_Z'] - df_final['Z']

    print("\nПерші 5 рядків DataFrame з істинними та виміряними координатами:")
    print(df_final[[
        'X', 'Y', 'Z',
        'measured_X', 'measured_Y', 'measured_Z',
        'error_X', 'error_Y', 'error_Z'
    ]].head())

    # Аналіз загальних похибок
    analyze_and_visualize_errors(df_final)

    # Аналіз впливу індивідуальних систематичних похибок
    analyze_individual_error_impact(df_final)

    # Опціонально: візуалізація кімнати та точок
    visualize_room_and_points(instrument_coords, true_points_df, room_length, room_width, room_height)