# FET-PET/MRI Tyrosine Kinetics Analyzer

Автоматический пайплайн анализа кинетики накопления FET (фторэтилтирозина) по данным ПЭТ/МРТ.

Поддерживает два режима:
- **Static** — 3 статические серии (20, 40, 60 мин)
- **Dynamic** — 3 динамические 4D-серии с переменным временем экспозиции (до 38 временных точек)

---

## Режимы работы

### Static (статика)

Три отдельные статические ПЭТ-серии, каждая в своей папке DICOM. Классический протокол с точками 20, 40, 60 минут.

### Dynamic (динамика)

Три динамические DICOM-серии (4D), каждая содержит несколько объёмов (фреймов) с разным временем экспозиции. Пайплайн автоматически конкатенирует их в одну 4D-серию и строит кривые TBR(t) по всем временным точкам.

Расписание временных точек (38 фреймов суммарно):

| Серия | Фреймов | Экспозиция | Время (сек) | Центры (сек) |
|-------|---------|------------|-------------|--------------|
| 1     | 12      | 5 с        | 0–60        | 2.5, 7.5, ..., 57.5 |
| 1     | 6       | 10 с       | 60–120      | 65, 75, ..., 115 |
| 1     | 3       | 20 с       | 120–180     | 130, 150, 170 |
| 1     | 5       | 60 с       | 180–480     | 210, 270, 330, 390, 450 |
| 1     | 4       | 180 с      | 480–1200    | 570, 750, 930, 1110 |
| 2     | 4       | 300 с      | 1200–2400   | 1350, 1650, 1950, 2250 |
| 3     | 4       | 300 с      | 2400–3600   | 2550, 2850, 3150, 3450 |

---

## Установка

### Создание venv и установка зависимостей (Windows)

```bash
# 1. Перейти в папку проекта
cd D:\nadelyaev\fet

# 2. Создать виртуальное окружение
python -m venv .venv

# 3. Активировать venv
.venv\Scripts\activate

# 4. Обновить pip
python -m pip install --upgrade pip

# 5. Установить зависимости
pip install pydicom numpy nibabel scipy dcm2niix antspyx antspynet

# 6. Проверить что всё установлено
pip list
```

### Создание venv и установка зависимостей (Linux)

```bash
# 1. Перейти в папку проекта
cd /path/to/fet

# 2. Создать виртуальное окружение
python3 -m venv .venv

# 3. Активировать venv
source .venv/bin/activate

# 4. Обновить pip
python -m pip install --upgrade pip

# 5. Установить зависимости
pip install pydicom numpy nibabel scipy dcm2niix antspyx antspynet

# 6. Проверить что всё установлено
pip list
```

### Зависимости

| Пакет | Назначение |
|-------|------------|
| `pydicom` | Чтение DICOM-файлов (метаданные, теги) |
| `numpy` | Работа с массивами, линейная алгебра |
| `nibabel` | Чтение/запись NIfTI-файлов |
| `scipy` | Гауссово сглаживание, статистика |
| `dcm2niix` | Конвертация DICOM → NIfTI (Python-обёртка + бинарник) |
| `antspyx` | ANTs регистрация, resample, работа с 4D |
| `antspynet` | Нейросетевой skull-stripping (brain extraction) |

### Активация venv перед каждым запуском

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

---

## Запуск

### Static (базовый, без T1, skull-stripping по PET)

```bash
python run_pipeline.py \
    --t20 <папка_20мин> \
    --t40 <папка_40мин> \
    --t60 <папка_60мин> \
    --output <папка_результата>
```

### Static с T1 (antspynet skull-stripping)

```bash
python run_pipeline.py \
    --t20 <папка_20мин> \
    --t40 <папка_40мин> \
    --t60 <папка_60мин> \
    --t1 <папка_T1> \
    --output <папка_результата>
```

### Dynamic (базовый)

```bash
python run_pipeline.py --dynamic \
    --dyn1 <папка_серия1> \
    --dyn2 <папка_серия2> \
    --dyn3 <папка_серия3> \
    --output <папка_результата>
```

### Dynamic с T1

```bash
python run_pipeline.py --dynamic \
    --dyn1 <папка_серия1> \
    --dyn2 <папка_серия2> \
    --dyn3 <папка_серия3> \
    --t1 <папка_T1> \
    --output <папка_результата>
```

---

## Все параметры

```
Обязательные:
  --t20 PATH                DICOM-папка статики 20 мин (static mode)
  --t40 PATH                DICOM-папка статики 40 мин (static mode)
  --t60 PATH                DICOM-папка статики 60 мин (static mode)
  --dyn1 PATH               DICOM-папка динамики серия 1 (dynamic mode)
  --dyn2 PATH               DICOM-папка динамики серия 2 (dynamic mode)
  --dyn3 PATH               DICOM-папка динамики серия 3 (dynamic mode)
  --output, -o PATH         Папка для результатов

Режим:
  --dynamic                 Ипользовать динамический режим (3 серии вместо 3 точек)

Необязательные (вход):
  --t1 PATH                 DICOM-папка T1 (для skull-stripping)

Пороги:
  --sul-threshold FLOAT     Порог SULmax для значимой маски (default: 2.0)
  --tbr-threshold FLOAT     Порог TBR для значимой маски (default: 2.0)
  --tbr-delta-threshold FLOAT  Минимальное изменение TBR за время измерения,
                              чтобы считать тренд (default: 0.3)
  --time-span FLOAT         Интервал первая→последняя точка в мин (default: 40.0)
                             (только для static mode)
  --time-points F F F       Три точки времени в мин (default: 20 40 60)
                             (только для static mode)
  --trim-percent FLOAT      % обрезки хвостов для SULmean (default: 2.5)

Препроцессинг:
  --no-skull-strip          Отключить skull-stripping
  --no-smoothing            Отключить сглаживание
  --smooth-sigma FLOAT      Сигма Гауссова сглаживания в вокселях (default: 1.0)
```

---

## Примеры

```bash
# === STATIC ===

# Минимальный запуск
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 -o TAV/output

# С T1 для точного skull-stripping
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 --t1 TAV/T1 -o TAV/output

# Без сглаживания (сохраняет исходные пиковые значения)
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 --no-smoothing -o TAV/output

# Без skull-stripping и без сглаживания (сырые данные)
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 --no-skull-strip --no-smoothing -o TAV/output

# Другие точки времени (15, 35, 55 мин)
python run_pipeline.py --t20 P/15 --t40 P/35 --t60 P/55 --time-points 15 35 55 --time-span 40 -o P/output

# Свои пороги: SULmax>1.5, TBR>1.5, delta>0.2
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 \
    --sul-threshold 1.5 --tbr-threshold 1.5 --tbr-delta-threshold 0.2 -o TAV/output

# Более сильное сглаживание (sigma=2)
python run_pipeline.py --t20 TAV/20 --t40 TAV/40 --t60 TAV/60 --smooth-sigma 2.0 -o TAV/output

# === DYNAMIC ===

# Базовый запуск
python run_pipeline.py --dynamic \
    --dyn1 DYN/ser1 --dyn2 DYN/ser2 --dyn3 DYN/ser3 \
    -o DYN/output

# С T1
python run_pipeline.py --dynamic \
    --dyn1 DYN/ser1 --dyn2 DYN/ser2 --dyn3 DYN/ser3 \
    --t1 DYN/T1 \
    -o DYN/output

# Без сглаживания
python run_pipeline.py --dynamic \
    --dyn1 DYN/ser1 --dyn2 DYN/ser2 --dyn3 DYN/ser3 \
    --no-smoothing -o DYN/output

# Свои пороги
python run_pipeline.py --dynamic \
    --dyn1 DYN/ser1 --dyn2 DYN/ser2 --dyn3 DYN/ser3 \
    --tbr-delta-threshold 0.2 -o DYN/output
```

---

## Выходные файлы

### Static mode

| Файл | Тип | Описание |
|------|-----|----------|
| `mask_clusters.nii.gz` | int8 | Кластеры: **1**=rising, **2**=falling, **3**=plateau. Вне маски=0 |
| `map_slope.nii.gz` | float32 | Наклон кривой TBR (TBR/мин). **Только в значимой маске**, вне неё=0 |
| `map_sulmax.nii.gz` | float32 | Max SUL по трём точкам времени |
| `map_tbrmax.nii.gz` | float32 | Max TBR по трём точкам времени |
| `mask_brain.nii.gz` | uint8 | Маска мозга (skull-stripping) |
| `mask_sulmax_gt2.nii.gz` | uint8 | SULmax > порог |
| `mask_tbr_gt2.nii.gz` | uint8 | TBR > порог хотя бы в одной точке (объединение) |
| `mask_tbr_gt2_t20.nii.gz` | uint8 | TBR > порог на 20 мин |
| `mask_tbr_gt2_t40.nii.gz` | uint8 | TBR > порог на 40 мин |
| `mask_tbr_gt2_t60.nii.gz` | uint8 | TBR > порог на 60 мин |
| `map_sul_t20.nii.gz` | float32 | Карта SUL на 20 мин |
| `map_sul_t40.nii.gz` | float32 | Карта SUL на 40 мин |
| `map_sul_t60.nii.gz` | float32 | Карта SUL на 60 мин |
| `map_tbr_t20.nii.gz` | float32 | Карта TBR на 20 мин |
| `map_tbr_t40.nii.gz` | float32 | Карта TBR на 40 мин |
| `map_tbr_t60.nii.gz` | float32 | Карта TBR на 60 мин |
| `report.json` | JSON | Параметры, SULmean, 95% ДИ, размеры кластеров, mean slope |

### Dynamic mode

| Файл | Тип | Описание |
|------|-----|----------|
| `mask_clusters.nii.gz` | int8 | Кластеры: **1**=rising, **2**=falling, **3**=plateau. Вне маски=0 |
| `map_slope.nii.gz` | float32 | Наклон кривой TBR (TBR/мин). **Только в значимой маске**, вне неё=0 |
| `mask_brain.nii.gz` | uint8 | Маска мозга (skull-stripping) |
| `report.json` | JSON | Параметры, SULmean по фреймам, временные точки, размеры кластеров, mean slope |

---

## Логика классификации

### Static

- Для каждого вокселя: slope = линейная регрессия TBR(t) по трём точкам
- `tbr_delta_threshold` (default 0.3) — минимальное изменение TBR за `time_span` (default 40 мин), чтобы считать тренд
- slope_threshold = tbr_delta_threshold / time_span = 0.3 / 40 = 0.0075 TBR/мин
- **Rising**: slope > slope_threshold
- **Falling**: slope < -slope_threshold
- **Plateau**: |slope| <= slope_threshold
- **Значимая маска**: SULmax > sul_threshold **И** хотя бы в одной точке TBR > tbr_threshold

### Dynamic

- Для каждого вокселя: slope = линейная регрессия TBR(t) по **всем** временным точкам (до 38)
- slope_threshold = tbr_delta_threshold / time_span_min, где time_span_min — общий диапазон (default ~57.5 мин для 3600 с)
- Классификация та же: rising / falling / plateau
- **Значимая маска**: mean(SUL) > sul_threshold **И** mean(TBR) > tbr_threshold (среднее по времени)

---

## SUL (SUVlbm) — формула James

```
SUL = activity[Bq/ml] × LBM[g] / InjectedDose[Bq]

Male:   LBM = 1.10 × W − 128 × (W / H_cm)²
Female: LBM = 1.07 × W − 148 × (W / H_cm)²

W = вес (кг), H_cm = рост (см)
```

---

## Skull-stripping

### Static mode
- **С T1**: antspynet нейросетевой brain extraction (глубокая сеть, высокая точность)
- **Без T1**: пороговый метод по 20-мин статике PET (фоллбэк)
- Если antspynet не сработал — автоматический фоллбэк на пороговый метод по T1
- T1 с другим resolution автоматом ресэмплируется в пространство PET через ANTs

### Dynamic mode
- **С T1**: antspynet на T1 (ресэмплированном в пространство PET)
- **Без T1**: пороговый метод по временному среднему PET (temporal mean всех фреймов)
- Маска применяется ко всем фреймам

---

## Технические детали

- DICOM → NIfTI конвертация через **dcm2niix** (золотой стандарт, правильная ориентация)
- Per-slice RescaleSlope (PET) — применяется корректно для каждого среза
- InjectedDose достаётся из Radiopharmaceutical Information Sequence (0054,0016)
- Trimmed mean: убирает 2.5% smallest + 2.5% largest значений перед усреднением
- Gaussian smoothing: лёгкое (σ=1 voxel default), только внутри brain mask
- Для динамики: сглаживание применяется пространственно (per-frame), не во временной оси

---

## Структура проекта

```
fet_project/
├── run_pipeline.py            # Главный скрипт (точка входа)
├── dicom_reader.py            # Чтение статических DICOM-серий
├── dicom_reader_dynamic.py    # Чтение динамических 4D DICOM-серий
├── preprocess.py              # Препроцессинг (skull-strip + smooth)
├── analysis.py                # Анализ для статики (TBR, slope, классификация)
├── analysis_dynamic.py        # Анализ для динамики (TBR(t), slope по N точкам)
├── output.py                  # Сохранение NIfTI + JSON
├── test_synthetic.py          # Синтетический тест (статика)
└── README.md                  # Этот файл
```

---

## report.json — структура

### Static

```json
{
  "mode": "static",
  "parameters": {
    "sul_threshold": 2.0,
    "tbr_threshold": 2.0,
    "tbr_delta_threshold": 0.3,
    "time_span_min": 40.0,
    "time_points_min": [20.0, 40.0, 60.0],
    "trim_percent": 2.5,
    "skull_strip": true,
    "t1_used": false,
    "smoothing": true,
    "smooth_sigma": 1.0,
    "patient_weight_kg": 70.0,
    "patient_height_cm": 175.0,
    "patient_sex": "M",
    "injected_dose_bq": 300000000.0
  },
  "per_timepoint": {
    "t20": {
      "time_min": 20.0,
      "sulmean": 1.2345,
      "sul_std": 0.1234,
      "ci_lower": 1.2300,
      "ci_upper": 1.2390,
      "n_voxels_tbr_gt_threshold": 1234
    },
    "t40": { ... },
    "t60": { ... }
  },
  "results": {
    "slope_threshold_tbr_per_min": 0.0075,
    "n_significant_voxels": 5000,
    "n_rising": 2000,
    "n_falling": 1500,
    "n_plateau": 1500,
    "n_sulmax_gt2": 8000,
    "n_tbr_gt2": 6000,
    "mean_slope_rising": 0.015,
    "mean_slope_falling": -0.012,
    "mean_slope_plateau": 0.001
  }
}
```

### Dynamic

```json
{
  "mode": "dynamic",
  "parameters": {
    "tbr_delta_threshold": 0.3,
    "time_span_min": 57.5,
    "time_span_sec": 3450.0,
    "trim_percent": 2.5,
    "skull_strip": true,
    "t1_used": false,
    "smoothing": true,
    "smooth_sigma": 1.0,
    "patient_weight_kg": 70.0,
    "patient_height_cm": 175.0,
    "patient_sex": "M",
    "injected_dose_bq": 300000000.0
  },
  "results": {
    "slope_threshold_tbr_per_min": 0.005217,
    "n_significant_voxels": 5000,
    "n_rising": 2000,
    "n_falling": 1500,
    "n_plateau": 1500,
    "mean_slope_rising": 0.015,
    "mean_slope_falling": -0.012,
    "mean_slope_plateau": 0.001
  },
  "sul_means_per_frame": [1.23, 1.25, ...],
  "time_points_min": [0.04, 0.12, ...],
  "n_frames": 38
}
```
