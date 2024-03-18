# Структура репозитория
```
.
├── README.md
├── configs
│   ├── cmip6_GhostWindNet27.yaml
│   ├── eval
│   ├── process
│   ├── raw
│   ├── train
│   ├── unit_test_configs
│   └── visual
├── data
│   ├── 00-raw
│   ├── 01-prepared
│   ├── 02-models
│   ├── 03-results
│   └── 04-feed
├── demo-corrector.ipynb
├── drun.sh
├── environments
│   └── poetry
├── out
│   └── visual_eobs27
├── preprocess.py
├── requirements.txt
├── run.py
├── slurm_run.sh
├── src
│   ├── data_assemble
│   ├── regression
│   └── utils
├── test
│   ├── dummy_data.py
│   ├── test_climate_padding.py
│   ├── test_dataload.py
│   ├── test_prepapre_cmip5.py
│   └── test_target.py
├── test.py
├── train.py
└── utils_corrector.py
```

# Конфигурационные файлы
Конфигурационные файлы обрабатываются с помощью пакета [hydra](https://hydra.cc/docs/intro/). Они находятся в папке `configs`.
Папка `configs` содержит **корневой** конфигурационный файл, описывающий логику модели, подготовки и обучения. Предполагается, что пользователь будет задавать всю логику через **корнейвой** файл, а именно: выбор модели, исходных сырых данных, этапы предобработки, обучения и предсказания. Пример файла -  `configs/cmip6_WindNet27x47.yaml`. **Корневой конфигурационный файл** содержит атрибуты `time_start` и `time_end`, которые используются только на этапе предсказания.
В следующих подсекциях будут описаны основные этапы.
## Модель
* `model_name` -- архитектура из запрограммированных моделей в `src/regression/models/models.py`
* `time_window` -- рецептивное поле по времени
* `half_side_size` -- рецептивное поле по пространству
## Подготовка данных 
Данный этап обращается к подфайлам в `configs/raw` и `configs/process`. Конфигурационные файлы из первой папки содержат пути к сырым данным (климатические проекции CMIP и измерения с метеостанций). Файлы из второй задают логику подготовки данных. О ней можно прочитать в соответсвутющей секции `README.md` ниже. Кратко,
* `defaults.raw` -- пути к сырым данным: CMIP, ЦМР, **сырые и распаршенные** данные метеостанций (см. отчет о разработке модели - парсер)
* `defaults.process` -- логика подготовки. См. комментарии к атрибутам в `configs/process/cmip6_elevation_dataset.yaml`.

## Обучение
Данный этап обращается к конфигурационным файлам из папки `configs/train`. Файлы описывают логику обучения и формирования датамодуля, задается в атрибуте `defaults.train`.

# Подготовка данных:
Данный этап необходимо осуществить **и перед обучением (`train.py`), и перед предсказанием (`run.py`).** Конфигурационные файлы, которые используются в скрипте `preprocess.py`, включаются себя файлы из `configs/raw` и `configs/process`. Конкретные конфигурационные файлы задаются в атрибутах **корневого** конфигурационного файла, например, `configs/cmip6_WindNet27x47.yaml`.

По итогу предобработки данных, будет создана директория по пути, указанном в атрибуте `cfg.process.data_dir`, куда будут сохранены предобработанные данные. Среди них: координатные оси, нормализованные признаки, параметры нормализации и предобработанная целевая переменная. Во время обучения и предсказания будут использоваться именно эти данные.
Общий пример запуска подготовки данных из консоли:
```
python preprocess.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Конкретные пример запуска подготовки данных из консоли:
```
python preprocess.py --config-path /app/wind/configs --config-name cmip6_WindNet27x47.yaml
```
# Обучение:
**Не забудьте запустить `preprocess.py`, чтобы подготовить данные!**
Конфигурационный файл обучения задает технические параметры: оптимизатор, размер батча, функция потерь, используемые видеокарты и т.п. Конкретная конфигурация задается из **корневого** конфигурационного фалйа, см. Секцию Конфигурационные файлы.
Общий пример запуска обучения:
```
python train.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME>
```
Конкретный пример запуска обучения:
```
python train.py --config-path configs --config-name cmip6_WindNet27x47.yaml
```

# Предсказание:
**Не забудьте подготовить данные с помощью `preprocess.py`!**
Параметры предсказания задаются в конфигурационных файлах из папки `configs/eval`, путь к конкретному файлу необходимо указать в атрибуте `defaults.eval` корневого конфигурационного файла. Пример: `configs/cmip6_WindNet27x47.yaml`
В файлах `configs/eval` можно указать пороговое значение скорости ветра, которое определяет риск, путь к чекпоинту обученной модели. Область задается в конфигурационном файле из папки `eval`, атрибуты `eval.lat_min`, `eval.lat_max`, `eval.lon_min`, `eval.lon_max` Временные рамки можно передать в командной строке, как в примере ниже:
```
python run.py --config-path <PATH TO FOLDER WITH CONFIGS> --config-name <CONFIG NAME> time_start='YYYY-MM-DD' time_end='YYYY-MM-DD'
```
Пример:
```
python run.py --config-path configs/ --config-name cmip6_WindNet27x47.yaml time_start='2019-01-30' time_end='2019-01-31'
```

# Docker:

Для сборки образа, выполнить в командной строке:

```
docker build --build-arg DOCKER_USER_ID=`id -u` --build-arg DOCKER_GROUP_ID=`id -g` -t wind_dev environments/poetry 

export WANDB_API_KEY=<key>

docker run -it \
   -v $(pwd):/app/wind \
   -v <DATA FOLDER>:/app/wind/data \
   -m 256000m --cpus=16 --gpus '"device=0,1"' \
   --ipc=host \
   -w="/app/wind" \
   -e "WANDB_API_KEY=$WANDB_API_KEY" \
   -e "WANDB_DATA_DIR=/app/wind/out" \
   -e "WANDB_DIR=/app/wind/out" \
   -e "WANDB_CACHE_DIR=/app/wind/out" \
   wind_dev

```
Пример:


```
   docker run -it \
   -v $(pwd):/app/wind \
   -v /mnt/data/lukashevich/:/app/wind/data \
   -m 256000m --cpus=16 --gpus '"device=0,1"' \
   --ipc=host \
   -w="/app/wind" \
   -e "WANDB_API_KEY=$WANDB_API_KEY" \
   -e "WANDB_DATA_DIR=/app/wind/out" \
   -e "WANDB_DIR=/app/wind/out" \
   -e "WANDB_CACHE_DIR=/app/wind/out" \
   wind_dev

```