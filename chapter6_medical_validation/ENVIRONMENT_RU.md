# Изолированное окружение главы 6

## Исходное состояние перед изоляцией

- Дата проверки: 2026-08-28.
- Системный Python: 3.14.7.
- Project venv: Python 3.14.7, torch 2.11.0+cu128, torchvision 0.26.0.
- Ошибка project venv: `RuntimeError: operator torchvision::nms does not exist`.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB.
- Driver: 610.57.04; CUDA UMD reported by `nvidia-smi`: 13.3.
- Свободно на `/home`: около 259 GiB.

## Зарегистрированное решение

Отдельный untracked overlay-venv (`$CH6_OVERLAY_VENV`), Python 3.14.7. Он повторно
использует уже установленный в project venv `torch 2.11.0+cu128`, но содержит
исправленный официальный wheel `torchvision 0.26.0+cu128`. Полная повторная
установка CUDA wheels в Python 3.12 была прекращена после исчерпания квоты
пакетного cache; системный/project venv при этом не изменялся.

Фактический smoke:

- CUDA доступна, runtime torch сообщает CUDA 12.8;
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU;
- `torchvision.ops.nms` выполняется и возвращает индексы `[0, 2]`;
- VGG16 и InceptionV3 выполняют CUDA forward до `(1, 1000)`.

Полный freeze формируется в финальном validation bundle без включения самого
venv.
