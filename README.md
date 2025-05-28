# Классификация рентгеновских снимков грудной клетки с помощью CNN

![Пример работы](https://imgur.com/a/kGFim1H)

## 📝 Описание проекта
Модель глубокого обучения для автоматической диагностики пневмонии по рентгеновским снимкам грудной клетки. Проект использует предобученную CNN (EfficientNetB0) с точностью **92-95%** на тестовых данных.

## 📊 Результаты
| Метрика       | Значение |
|--------------|----------|
| Accuracy     | 92.6%    |
| Precision    | 91.8%    |
| Recall       | 95.6%    |
| F1-Score     | 93.7%    |

Confusion Matrix:  
![Confusion Matrix](https://imgur.com/wzHOCxR)

## 🛠 Технологии
- Python 3.8+
- TensorFlow 2.x
- EfficientNetB0 (трансферное обучение)
- Gradio (веб-интерфейс)
- Scikit-learn (метрики)

## 📁 Структура проекта
- ├── train/
- │ ├── NORMAL/
- │ └── PNEUMONIA/
- ├── val/
- │ ├── NORMAL/
- │ └── PNEUMONIA/
- └── test/
- ├── NORMAL/
- └── PNEUMONIA/


## 🚀 Запуск
1. Установите зависимости:
```bash
pip install -r requirements.txt
python main.py
```
2. Запуск веб-интерфейса (после обучения):
```bash
iface.launch()
```

📌 Особенности
Аугментация данных (повороты, сдвиги, зеркалирование)

Fine-tuning предобученной EfficientNet

Визуализация активаций слоёв

Графики обучения (accuracy/loss)
