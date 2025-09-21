// Глобальный объект для хранения состояния разметки
const labels = {};

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', function() {
    initializeLabels();
    setupEventListeners();
    loadExistingLabels();
});

// Инициализация состояния для каждого образца
function initializeLabels() {
    const sampleCards = document.querySelectorAll('.sample-card');
    sampleCards.forEach(card => {
        const sampleId = card.id.split('-')[1];
        const index = card.getAttribute('data-index');
        
        labels[sampleId] = {
            index: parseInt(index),
            category: '',
            reason: '',
            real_is_command: false
        };
    });
}

// Настройка обработчиков событий
function setupEventListeners() {
    // Обработчики для радио-кнопок
    document.querySelectorAll('input[type="radio"]').forEach(radio => {
        radio.addEventListener('change', function() {
            const sampleId = this.name.split('_')[1];
            saveLabel(sampleId, this.value);
        });
    });

    // Обработчики для текстовых полей
    document.querySelectorAll('.reason-input').forEach(textarea => {
        textarea.addEventListener('blur', function() {
            const sampleId = this.getAttribute('data-sample-id');
            saveReason(sampleId, this.value);
        });
    });

    // Обработчики для чекбоксов
    document.querySelectorAll('.real-command-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', function() {
            const sampleId = this.getAttribute('data-sample-id');
            saveRealIsCommand(sampleId, this.checked);
        });
    });
}

// Функция для сохранения категории
function saveLabel(sampleId, category) {
    labels[sampleId].category = category;
    updateStatus(sampleId, 'Категория сохранена');
    saveToServer(sampleId);
}

// Функция для сохранения причины
function saveReason(sampleId, reason) {
    labels[sampleId].reason = reason;
    updateStatus(sampleId, 'Причина сохранена');
    saveToServer(sampleId);
}

// Функция для сохранения реального is_command
function saveRealIsCommand(sampleId, realIsCommand) {
    labels[sampleId].real_is_command = realIsCommand;
    updateStatus(sampleId, 'Статус команды сохранен');
    saveToServer(sampleId);
}

// Функция для обновления статуса
function updateStatus(sampleId, message) {
    const statusElement = document.getElementById(`status-${sampleId}`);
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.className = 'status saved';
        
        // Через 2 секунды очищаем статус
        setTimeout(() => {
            statusElement.textContent = '';
            statusElement.className = 'status';
        }, 2000);
    }
}

// Функция для отправки данных на сервер
function saveToServer(sampleId) {
    const data = labels[sampleId];
    
    fetch('/save_label', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            console.log(`Данные для sample ${sampleId} успешно сохранены`);
        }
    })
    .catch((error) => {
        console.error('Ошибка:', error);
        updateStatus(sampleId, 'Ошибка сохранения');
    });
}

// Загрузка существующей разметки
function loadExistingLabels() {
    fetch('/get_labeled_data')
        .then(response => response.json())
        .then(data => {
            data.forEach(item => {
                // Находим соответствующий sample по индексу
                for (const sampleId in labels) {
                    if (labels[sampleId].index === item.index) {
                        // Заполняем форму сохраненными значениями
                        if (item.markup_category) {
                            const radio = document.querySelector(`input[name="category_${sampleId}"][value="${item.markup_category}"]`);
                            if (radio) {
                                radio.checked = true;
                                labels[sampleId].category = item.markup_category;
                            }
                        }
                        
                        if (item.markup_reason) {
                            const textarea = document.querySelector(`.reason-input[data-sample-id="${sampleId}"]`);
                            if (textarea) {
                                textarea.value = item.markup_reason;
                                labels[sampleId].reason = item.markup_reason;
                            }
                        }
                        
                        if (item.real_is_command) {
                            const checkbox = document.querySelector(`.real-command-checkbox[data-sample-id="${sampleId}"]`);
                            if (checkbox) {
                                const isChecked = item.real_is_command === 'true' || item.real_is_command === true;
                                checkbox.checked = isChecked;
                                labels[sampleId].real_is_command = isChecked;
                            }
                        }
                        
                        break;
                    }
                }
            });
        })
        .catch(error => console.error('Ошибка загрузки данных:', error));
}