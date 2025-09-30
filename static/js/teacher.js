// ГЛАВНЫЙ КЛАСС ДЛЯ TEACHER PANEL
class TeacherPanel {
    constructor() {
        this.socket = io();
        this.currentRoom = 'default';
        this.currentAvatar = '';
        this.init();
    }

    init() {
        this.setupSocket();
        this.loadInitialData();
        this.setupEventListeners();
        this.setupTabs();
    }

    setupSocket() {
        this.socket.on('connect', () => {
            this.log('Подключено к серверу');
            this.socket.emit('join_room', { room_id: this.currentRoom });
        });

        this.socket.on('disconnect', () => {
            this.log('Отключено от сервера');
        });
    }

    async loadInitialData() {
        await this.loadApiKeys();
        await this.loadLLMMode();
        await this.loadAvatars();
        await this.loadPracticeFiles();
    }

    // API KEYS MANAGEMENT
    async loadApiKeys() {
        try {
            const response = await fetch('/api/config/keys');
            const data = await response.json();
            if (data.success) {
                document.getElementById('openrouter-key').value = data.keys.openrouter || '';
                this.log('API ключи загружены');
            }
        } catch (error) {
            this.log('Ошибка загрузки API ключей: ' + error);
        }
    }

    async saveApiKey(provider, apiKey) {
        try {
            const response = await fetch('/api/config/keys', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, api_key: apiKey })
            });
            const data = await response.json();
            this.showStatus('openrouter-status', data.message, data.success);
        } catch (error) {
            this.showStatus('openrouter-status', 'Ошибка сети: ' + error, false);
        }
    }

    async testApiKey(provider, apiKey) {
        try {
            const response = await fetch('/api/config/test', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ provider, api_key: apiKey })
            });
            const data = await response.json();
            this.showStatus('openrouter-status', data.message, data.valid);
        } catch (error) {
            this.showStatus('openrouter-status', 'Ошибка проверки ключа: ' + error, false);
        }
    }

    // LLM MODE MANAGEMENT
    async loadLLMMode() {
        try {
            const response = await fetch('/api/config/llm_mode');
            const data = await response.json();
            if (data.success) {
                document.getElementById('llm-mode-select').value = data.mode;
                this.log('Режим LLM загружен: ' + data.mode);
            }
        } catch (error) {
            this.log('Ошибка загрузки режима LLM: ' + error);
        }
    }

    async setLLMMode(mode) {
        try {
            const response = await fetch('/api/config/llm_mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            const data = await response.json();
            if (data.success) {
                this.showStatus('llm-mode-status', data.message, true);
                // Отправляем сокет-запрос для применения режима
                this.socket.emit('set_llm_mode', { 
                    room_id: this.currentRoom,
                    mode: mode
                });
            } else {
                this.showStatus('llm-mode-status', 'Ошибка: ' + data.error, false);
            }
        } catch (error) {
            this.showStatus('llm-mode-status', 'Ошибка сети: ' + error, false);
        }
    }

    // AVATAR MANAGEMENT
    async loadAvatars() {
        try {
            const response = await fetch('/api/avatars');
            const data = await response.json();
            
            if (data.error) throw new Error(data.error);
            
            const select = document.getElementById('avatar-select');
            select.innerHTML = '<option value="">Выберите аватар</option>';
            
            data.avatars.forEach(avatar => {
                const option = document.createElement('option');
                option.value = avatar;
                option.textContent = avatar;
                select.appendChild(option);
            });
            
            this.log('Загружено аватаров: ' + data.avatars.length);
        } catch (error) {
            this.log(error.message);
        }
    }

    loadAvatarFrames(avatarName) {
        this.updateStatus(`Загрузка ${avatarName}...`);
        
        fetch(`/api/frames/${encodeURIComponent(avatarName)}`)
            .then(response => response.json())
            .then(data => {
                if (data.error) throw new Error(data.error);
                this.setAvatar(avatarName, data.frames);
            })
            .catch(error => {
                this.updateStatus('Ошибка: ' + error.message);
                this.log(error.message);
            });
    }

    setAvatar(avatarName, frames) {
        this.socket.emit('avatar_changed', {
            room_id: this.currentRoom,
            avatar_name: avatarName
        });
        
        // Показываем первое изображение для предпросмотра
        if (frames && frames.length > 0) {
            const firstFrame = `/frames/${avatarName}/${frames[0]}`;
            this.loadImage(firstFrame).then(img => {
                document.getElementById('avatar').src = img.src;
            }).catch(error => {
                this.log('Ошибка загрузки изображения: ' + error);
            });
        }
        
        this.log(`Аватар "${avatarName}" установлен для комнаты ${this.currentRoom}`);
        this.updateStatus(`${avatarName}: загружено ${frames ? frames.length : 0} кадров`);
    }

    loadImage(src, retries = 3) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = () => {
                if (retries > 0) {
                    setTimeout(() => {
                        this.loadImage(src, retries - 1).then(resolve).catch(reject);
                    }, 300);
                } else {
                    reject('Не удалось загрузить: ' + src);
                }
            };
            img.src = src;
        });
    }

    // ROOM MANAGEMENT
    changeRoom(newRoom) {
        this.currentRoom = newRoom || 'default';
        document.getElementById('room-id').textContent = this.currentRoom;
        document.getElementById('conference-link').href = `/conference?room=${this.currentRoom}`;
        this.socket.emit('join_room', { room_id: this.currentRoom });
        this.log('Переключено в комнату: ' + this.currentRoom);
    }

    // KNOWLEDGE MANAGEMENT
    async addKnowledge() {
        const subject = document.getElementById('knowledge-subject').value;
        const text = document.getElementById('knowledge-text').value;
        
        if (!text.trim()) {
            this.showStatus('knowledge-status', 'Введите текст для добавления в базу знаний', false);
            return;
        }
        
        try {
            const response = await fetch('/api/add_knowledge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject, text })
            });
            const data = await response.json();
            if (data.success) {
                this.showStatus('knowledge-status', 'Данные успешно добавлены в базу знаний! Добавлено элементов: ' + data.added_items, true);
                document.getElementById('knowledge-text').value = '';
            } else {
                this.showStatus('knowledge-status', 'Ошибка: ' + data.error, false);
            }
        } catch (error) {
            this.showStatus('knowledge-status', 'Ошибка при добавлении данных: ' + error, false);
        }
    }

    async addLesson() {
        const subject = document.getElementById('lesson-subject').value;
        const title = document.getElementById('lesson-title').value;
        const content = document.getElementById('lesson-content').value;
        
        if (!title.trim() || !content.trim()) {
            this.showStatus('lesson-status', 'Заполните название и содержание урока', false);
            return;
        }
        
        try {
            const response = await fetch('/api/add_lesson', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ subject, title, content })
            });
            const data = await response.json();
            if (data.success) {
                this.showStatus('lesson-status', 'Урок "' + title + '" успешно добавлен по предмету "' + subject + '"!', true);
                document.getElementById('lesson-title').value = '';
                document.getElementById('lesson-content').value = '';
            } else {
                this.showStatus('lesson-status', 'Ошибка: ' + data.error, false);
            }
        } catch (error) {
            this.showStatus('lesson-status', 'Ошибка при добавлении урока: ' + error, false);
        }
    }

    // PRACTICE FILES MANAGEMENT
    async loadPracticeFiles() {
        try {
            const response = await fetch('/api/practice_files');
            const data = await response.json();
            if (data.success) {
                this.displayPracticeFiles(data.files);
            } else {
                this.displayPracticeFiles([]);
            }
        } catch (error) {
            this.log('Ошибка загрузки файлов практики: ' + error);
            this.displayPracticeFiles([]);
        }
    }

    displayPracticeFiles(files) {
        const container = document.getElementById('practice-files-list');
        if (!files || files.length === 0) {
            container.innerHTML = '<div class="file-item"><div class="file-info"><div class="file-name">Файлы не найдены</div></div></div>';
            return;
        }
        
        container.innerHTML = files.map(file => `
            <div class="file-item">
                <div class="file-info">
                    <div class="file-name">${file.filename}</div>
                    <div class="file-size">${this.formatFileSize(file.size)} · ${new Date(file.modified).toLocaleDateString()}</div>
                </div>
                <div class="file-actions">
                    <button class="file-btn" onclick="teacherPanel.downloadPracticeFile('${file.filename}')">Скачать</button>
                    <button class="file-btn delete" data-filename="${file.filename}">Удалить</button>
                </div>
            </div>
        `).join('');
        
        // Добавляем обработчики для кнопок удаления
        container.querySelectorAll('.file-btn.delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const filename = e.target.dataset.filename;
                this.deletePracticeFile(filename);
            });
        });
    }

    async deletePracticeFile(filename) {
        if (!confirm('Удалить файл "' + filename + '"?')) return;
        
        try {
            const response = await fetch('/api/delete_practice/' + filename);
            const data = await response.json();
            if (data.success) {
                this.showStatus('upload-status', data.message, true);
                this.loadPracticeFiles();
            } else {
                this.showStatus('upload-status', 'Ошибка: ' + data.error, false);
            }
        } catch (error) {
            this.showStatus('upload-status', 'Ошибка сети: ' + error, false);
        }
    }

    downloadPracticeFile(filename) {
        window.open('/api/download_practice/' + filename, '_blank');
    }

    async uploadPracticeFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/upload_practice', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            if (data.success) {
                this.showStatus('upload-status', data.message, true);
                this.loadPracticeFiles();
            } else {
                this.showStatus('upload-status', 'Ошибка: ' + data.error, false);
            }
        } catch (error) {
            this.showStatus('upload-status', 'Ошибка сети: ' + error, false);
        }
    }

    // DOWNLOAD MANAGEMENT
    async downloadKnowledge() {
        const subject = document.getElementById('download-subject').value;
        
        try {
            const response = await fetch('/api/download_knowledge?subject=' + encodeURIComponent(subject));
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = subject + '_knowledge.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showStatus('download-status', 'База знаний по предмету "' + subject + '" успешно скачана!', true);
            } else {
                throw new Error('Ошибка скачивания');
            }
        } catch (error) {
            this.showStatus('download-status', 'Ошибка при скачивании: ' + error.message, false);
        }
    }

    async downloadLessons() {
        try {
            const response = await fetch('/api/download_lessons');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'ai_teacher_lessons.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showStatus('download-status', 'Все уроки успешно скачаны!', true);
            } else {
                throw new Error('Ошибка скачивания');
            }
        } catch (error) {
            this.showStatus('download-status', 'Ошибка при скачивании уроков: ' + error.message, false);
        }
    }

    async downloadPractice() {
        try {
            const response = await fetch('/api/download_practice');
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.style.display = 'none';
                a.href = url;
                a.download = 'ai_teacher_practice.zip';
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                this.showStatus('download-status', 'Практические задания успешно скачаны!', true);
            } else {
                throw new Error('Ошибка скачивания');
            }
        } catch (error) {
            this.showStatus('download-status', 'Ошибка при скачивании практики: ' + error.message, false);
        }
    }

    // UTILITIES
    log(message) {
        const now = new Date().toLocaleTimeString();
        const debugDiv = document.getElementById('debug');
        debugDiv.innerHTML += `[${now}] ${message}<br>`;
        debugDiv.scrollTop = debugDiv.scrollHeight;
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    showStatus(elementId, message, isSuccess = true) {
        const element = document.getElementById(elementId);
        if (!element) return;
        
        element.textContent = message;
        element.className = isSuccess ? 'success-message' : 'error-message';
        element.style.display = 'block';
        
        setTimeout(() => {
            element.style.display = 'none';
        }, 5000);
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    setupEventListeners() {
        // API Keys
        document.getElementById('save-openrouter-btn').addEventListener('click', () => {
            const key = document.getElementById('openrouter-key').value;
            this.saveApiKey('openrouter', key);
        });

        document.getElementById('test-openrouter-btn').addEventListener('click', () => {
            const key = document.getElementById('openrouter-key').value;
            this.testApiKey('openrouter', key);
        });

        // LLM Mode
        document.getElementById('set-llm-mode-btn').addEventListener('click', () => {
            const mode = document.getElementById('llm-mode-select').value;
            this.setLLMMode(mode);
        });

        // Avatar
        document.getElementById('avatar-select').addEventListener('change', (e) => {
            this.currentAvatar = e.target.value;
            document.getElementById('load-btn').disabled = !this.currentAvatar;
        });

        document.getElementById('load-btn').addEventListener('click', () => {
            if (this.currentAvatar) {
                this.loadAvatarFrames(this.currentAvatar);
            }
        });

        // Room
        document.getElementById('change-room-btn').addEventListener('click', () => {
            const newRoom = document.getElementById('room-input').value.trim() || 'default';
            this.changeRoom(newRoom);
            document.getElementById('room-input').value = '';
        });

        // Knowledge Management
        document.getElementById('add-knowledge-btn').addEventListener('click', () => {
            this.addKnowledge();
        });

        document.getElementById('add-lesson-btn').addEventListener('click', () => {
            this.addLesson();
        });

        // Practice Files
        document.getElementById('select-file-btn').addEventListener('click', () => {
            document.getElementById('practice-file').click();
        });

        document.getElementById('practice-file').addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.uploadPracticeFile(e.target.files[0]);
            }
        });

        document.getElementById('refresh-files-btn').addEventListener('click', () => {
            this.loadPracticeFiles();
        });

        // Download
        document.getElementById('download-knowledge-btn').addEventListener('click', () => {
            this.downloadKnowledge();
        });

        document.getElementById('download-lessons-btn').addEventListener('click', () => {
            this.downloadLessons();
        });

        document.getElementById('download-practice-btn').addEventListener('click', () => {
            this.downloadPractice();
        });

        // Drag and drop для загрузки файлов
        const uploadArea = document.getElementById('upload-area');
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.name.endsWith('.txt')) {
                    this.uploadPracticeFile(file);
                } else {
                    this.showStatus('upload-status', 'Только TXT файлы поддерживаются', false);
                }
            }
        });
    }

    setupTabs() {
        document.querySelectorAll('.tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                
                tab.classList.add('active');
                document.getElementById(tab.dataset.tab + '-tab').classList.add('active');
            });
        });
    }
}

// ИНИЦИАЛИЗАЦИЯ
document.addEventListener('DOMContentLoaded', () => {
    window.teacherPanel = new TeacherPanel();
});
