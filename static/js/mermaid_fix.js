// static/js/mermaid_fix.js

class MermaidFix {
    constructor() {
        this.initialized = false;
        this.queue = [];
        this.init();
    }

    init() {
        if (typeof mermaid === 'undefined') {
            console.error('Mermaid library not loaded');
            return;
        }

        try {
            mermaid.initialize({
                startOnLoad: false, // Важно: отключаем авто-инициализацию
                theme: 'default',
                securityLevel: 'loose',
                flowchart: {
                    useMaxWidth: true,
                    htmlLabels: true,
                    curve: 'basis'
                },
                fontFamily: 'Arial, sans-serif'
            });
            this.initialized = true;
            console.log('Mermaid initialized successfully');
            this.processQueue();
        } catch (error) {
            console.error('Mermaid initialization error:', error);
        }
    }

    renderDiagram(containerId, mermaidCode, topic) {
        if (!this.initialized) {
            this.queue.push({ containerId, mermaidCode, topic });
            return;
        }

        this._render(containerId, mermaidCode, topic);
    }

    _render(containerId, mermaidCode, topic) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`Container ${containerId} not found`);
            return;
        }

        // Очищаем контейнер
        container.innerHTML = '';

        // Создаем новый элемент для диаграммы
        const diagramElement = document.createElement('div');
        diagramElement.className = 'mermaid';
        diagramElement.textContent = this.cleanMermaidCode(mermaidCode);
        
        container.appendChild(diagramElement);

        // Добавляем заголовок
        const title = document.createElement('div');
        title.className = 'diagram-title';
        title.textContent = topic || 'Диаграмма';
        title.style.cssText = `
            text-align: center;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
            font-size: 16px;
        `;
        container.insertBefore(title, diagramElement);

        // Рендерим с задержкой для гарантии отрисовки
        setTimeout(() => {
            try {
                mermaid.init(undefined, [diagramElement]).then(() => {
                    console.log('Diagram rendered successfully:', topic);
                    
                    // Применяем дополнительные стили
                    this.applyStyles(container);
                }).catch(error => {
                    console.error('Mermaid rendering error:', error);
                    this.showError(container, error, mermaidCode);
                });
            } catch (error) {
                console.error('Mermaid execution error:', error);
                this.showError(container, error, mermaidCode);
            }
        }, 100);
    }

    cleanMermaidCode(code) {
        if (!code) return 'graph TD\nA["Пустая диаграмма"]';
        
        // Удаляем markdown обратные кавычки
        let cleaned = code.replace(/```[\s\S]*?```/g, '');
        cleaned = cleaned.replace(/`/g, '');
        
        // Удаляем комментарии
        cleaned = cleaned.replace(/%%[^%\n]*/g, '');
        
        // Проверяем базовый синтаксис
        const validStarts = ['graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 'stateDiagram', 'pie', 'gantt'];
        const hasValidStart = validStarts.some(start => cleaned.trim().startsWith(start));
        
        if (!hasValidStart) {
            cleaned = 'flowchart TD\n' + cleaned;
        }
        
        // Обеспечиваем минимальную структуру
        if (!cleaned.includes('-->') && !cleaned.includes('->')) {
            cleaned += '\nA["Элемент A"] --> B["Элемент B"]';
        }
        
        return cleaned.trim();
    }

    applyStyles(container) {
        const svg = container.querySelector('svg');
        if (svg) {
            svg.style.maxWidth = '100%';
            svg.style.height = 'auto';
            svg.style.display = 'block';
            svg.style.margin = '0 auto';
            
            // Улучшаем читаемость текста
            const textElements = svg.querySelectorAll('text');
            textElements.forEach(text => {
                text.style.fontFamily = 'Arial, sans-serif';
                text.style.fontSize = '14px';
            });
        }
    }

    showError(container, error, originalCode) {
        container.innerHTML = `
            <div class="visualization-error" style="
                padding: 20px;
                text-align: center;
                color: #d32f2f;
                background: #ffebee;
                border-radius: 8px;
                border: 1px solid #ffcdd2;
            ">
                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 10px;"></i>
                <p><strong>Ошибка отображения диаграммы</strong></p>
                <p style="font-size: 12px; margin: 10px 0;">${error.message}</p>
                <details style="text-align: left; margin-top: 10px;">
                    <summary style="cursor: pointer; font-size: 12px;">Исходный код</summary>
                    <pre style="background: #f5f5f5; padding: 10px; border-radius: 4px; overflow: auto; font-size: 10px; margin-top: 5px;">${originalCode}</pre>
                </details>
                <button onclick="mermaidFix.retryLast()" style="
                    margin-top: 10px;
                    padding: 8px 16px;
                    background: #4263EB;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                ">Повторить</button>
            </div>
        `;
    }

    processQueue() {
        while (this.queue.length > 0) {
            const item = this.queue.shift();
            this._render(item.containerId, item.mermaidCode, item.topic);
        }
    }

    retryLast() {
        // Можно добавить логику для повторной попытки
        console.log('Retry last visualization');
    }
}

// Глобальный экземпляр
window.mermaidFix = new MermaidFix();
