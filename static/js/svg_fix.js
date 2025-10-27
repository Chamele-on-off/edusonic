// static/js/svg_fix.js

class SVGFix {
    constructor() {
        this.initialized = false;
        this.queue = [];
        this.init();
    }

    init() {
        this.initialized = true;
        this.processQueue();
        console.log('SVG Fix initialized');
    }

    renderSVG(containerId, svgCode, topic) {
        if (!this.initialized) {
            this.queue.push({ containerId, svgCode, topic });
            return;
        }

        this._render(containerId, svgCode, topic);
    }

    _render(containerId, svgCode, topic) {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error(`SVG container ${containerId} not found`);
            return;
        }

        // Показываем индикатор загрузки
        this.showLoading(container, topic);

        // Обрабатываем SVG с задержкой для стабильности
        setTimeout(() => {
            try {
                const processedSVG = this.processSVGCode(svgCode, topic);
                container.innerHTML = processedSVG;
                
                // Применяем стили и настройки
                this.applySVGStyles(container);
                
                console.log('SVG rendered successfully:', topic);
                
            } catch (error) {
                console.error('SVG rendering error:', error);
                this.showError(container, error, svgCode, topic);
            }
        }, 100);
    }

    processSVGCode(svgCode, topic) {
        if (!svgCode || svgCode.trim() === '') {
            return this.generateFallbackSVG(topic || 'Пустая диаграмма');
        }

        // Очищаем и валидируем SVG код
        let cleaned = this.cleanSVGCode(svgCode);
        
        // Парсим и валидируем SVG
        const parser = new DOMParser();
        const doc = parser.parseFromString(cleaned, 'image/svg+xml');
        
        // Проверяем на ошибки парсинга
        const parserError = doc.querySelector('parsererror');
        if (parserError) {
            throw new Error(`SVG parsing error: ${parserError.textContent}`);
        }

        const svgElement = doc.querySelector('svg');
        if (!svgElement) {
            throw new Error('No SVG element found in code');
        }

        // Применяем обязательные атрибуты
        this.applyRequiredAttributes(svgElement);
        
        // Добавляем заголовок
        return this.wrapWithTitle(svgElement.outerHTML, topic);
    }

    cleanSVGCode(svgCode) {
        let cleaned = svgCode.trim();
        
        // Удаляем XML declaration и комментарии
        cleaned = cleaned.replace(/<\?xml[^>]*\?>/g, '');
        cleaned = cleaned.replace(/<!--[\s\S]*?-->/g, '');
        
        // Удаляем лишние теги script и style которые могут вызывать проблемы
        cleaned = cleaned.replace(/<script[\s\S]*?<\/script>/gi, '');
        cleaned = cleaned.replace(/<style[\s\S]*?<\/style>/gi, '');
        
        // Убеждаемся что есть namespace
        if (!cleaned.includes('xmlns=')) {
            cleaned = cleaned.replace('<svg', '<svg xmlns="http://www.w3.org/2000/svg"');
        }
        
        // Добавляем закрывающий тег если его нет
        if (!cleaned.includes('</svg>')) {
            cleaned += '</svg>';
        }
        
        return cleaned;
    }

    applyRequiredAttributes(svgElement) {
        // Устанавливаем обязательные атрибуты
        if (!svgElement.hasAttribute('width')) {
            svgElement.setAttribute('width', '100%');
        }
        
        if (!svgElement.hasAttribute('height')) {
            svgElement.setAttribute('height', '100%');
        }
        
        if (!svgElement.hasAttribute('viewBox')) {
            // Создаем разумный viewBox по умолчанию
            svgElement.setAttribute('viewBox', '0 0 400 300');
        }
        
        if (!svgElement.hasAttribute('preserveAspectRatio')) {
            svgElement.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        }
        
        // Убираем проблемные атрибуты
        svgElement.removeAttribute('onload');
        svgElement.removeAttribute('onerror');
        
        // Добавляем базовые стили для всех элементов
        this.addBaseStyles(svgElement);
    }

    addBaseStyles(svgElement) {
        // Создаем элемент style если его нет
        let styleElement = svgElement.querySelector('style');
        if (!styleElement) {
            styleElement = document.createElementNS('http://www.w3.org/2000/svg', 'style');
            svgElement.insertBefore(styleElement, svgElement.firstChild);
        }
        
        const baseStyles = `
            text { 
                font-family: Arial, sans-serif; 
                font-size: 14px; 
                fill: #333; 
            }
            rect, circle, ellipse, polygon { 
                stroke-width: 2; 
            }
            path, line, polyline {
                stroke-width: 2;
                fill: none;
            }
        `;
        
        if (!styleElement.textContent.includes('text {')) {
            styleElement.textContent += baseStyles;
        }
    }

    wrapWithTitle(svgContent, topic) {
        return `
            <div class="svg-diagram-container">
                <div class="diagram-title">
                    <i class="fas fa-shapes"></i>
                    ${topic || 'SVG Диаграмма'}
                </div>
                <div class="svg-content">
                    ${svgContent}
                </div>
            </div>
        `;
    }

    generateFallbackSVG(topic) {
        return `
            <div class="svg-diagram-container">
                <div class="diagram-title">
                    <i class="fas fa-shapes"></i>
                    ${topic}
                </div>
                <div class="svg-content">
                    <svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid meet">
                        <style>
                            .fallback-text { font-family: Arial; font-size: 14px; fill: #666; }
                            .fallback-shape { fill: #4263EB; stroke: #3a0ca3; stroke-width: 2; }
                        </style>
                        <rect x="50" y="50" width="300" height="200" rx="10" class="fallback-shape" opacity="0.1"/>
                        <text x="200" y="120" text-anchor="middle" class="fallback-text">
                            <tspan x="200" dy="0">Диаграмма</tspan>
                            <tspan x="200" dy="20">не сгенерирована</tspan>
                        </text>
                        <text x="200" y="180" text-anchor="middle" class="fallback-text" font-size="12">
                            Попробуйте обновить страницу
                        </text>
                    </svg>
                </div>
            </div>
        `;
    }

    applySVGStyles(container) {
        const svg = container.querySelector('svg');
        if (svg) {
            // Применяем responsive стили
            svg.style.maxWidth = '100%';
            svg.style.maxHeight = '100%';
            svg.style.display = 'block';
            svg.style.margin = 'auto';
            
            // Предотвращаем переполнение
            svg.style.overflow = 'visible';
        }
        
        // Настраиваем контейнер
        const svgContent = container.querySelector('.svg-content');
        if (svgContent) {
            svgContent.style.flex = '1';
            svgContent.style.display = 'flex';
            svgContent.style.alignItems = 'center';
            svgContent.style.justifyContent = 'center';
            svgContent.style.overflow = 'auto';
        }
    }

    showLoading(container, topic) {
        container.innerHTML = `
            <div class="svg-diagram-container">
                <div class="diagram-title">
                    <i class="fas fa-shapes"></i>
                    ${topic || 'Загрузка SVG...'}
                </div>
                <div class="svg-loading">
                    <div class="loading-spinner"></div>
                    <p>Генерация SVG диаграммы...</p>
                </div>
            </div>
        `;
    }

    showError(container, error, originalCode, topic) {
        console.error('SVG Error:', error);
        
        container.innerHTML = `
            <div class="svg-diagram-container">
                <div class="diagram-title error-title">
                    <i class="fas fa-exclamation-triangle"></i>
                    Ошибка загрузки: ${topic}
                </div>
                <div class="svg-error">
                    <div class="error-icon">
                        <i class="fas fa-times-circle"></i>
                    </div>
                    <div class="error-message">
                        <p><strong>Не удалось отобразить SVG диаграмму</strong></p>
                        <p class="error-details">${error.message}</p>
                        <button onclick="svgFix.retryRender('${container.id}', ${JSON.stringify(originalCode).replace(/'/g, "\\'")}, '${topic}')" 
                                class="retry-button">
                            <i class="fas fa-redo"></i> Повторить
                        </button>
                        <button onclick="svgFix.showFallback('${container.id}', '${topic}')" 
                                class="fallback-button">
                            <i class="fas fa-shapes"></i> Показать шаблон
                        </button>
                    </div>
                </div>
            </div>
        `;
    }

    retryRender(containerId, svgCode, topic) {
        console.log('Retrying SVG render...');
        this.renderSVG(containerId, svgCode, topic);
    }

    showFallback(containerId, topic) {
        const container = document.getElementById(containerId);
        if (container) {
            container.innerHTML = this.generateFallbackSVG(topic);
            this.applySVGStyles(container);
        }
    }

    processQueue() {
        while (this.queue.length > 0) {
            const item = this.queue.shift();
            this._render(item.containerId, item.svgCode, item.topic);
        }
    }
}

// Глобальный экземпляр
window.svgFix = new SVGFix();
