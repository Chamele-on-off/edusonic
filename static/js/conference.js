// ГЛАВНЫЙ КЛАСС ДЛЯ КОНФЕРЕНЦИИ
class Conference {
    constructor() {
        this.roomId = new URLSearchParams(window.location.search).get('room') || 'default';
        this.isEmbedded = window.location.search.includes('embed=true');
        this.socket = null;
        this.isMicOn = false;
        this.isCamOn = false;
        this.isFullscreen = false;
        this.userStream = null;
        this.recognition = null;
        this.aiTeacherActivated = false;
        this.currentAvatar = 'teacher';
        this.loadedImages = [];
        this.currentFrame = 0;
        this.animationInterval = null;
        this.isSpeechAnimating = false;
        this.speechAnimationTimeout = null;
        this.isDrawing = false;
        this.currentTool = 'pen';
        this.drawingCtx = null;
        this.currentColor = '#000000';
        this.isEraser = false;
        this.isDrawingWindowOpen = false;
        this.isDraggingWindow = false;
        this.currentDraggedWindow = null;
        this.dragOffsetX = 0;
        this.dragOffsetY = 0;
        this.audioQueue = [];
        this.isPlayingAudio = false;
        this.audioElements = [];
        this.audioContext = null;
        this.isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream;
        this.currentLesson = null;
        this.currentParagraph = 0;
        this.currentLessonContent = [];
        this.devicesEnabled = false;
        this.currentAvatarAspectRatio = 1;
        this.isLessonActive = false;
        this.currentTypingAnimation = null;
        
        this.init();
    }

    async init() {
        this.setupUI();
        this.createBackgroundAnimation();
        this.setupSocket();
        this.setupDrawingCanvas();
        this.setupWindowDragging();
        this.setupZoom();
        await this.setupMedia();
        this.setupSpeechRecognition();
    }

    setupUI() {
        document.getElementById('room-name').textContent = this.roomId;
        
        if (this.isEmbedded) {
            document.body.classList.add('embedded');
        }
        
        this.setupControlButtons();
    }

    setupControlButtons() {
        document.getElementById('mic-btn').addEventListener('click', () => this.toggleMicrophone());
        document.getElementById('cam-btn').addEventListener('click', () => this.toggleCamera());
        document.getElementById('screen-btn').addEventListener('click', () => this.toggleScreenShare());
        document.getElementById('fullscreen-btn').addEventListener('click', () => this.toggleFullscreen());
        document.getElementById('end-call-btn').addEventListener('click', () => this.endCall());
        document.getElementById('activate-ai-btn').addEventListener('click', () => this.activateAI());
        document.getElementById('enable-camera-btn').addEventListener('click', () => this.enableCameraAndMicrophone());
        
        // Lesson controls
        document.getElementById('close-lesson').addEventListener('click', () => this.closeLesson());
        document.getElementById('text-tool').addEventListener('click', () => this.setTool('text'));
        document.getElementById('draw-tool').addEventListener('click', () => this.setTool('draw'));
        document.getElementById('answer-tool').addEventListener('click', () => this.setTool('answer'));
        document.getElementById('clear-text').addEventListener('click', () => this.clearText());
        document.getElementById('submit-answer').addEventListener('click', () => this.submitAnswer());
        document.getElementById('answer-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.submitAnswer();
        });
        
        // Drawing controls
        document.getElementById('close-drawing').addEventListener('click', () => this.closeDrawing());
        document.getElementById('drawing-pen').addEventListener('click', () => this.setDrawingTool('pen'));
        document.getElementById('drawing-eraser').addEventListener('click', () => this.setDrawingTool('eraser'));
        document.getElementById('clear-drawing').addEventListener('click', () => this.clearDrawing());
        document.getElementById('close-text-widget').addEventListener('click', () => this.closeTextWidget());
        
        // Color picker
        document.querySelectorAll('.color-option').forEach(option => {
            option.addEventListener('click', () => {
                document.querySelectorAll('.color-option').forEach(opt => opt.classList.remove('active'));
                option.classList.add('active');
                this.currentColor = option.dataset.color;
                if (!this.isEraser) {
                    this.drawingCtx.strokeStyle = this.currentColor;
                }
            });
        });
    }

    setupSocket() {
        this.socket = io();
        
        this.socket.on('connect', () => {
            this.updateStatus('Подключено к серверу');
            this.socket.emit('join_room', { room_id: this.roomId });
            
            setTimeout(() => {
                this.socket.emit('get_current_avatar', { room_id: this.roomId });
            }, 500);
        });

        this.socket.on('disconnect', () => {
            this.updateStatus('Отключено от сервера');
        });

        this.socket.on('avatar_changed', (data) => {
            this.log('Смена аватара на: ' + data.avatar_name);
            this.currentAvatar = data.avatar_name;
            this.loadAvatarFrames(data.avatar_name);
        });

        this.socket.on('current_avatar', (data) => {
            this.log('Текущий аватар: ' + data.avatar_name);
            this.currentAvatar = data.avatar_name;
            this.loadAvatarFrames(data.avatar_name);
        });

        this.socket.on('speech_audio', (data) => {
            try {
                this.log('Получено аудио сообщение, текст: ' + data.text.substring(0, 100) + '...');
                this.startSpeechAnimation(data.text);
                this.playAudio(data.audio);
                if (!this.isLessonActive) {
                    this.showTeacherSpeech(data.text);
                }
            } catch (error) {
                this.log('Ошибка обработки аудио: ' + error);
                if (!this.isLessonActive) {
                    this.showTeacherSpeech(data.text);
                }
            }
        });

        this.socket.on('speech_text', (data) => {
            if (data.is_teacher) {
                const teacherText = data.text.replace('Учитель: ', '');
                this.startSpeechAnimation(teacherText);
                if (!this.isLessonActive) {
                    this.showTeacherSpeech(data.text);
                }
                if (this.currentLesson && data.text.startsWith('Учитель: ')) {
                    const lessonText = data.text.replace('Учитель: ', '');
                    this.updateLessonText(lessonText);
                }
            } else {
                this.showStudentSpeech(data.text, data.sid);
            }
        });

        this.socket.on('participants_update', (data) => {
            document.getElementById('participants-count').textContent = data.count;
        });

        this.socket.on('ai_teacher_available', () => {
            if (this.devicesEnabled) {
                document.getElementById('activate-ai-btn').style.display = 'block';
            }
        });

        this.socket.on('ai_teacher_activated', () => {
            this.aiTeacherActivated = true;
            document.getElementById('activate-ai-btn').style.display = 'none';
            this.updateStatus('AI-Teacher активирован');
            
            if (this.isIOS) {
                setTimeout(() => {
                    if (this.recognition && this.isMicOn) {
                        this.recognition.start();
                    }
                }, 2000);
            }
        });

        this.socket.on('lesson_started', (data) => {
            this.isLessonActive = true;
            this.currentLesson = data;
            this.currentParagraph = 0;
            this.loadLessonContent(data.lesson_id);
        });

        this.socket.on('animation_ready', (data) => {
            this.log('Анимация готова: ' + data.status);
            if (data.status === 'ready' && this.loadedImages.length > 0) {
                this.startAnimation();
            }
        });

        this.socket.on('practice_started', () => {
            this.log('Практика начата');
        });

        this.socket.on('practice_ended', () => {
            this.log('Практика завершена');
        });
    }

    createBackgroundAnimation() {
        const backgroundAnimation = document.getElementById('background-animation');
        for (let i = 0; i < 8; i++) {
            const circle = document.createElement('div');
            circle.className = 'animated-circle';
            
            const size = Math.random() * 80 + 40;
            circle.style.width = `${size}px`;
            circle.style.height = `${size}px`;
            circle.style.left = `${Math.random() * 100}%`;
            circle.style.top = `${Math.random() * 100}%`;
            
            const colors = ['#4361ee', '#3a0ca3', '#4cc9f0', '#f72585'];
            circle.style.backgroundColor = colors[Math.floor(Math.random() * colors.length)];
            
            backgroundAnimation.appendChild(circle);
            
            // Анимация с anime.js
            anime({
                targets: circle,
                translateX: () => anime.random(-200, 200),
                translateY: () => anime.random(-200, 200),
                scale: () => anime.random(0.5, 1.2),
                duration: () => anime.random(4000, 12000),
                easing: 'easeInOutQuad',
                complete: function(anim) {
                    anim.restart();
                }
            });
        }
    }

    async setupMedia() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                },
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    sampleRate: 44100
                }
            });
            
            this.userStream = stream;
            document.getElementById('self-video').srcObject = this.userStream;
            this.isCamOn = true;
            document.getElementById('cam-btn').classList.add('active');
            document.getElementById('cam-btn').querySelector('span').textContent = 'Выкл';
            document.getElementById('self-video-container').style.display = 'block';
            
            this.isMicOn = true;
            document.getElementById('mic-btn').classList.add('active');
            document.getElementById('mic-btn').querySelector('span').textContent = 'Выкл';
            
            this.updateStatus('Камера и микрофон включены');
            document.getElementById('camera-permission').style.display = 'none';
            this.devicesEnabled = true;
            document.getElementById('activate-ai-btn').style.display = 'block';
            
        } catch (error) {
            this.log('Ошибка доступа к камере/микрофону: ' + error);
            document.getElementById('camera-permission').style.display = 'block';
            
            if (error.name === 'NotAllowedError') {
                this.updateStatus('Доступ к камере/микрофону запрещен');
            } else {
                this.updateStatus('Ошибка доступа к устройствам');
            }
        }
    }

    async enableCameraAndMicrophone() {
        await this.setupMedia();
    }

    async toggleMicrophone() {
        try {
            if (!this.isMicOn) {
                const audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        echoCancellation: true,
                        noiseSuppression: true,
                        sampleRate: 44100
                    }
                });
                if (this.userStream) {
                    this.userStream.getAudioTracks().forEach(track => this.userStream.addTrack(track));
                } else {
                    this.userStream = audioStream;
                }
                this.isMicOn = true;
                document.getElementById('mic-btn').classList.add('active');
                document.getElementById('mic-btn').querySelector('span').textContent = 'Выкл';
                this.updateStatus('Микрофон включен');
                
                if (this.recognition) {
                    if (this.isIOS) {
                        setTimeout(() => {
                            this.recognition.start();
                        }, 500);
                    } else {
                        this.recognition.start();
                    }
                }
            } else {
                if (this.userStream) {
                    this.userStream.getAudioTracks().forEach(track => track.stop());
                }
                this.isMicOn = false;
                document.getElementById('mic-btn').classList.remove('active');
                document.getElementById('mic-btn').querySelector('span').textContent = 'Вкл';
                this.updateStatus('Микрофон выключен');
                
                if (this.recognition) {
                    this.recognition.stop();
                }
            }
        } catch (error) {
            this.updateStatus('Ошибка доступа к микрофону');
            this.log('Ошибка доступа к микрофону: ' + error);
        }
    }

    async toggleCamera() {
        try {
            if (!this.isCamOn) {
                const videoStream = await navigator.mediaDevices.getUserMedia({ 
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    }
                });
                
                if (this.userStream) {
                    this.userStream.getVideoTracks().forEach(track => this.userStream.addTrack(track));
                } else {
                    this.userStream = videoStream;
                }
                
                document.getElementById('self-video').srcObject = this.userStream;
                this.isCamOn = true;
                document.getElementById('cam-btn').classList.add('active');
                document.getElementById('cam-btn').querySelector('span').textContent = 'Выкл';
                this.updateStatus('Камера включена');
                document.getElementById('self-video-container').style.display = 'block';
                document.getElementById('camera-permission').style.display = 'none';
            } else {
                if (this.userStream) {
                    this.userStream.getVideoTracks().forEach(track => track.stop());
                    document.getElementById('self-video').srcObject = null;
                }
                this.isCamOn = false;
                document.getElementById('cam-btn').classList.remove('active');
                document.getElementById('cam-btn').querySelector('span').textContent = 'Вкл';
                this.updateStatus('Камера выключена');
                document.getElementById('self-video-container').style.display = 'none';
            }
        } catch (error) {
            this.updateStatus('Ошибка доступа к камере');
            this.log('Ошибка доступа к камере: ' + error);
            
            if (error.name === 'NotAllowedError') {
                document.getElementById('camera-permission').style.display = 'block';
            }
        }
    }

    toggleScreenShare() {
        // Заглушка для демонстрации экрана
        this.log('Демонстрация экрана временно недоступна');
    }

    toggleFullscreen() {
        if (!this.isFullscreen) {
            if (document.documentElement.requestFullscreen) {
                document.documentElement.requestFullscreen();
            } else if (document.documentElement.webkitRequestFullscreen) {
                document.documentElement.webkitRequestFullscreen();
            } else if (document.documentElement.msRequestFullscreen) {
                document.documentElement.msRequestFullscreen();
            }
            this.isFullscreen = true;
            document.getElementById('fullscreen-btn').querySelector('span').textContent = 'Обычный';
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            } else if (document.msExitFullscreen) {
                document.msExitFullscreen();
            }
            this.isFullscreen = false;
            document.getElementById('fullscreen-btn').querySelector('span').textContent = 'Полный';
        }
    }

    activateAI() {
        this.socket.emit('activate_ai_teacher', { room_id: this.roomId });
    }

    endCall() {
        if (this.userStream) {
            this.userStream.getTracks().forEach(track => track.stop());
        }
        if (this.recognition) {
            this.recognition.stop();
        }
        this.stopAnimation();
        if (this.socket) {
            this.socket.emit('leave_room', { room_id: this.roomId });
            this.socket.disconnect();
        }
        
        if (this.isEmbedded) {
            window.parent.postMessage('conference_ended', '*');
        } else {
            window.location.href = '/';
        }
    }

    // AVATAR ANIMATION
    loadAvatarFrames(avatarName) {
        this.log('Начало загрузки аватара: ' + avatarName);
        this.updateStatus('Загрузка ' + avatarName + '...');
        this.loadedImages = [];
        document.getElementById('avatar-loading').style.display = 'block';
        
        this.stopAnimation();
        
        fetch('/api/frames/' + encodeURIComponent(avatarName))
            .then(response => {
                if (!response.ok) throw new Error('HTTP error ' + response.status);
                return response.json();
            })
            .then(data => {
                if (data.error) throw new Error(data.error);
                if (!data.frames || data.frames.length === 0) throw new Error('Нет доступных кадров');
                
                this.log('Найдено кадров: ' + data.frames.length);
                return this.loadAllFrames(avatarName, data.frames);
            })
            .then(images => {
                document.getElementById('avatar-loading').style.display = 'none';
                
                if (images && images.length > 0) {
                    this.loadedImages = images;
                    this.updateStatus(avatarName + ': загружено ' + this.loadedImages.length + ' кадров');
                    this.startAnimation();
                    this.log('Анимация запущена с ' + this.loadedImages.length + ' кадрами');
                } else {
                    throw new Error('Не удалось загрузить кадры');
                }
            })
            .catch(error => {
                document.getElementById('avatar-loading').style.display = 'none';
                this.updateStatus('Ошибка: ' + error.message);
                this.log('Ошибка загрузки аватара: ' + error);
                this.showAvatarPlaceholder();
            });
    }

    loadAllFrames(avatarName, frameList) {
        return new Promise((resolve) => {
            const images = [];
            let loadedCount = 0;
            let errorCount = 0;
            
            if (!frameList || frameList.length === 0) {
                resolve([]);
                return;
            }
            
            const loadNext = (index) => {
                if (index >= frameList.length) {
                    this.log('Загружено кадров: ' + images.length + ' из ' + frameList.length);
                    resolve(images);
                    return;
                }
                
                const frame = frameList[index];
                const imageUrl = '/frames/' + encodeURIComponent(avatarName) + '/' + encodeURIComponent(frame);
                
                this.loadImage(imageUrl, 2)
                    .then(img => {
                        images.push(img);
                        loadedCount++;
                        
                        if (loadedCount % 5 === 0 || loadedCount === frameList.length) {
                            this.updateStatus(`${avatarName}: ${loadedCount}/${frameList.length}`);
                        }
                        
                        loadNext(index + 1);
                    })
                    .catch(error => {
                        errorCount++;
                        this.log('Ошибка загрузки кадра ' + frame + ': ' + error);
                        loadNext(index + 1);
                    });
            };
            
            loadNext(0);
        });
    }

    loadImage(src, retries = 3) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = "anonymous";
            
            img.onload = () => {
                this.log('Изображение загружено: ' + src);
                resolve(img);
            };
            
            img.onerror = () => {
                if (retries > 0) {
                    this.log(`Повторная попытка загрузки: ${src} (осталось попыток: ${retries})`);
                    setTimeout(() => {
                        this.loadImage(src, retries - 1).then(resolve).catch(reject);
                    }, 300);
                } else {
                    reject(new Error(`Не удалось загрузить: ${src}`));
                }
            };
            
            const timestamp = new Date().getTime();
            const separator = src.includes('?') ? '&' : '?';
            img.src = src + separator + 't=' + timestamp;
        });
    }

    showAvatarPlaceholder() {
        const canvas = document.createElement('canvas');
        canvas.width = 400;
        canvas.height = 400;
        const ctx = canvas.getContext('2d');
        
        ctx.fillStyle = '#3498db';
        ctx.fillRect(0, 0, 400, 400);
        
        ctx.fillStyle = '#fff';
        ctx.beginPath();
        ctx.arc(200, 150, 50, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 10;
        ctx.beginPath();
        ctx.arc(200, 150, 30, 0, Math.PI);
        ctx.stroke();
        
        ctx.fillStyle = '#fff';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Аватар не найден', 200, 350);
        
        document.getElementById('teacher-video').src = canvas.toDataURL();
        this.stopAnimation();
    }

    showFrame() {
        if (this.loadedImages.length === 0) {
            this.log('Нет кадров для отображения');
            return;
        }
        
        try {
            this.currentFrame = this.currentFrame % this.loadedImages.length;
            const img = this.loadedImages[this.currentFrame];
            
            if (img && img.src) {
                document.getElementById('teacher-video').src = img.src;
                document.getElementById('teacher-video').style.display = 'block';
            }
        } catch (error) {
            this.log('Ошибка показа кадра: ' + error);
        }
    }

    startSpeechAnimation(text) {
        this.stopSpeechAnimation();
        
        const duration = this.calculateSpeechDuration(text);
        
        this.log(`Запуск анимации речи на ${duration}ms для текста: "${text.substring(0, 50)}..."`);
        
        this.startFastAnimation();
        this.isSpeechAnimating = true;
        this.updateAnimationStatus();
        
        this.speechAnimationTimeout = setTimeout(() => {
            this.stopSpeechAnimation();
            this.log('Анимация речи завершена');
        }, duration);
    }

    stopSpeechAnimation() {
        if (this.speechAnimationTimeout) {
            clearTimeout(this.speechAnimationTimeout);
            this.speechAnimationTimeout = null;
        }
        
        if (this.isSpeechAnimating) {
            this.isSpeechAnimating = false;
            this.startSlowAnimation();
            this.updateAnimationStatus();
            this.log('Анимация речи остановлена, переход в режим покоя');
        }
    }

    calculateSpeechDuration(text) {
        if (!text || text.length === 0) return 2000;
        
        const punctuationPauses = (text.match(/[.!?;:]/g) || []).length;
        const pauseBonus = punctuationPauses * 300;
        
        const charsPerSecond = 15;
        const baseDuration = (text.length / charsPerSecond) * 1000;
        
        const totalDuration = baseDuration + pauseBonus;
        
        const minDuration = 2000;
        const maxDuration = 45000;
        
        return Math.max(minDuration, Math.min(maxDuration, totalDuration));
    }

    startFastAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
        }
        
        if (this.loadedImages.length === 0) return;
        
        this.animationInterval = setInterval(() => {
            this.currentFrame = (this.currentFrame + 1) % this.loadedImages.length;
            this.showFrame();
        }, 1000 / 15);
    }

    startSlowAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
        }
        
        if (this.loadedImages.length === 0) return;
        
        this.animationInterval = setInterval(() => {
            this.currentFrame = (this.currentFrame + 1) % this.loadedImages.length;
            this.showFrame();
        }, 1000 / 3);
    }

    startAnimation() {
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
        }
        
        if (this.loadedImages.length === 0) {
            this.log('Нет загруженных кадров для анимации');
            return;
        }
        
        this.log('Запуск анимации покоя с ' + this.loadedImages.length + ' кадрами');
        this.startSlowAnimation();
        this.showFrame();
        this.updateAnimationStatus();
    }

    stopAnimation() {
        this.stopSpeechAnimation();
        
        if (this.animationInterval) {
            clearInterval(this.animationInterval);
            this.animationInterval = null;
        }
        
        if (this.loadedImages.length > 0) {
            this.currentFrame = 0;
            this.showFrame();
        }
        
        this.updateAnimationStatus();
    }

    updateAnimationStatus() {
        const animationStatus = document.getElementById('animation-status');
        if (!animationStatus) return;
        
        if (this.isSpeechAnimating) {
            animationStatus.textContent = '🗣️ Анимация речи';
            animationStatus.style.color = '#4cc9f0';
        } else {
            animationStatus.textContent = '😴 Режим покоя';
            animationStatus.style.color = '#95a5a6';
        }
    }

    // AUDIO MANAGEMENT
    playAudio(base64Audio) {
        try {
            if (this.isIOS) {
                this.playAudioForIOS(base64Audio);
            } else {
                const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
                audio.preload = 'auto';
                
                this.audioQueue.push(audio);
                this.playNextAudio();
            }
        } catch (error) {
            this.log('Ошибка воспроизведения аудио: ' + error);
        }
    }

    playNextAudio() {
        if (this.isPlayingAudio || this.audioQueue.length === 0) return;
        
        this.isPlayingAudio = true;
        const audio = this.audioQueue.shift();
        
        const playPromise = audio.play();
        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    audio.onended = () => {
                        this.isPlayingAudio = false;
                        setTimeout(() => this.playNextAudio(), 100);
                    };
                })
                .catch(error => {
                    this.log('Ошибка автовоспроизведения: ' + error);
                    this.isPlayingAudio = false;
                    setTimeout(() => this.playNextAudio(), 100);
                });
        }
    }

    playAudioForIOS(base64Audio) {
        const audioFix = document.getElementById('audio-fix');
        audioFix.src = `data:audio/mp3;base64,${base64Audio}`;
        
        const playPromise = audioFix.play();
        if (playPromise !== undefined) {
            playPromise
                .then(() => {
                    // Audio started successfully
                })
                .catch(error => {
                    this.log('iOS audio play failed: ' + error);
                    this.showTeacherSpeech("(Аудио сообщение)");
                });
        }
    }

    // SPEECH RECOGNITION
    setupSpeechRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
            this.recognition.continuous = true;
            this.recognition.interimResults = true;
            this.recognition.lang = 'ru-RU';
            
            // Настройки чувствительности
            this.recognition.continuous = true;
            this.recognition.interimResults = false;
            this.recognition.maxAlternatives = 1;
            
            let silenceTimer = null;
            let lastFinalResult = '';

            this.recognition.onresult = (event) => {
                if (silenceTimer) clearTimeout(silenceTimer);
                
                let finalTranscript = '';
                let interimTranscript = '';
                
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    const transcript = event.results[i][0].transcript;
                    if (event.results[i].isFinal) {
                        finalTranscript += transcript;
                    } else {
                        interimTranscript += transcript;
                    }
                }

                if (finalTranscript) {
                    const cleanText = finalTranscript.trim();
                    
                    // Фильтры для уменьшения ложных срабатываний
                    if (cleanText.length < 3) {
                        this.log('Отфильтрована короткая фраза: ' + cleanText);
                        return;
                    }
                    
                    if (cleanText === lastFinalResult) {
                        this.log('Повторяющаяся фраза: ' + cleanText);
                        return;
                    }
                    
                    const noisePatterns = [
                        /^[а-я]*ммм[а-я]*$/i,
                        /^[а-я]*эээ[а-я]*$/i,
                        /^[а-я]*ах[а-я]*$/i,
                        /^[а-я]*ох[а-я]*$/i,
                    ];
                    
                    const isNoise = noisePatterns.some(pattern => pattern.test(cleanText));
                    if (isNoise) {
                        this.log('Отфильтрован шум: ' + cleanText);
                        return;
                    }
                    
                    this.log('Распознана речь: ' + cleanText);
                    lastFinalResult = cleanText;
                    
                    if (this.socket) {
                        this.socket.emit('recognized_speech', {
                            room_id: this.roomId,
                            text: cleanText
                        });
                        this.showStudentSpeech(cleanText, 'me');
                    }
                }
                
                this.updateSpeechStatus(!!interimTranscript);
                
                silenceTimer = setTimeout(() => {
                    this.log('Автоостановка из-за молчания');
                    this.updateSpeechStatus(false);
                    if (this.recognition && this.isMicOn) {
                        this.recognition.stop();
                    }
                }, 3000);
            };

            this.recognition.onerror = (event) => {
                this.log('Ошибка распознавания речи: ' + event.error);
                this.updateSpeechStatus(false);
                
                if (event.error === 'no-speech' || event.error === 'audio-capture') {
                    setTimeout(() => {
                        if (this.isMicOn && this.recognition) {
                            try {
                                this.recognition.start();
                                this.updateSpeechStatus(true);
                            } catch (e) {
                                this.log('Ошибка перезапуска распознавания: ' + e);
                            }
                        }
                    }, 1000);
                }
            };

            this.recognition.onend = () => {
                this.updateSpeechStatus(false);
                if (silenceTimer) clearTimeout(silenceTimer);
                
                if (this.isMicOn) {
                    this.log('Перезапуск распознавания речи');
                    setTimeout(() => {
                        if (this.isMicOn && this.recognition) {
                            try {
                                this.recognition.start();
                            } catch (e) {
                                this.log('Ошибка перезапуска: ' + e);
                            }
                        }
                    }, 500);
                }
            };
            
            if (this.isMicOn) {
                setTimeout(() => {
                    this.recognition.start();
                    this.updateSpeechStatus(true);
                }, 1000);
            }
            
        } else {
            this.log('Распознавание речи не поддерживается в этом браузере');
            this.updateStatus('Распознавание речи не поддерживается');
            this.updateSpeechStatus('не поддерживается');
        }
    }

    updateSpeechStatus(listening) {
        const speechStatus = document.getElementById('speech-status');
        if (speechStatus) {
            if (listening) {
                speechStatus.textContent = '🎤 Распознавание: слушаю...';
                speechStatus.style.color = '#4cc9f0';
            } else {
                speechStatus.textContent = '🎤 Распознавание: выкл';
                speechStatus.style.color = '#95a5a6';
            }
        }
    }

    // DRAWING MANAGEMENT
    setupDrawingCanvas() {
        const resizeCanvas = () => {
            const drawingCanvas = document.getElementById('drawing-canvas');
            drawingCanvas.width = drawingCanvas.offsetWidth;
            drawingCanvas.height = drawingCanvas.offsetHeight;
            
            if (this.drawingCtx) {
                this.drawingCtx.lineWidth = this.isEraser ? 20 : 3;
                this.drawingCtx.lineCap = 'round';
                this.drawingCtx.lineJoin = 'round';
                this.drawingCtx.strokeStyle = this.isEraser ? '#F5F6F8' : this.currentColor;
            }
        };
        
        resizeCanvas();
        
        this.drawingCtx = document.getElementById('drawing-canvas').getContext('2d');
        this.drawingCtx.lineWidth = 3;
        this.drawingCtx.lineCap = 'round';
        this.drawingCtx.lineJoin = 'round';
        this.drawingCtx.strokeStyle = this.currentColor;
        
        if (this.isIOS) {
            const drawingCanvas = document.getElementById('drawing-canvas');
            drawingCanvas.style.touchAction = 'none';
            drawingCanvas.style.webkitUserSelect = 'none';
            drawingCanvas.style.webkitTouchCallout = 'none';
        }
        
        const drawingCanvas = document.getElementById('drawing-canvas');
        drawingCanvas.addEventListener('mousedown', this.startDrawing.bind(this));
        drawingCanvas.addEventListener('mousemove', this.draw.bind(this));
        drawingCanvas.addEventListener('mouseup', this.stopDrawing.bind(this));
        drawingCanvas.addEventListener('mouseout', this.stopDrawing.bind(this));
        
        drawingCanvas.addEventListener('touchstart', this.handleTouchStart.bind(this), { passive: false });
        drawingCanvas.addEventListener('touchmove', this.handleTouchMove.bind(this), { passive: false });
        drawingCanvas.addEventListener('touchend', this.handleTouchEnd.bind(this));
        
        window.addEventListener('resize', resizeCanvas);
    }

    isDrawingWindowVisible() {
        return this.isDrawingWindowOpen && 
               document.getElementById('drawing-window').style.display === 'block' &&
               document.getElementById('drawing-window').offsetWidth > 0 &&
               document.getElementById('drawing-window').offsetHeight > 0;
    }

    getCanvasCoordinates(e) {
        const drawingCanvas = document.getElementById('drawing-canvas');
        const rect = drawingCanvas.getBoundingClientRect();
        const scaleX = drawingCanvas.width / rect.width;
        const scaleY = drawingCanvas.height / rect.height;
        
        let clientX, clientY;
        
        if (e.type.includes('touch')) {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
        } else {
            clientX = e.clientX;
            clientY = e.clientY;
        }
        
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    startDrawing(e) {
        if (!this.isDrawingWindowVisible()) return;
        
        this.isDrawing = true;
        const coords = this.getCanvasCoordinates(e);
        this.lastX = coords.x;
        this.lastY = coords.y;
        
        this.drawingCtx.beginPath();
        this.drawingCtx.moveTo(this.lastX, this.lastY);
        
        e.preventDefault();
    }

    draw(e) {
        if (!this.isDrawing) return;
        
        const coords = this.getCanvasCoordinates(e);
        const currentX = coords.x;
        const currentY = coords.y;
        
        this.drawingCtx.lineTo(currentX, currentY);
        this.drawingCtx.stroke();
        
        this.lastX = currentX;
        this.lastY = currentY;
        
        e.preventDefault();
    }

    stopDrawing() {
        if (!this.isDrawing) return;
        
        this.isDrawing = false;
        this.drawingCtx.closePath();
    }

    handleTouchStart(e) {
        if (!this.isDrawingWindowVisible()) return;
        
        e.preventDefault();
        
        const touch = e.touches[0];
        const customEvent = {
            type: 'mousedown',
            clientX: touch.clientX,
            clientY: touch.clientY,
            target: document.getElementById('drawing-canvas')
        };
        
        this.startDrawing(customEvent);
    }

    handleTouchMove(e) {
        if (!this.isDrawing || !this.isDrawingWindowVisible()) return;
        
        e.preventDefault();
        
        const touch = e.touches[0];
        const customEvent = {
            type: 'mousemove',
            clientX: touch.clientX,
            clientY: touch.clientY,
            target: document.getElementById('drawing-canvas')
        };
        
        this.draw(customEvent);
    }

    handleTouchEnd() {
        if (!this.isDrawing) return;
        
        this.stopDrawing();
    }

    clearDrawing() {
        this.drawingCtx.clearRect(0, 0, document.getElementById('drawing-canvas').width, document.getElementById('drawing-canvas').height);
    }

    setDrawingTool(tool) {
        this.currentTool = tool;
        this.isEraser = (tool === 'eraser');
        
        document.getElementById('drawing-pen').classList.remove('active');
        document.getElementById('drawing-eraser').classList.remove('active');
        
        if (tool === 'pen') {
            document.getElementById('drawing-pen').classList.add('active');
            this.drawingCtx.lineWidth = 3;
            this.drawingCtx.strokeStyle = this.currentColor;
            this.drawingCtx.globalCompositeOperation = 'source-over';
        } else if (tool === 'eraser') {
            document.getElementById('drawing-eraser').classList.add('active');
            this.drawingCtx.lineWidth = 20;
            this.drawingCtx.strokeStyle = '#F5F6F8';
            this.drawingCtx.globalCompositeOperation = 'destination-out';
        }
    }

    // WINDOW MANAGEMENT
    setupWindowDragging() {
        const initDragForWindow = (windowElement) => {
            const header = windowElement.querySelector('.window-header');
            
            header.addEventListener('mousedown', this.startDragWindow.bind(this));
            header.addEventListener('touchstart', this.startDragWindow.bind(this), { passive: false });
            
            header.addEventListener('touchstart', (e) => {
                e.stopPropagation();
            });
            header.addEventListener('mousedown', (e) => {
                e.stopPropagation();
            });
        };
        
        initDragForWindow(document.getElementById('lesson-content'));
        initDragForWindow(document.getElementById('drawing-window'));
    }

    startDragWindow(e) {
        this.isDraggingWindow = true;
        this.currentDraggedWindow = e.currentTarget.parentElement;
        
        const rect = this.currentDraggedWindow.getBoundingClientRect();
        if (e.type === 'mousedown') {
            this.dragOffsetX = e.clientX - rect.left;
            this.dragOffsetY = e.clientY - rect.top;
        } else {
            this.dragOffsetX = e.touches[0].clientX - rect.left;
            this.dragOffsetY = e.touches[0].clientY - rect.top;
            e.preventDefault();
        }
        
        document.addEventListener('mousemove', this.dragWindow.bind(this));
        document.addEventListener('touchmove', this.dragWindow.bind(this), { passive: false });
        document.addEventListener('mouseup', this.stopDragWindow.bind(this));
        document.addEventListener('touchend', this.stopDragWindow.bind(this));
    }

    dragWindow(e) {
        if (!this.isDraggingWindow || !this.currentDraggedWindow) return;
        
        let clientX, clientY;
        if (e.type === 'mousemove') {
            clientX = e.clientX;
            clientY = e.clientY;
        } else {
            clientX = e.touches[0].clientX;
            clientY = e.touches[0].clientY;
            e.preventDefault();
        }
        
        const newX = clientX - this.dragOffsetX;
        const newY = clientY - this.dragOffsetY;
        
        const maxX = window.innerWidth - this.currentDraggedWindow.offsetWidth;
        const maxY = window.innerHeight - this.currentDraggedWindow.offsetHeight;
        
        this.currentDraggedWindow.style.left = Math.max(0, Math.min(newX, maxX)) + 'px';
        this.currentDraggedWindow.style.top = Math.max(0, Math.min(newY, maxY)) + 'px';
        this.currentDraggedWindow.style.transform = 'none';
    }

    stopDragWindow() {
        this.isDraggingWindow = false;
        this.currentDraggedWindow = null;
        document.removeEventListener('mousemove', this.dragWindow.bind(this));
        document.removeEventListener('touchmove', this.dragWindow.bind(this));
        document.removeEventListener('mouseup', this.stopDragWindow.bind(this));
        document.removeEventListener('touchend', this.stopDragWindow.bind(this));
    }

    setTool(tool) {
        this.isDrawingWindowOpen = false;
        
        document.getElementById('lesson-content').style.display = 'none';
        document.getElementById('drawing-window').style.display = 'none';
        document.getElementById('answer-input-container').style.display = 'none';
        
        document.getElementById('text-tool').classList.remove('active');
        document.getElementById('draw-tool').classList.remove('active');
        document.getElementById('answer-tool').classList.remove('active');
        
        if (tool === 'text') {
            document.getElementById('text-tool').classList.add('active');
            document.getElementById('lesson-content').style.display = 'block';
        } else if (tool === 'draw') {
            document.getElementById('draw-tool').classList.add('active');
            document.getElementById('drawing-window').style.display = 'block';
            this.isDrawingWindowOpen = true;
            
            setTimeout(() => {
                const drawingCanvas = document.getElementById('drawing-canvas');
                const rect = drawingCanvas.getBoundingClientRect();
                drawingCanvas.width = rect.width;
                drawingCanvas.height = rect.height;
            }, 50);
        } else if (tool === 'answer') {
            document.getElementById('answer-tool').classList.add('active');
            document.getElementById('lesson-content').style.display = 'block';
            document.getElementById('answer-input-container').style.display = 'block';
            document.getElementById('answer-input').focus();
        }
    }

    submitAnswer() {
        const answer = document.getElementById('answer-input').value.trim();
        if (answer) {
            if (this.socket) {
                this.socket.emit('recognized_speech', {
                    room_id: this.roomId,
                    text: answer
                });
            }
            document.getElementById('answer-input').value = '';
            this.setTool('text');
            this.showStudentSpeech(answer, 'me');
        }
    }

    setupZoom() {
        let lessonScale = 1;
        const zoomInBtn = document.getElementById('zoom-in');
        const zoomOutBtn = document.getElementById('zoom-out');
        const zoomResetBtn = document.getElementById('zoom-reset');
        
        zoomInBtn.addEventListener('click', () => {
            lessonScale = Math.min(lessonScale + 0.1, 2);
            this.updateLessonZoom(lessonScale);
        });
        
        zoomOutBtn.addEventListener('click', () => {
            lessonScale = Math.max(lessonScale - 0.1, 0.5);
            this.updateLessonZoom(lessonScale);
        });
        
        zoomResetBtn.addEventListener('click', () => {
            lessonScale = 1;
            this.updateLessonZoom(lessonScale);
        });
        
        let drawingScale = 1;
        const drawingZoomInBtn = document.getElementById('drawing-zoom-in');
        const drawingZoomOutBtn = document.getElementById('drawing-zoom-out');
        const drawingZoomResetBtn = document.getElementById('drawing-zoom-reset');
        
        drawingZoomInBtn.addEventListener('click', () => {
            drawingScale = Math.min(drawingScale + 0.1, 2);
            this.updateDrawingZoom(drawingScale);
        });
        
        drawingZoomOutBtn.addEventListener('click', () => {
            drawingScale = Math.max(drawingScale - 0.1, 0.5);
            this.updateDrawingZoom(drawingScale);
        });
        
        drawingZoomResetBtn.addEventListener('click', () => {
            drawingScale = 1;
            this.updateDrawingZoom(drawingScale);
        });
    }

    updateLessonZoom(scale) {
        document.getElementById('lesson-canvas').style.transform = `scale(${scale})`;
        document.getElementById('lesson-canvas').style.transformOrigin = 'center center';
    }

    updateDrawingZoom(scale) {
        document.getElementById('drawing-canvas').style.transform = `scale(${scale})`;
        document.getElementById('drawing-canvas').style.transformOrigin = 'center center';
    }

    // LESSON MANAGEMENT
    loadLessonContent(lessonId) {
        // Временный контент для демонстрации
        this.currentLessonContent = [
            "Добро пожаловать на урок!",
            "Сегодня мы изучим основы работы с виртуальным классом.",
            "Вы можете использовать текстовую доску для заметок и отдельное окно для рисования.",
            "AI-учитель поможет вам освоить материал и ответит на вопросы.",
            "Не стесняйтесь задавать вопросы - это поможет лучше понять тему."
        ];
        this.displayLessonParagraph(0);
    }

    displayLessonParagraph(paragraphIndex) {
        if (paragraphIndex < this.currentLessonContent.length) {
            const paragraph = this.currentLessonContent[paragraphIndex];
            
            const lessonText = document.getElementById('lesson-text');
            if (lessonText.textContent !== '') {
                lessonText.classList.add('slide-out');
                setTimeout(() => {
                    lessonText.classList.remove('slide-out');
                    lessonText.textContent = '';
                    this.animateTextTyping(paragraph);
                }, 500);
            } else {
                this.animateTextTyping(paragraph);
            }
            
            this.currentParagraph = paragraphIndex + 1;
        } else {
            document.getElementById('lesson-content').style.display = 'none';
            this.currentLesson = null;
            this.isLessonActive = false;
        }
    }

    updateLessonText(text) {
        if (this.currentLesson) {
            const paragraphIndex = this.currentLessonContent.indexOf(text);
            if (paragraphIndex !== -1 && paragraphIndex >= this.currentParagraph) {
                this.displayLessonParagraph(paragraphIndex);
            } else if (paragraphIndex === -1) {
                this.animateTextTyping(text);
            }
        }
    }

    animateTextTyping(text) {
        document.getElementById('lesson-content').style.display = 'block';
        
        if (this.currentTypingAnimation) {
            clearInterval(this.currentTypingAnimation);
        }
        
        const lessonText = document.getElementById('lesson-text');
        lessonText.textContent = '';
        let currentIndex = 0;
        const typingSpeed = 30;
        
        this.currentTypingAnimation = setInterval(() => {
            if (currentIndex < text.length) {
                lessonText.textContent += text.charAt(currentIndex);
                currentIndex++;
                document.getElementById('lesson-canvas').scrollTop = document.getElementById('lesson-canvas').scrollHeight;
            } else {
                clearInterval(this.currentTypingAnimation);
                const cursor = document.querySelector('.typing-cursor');
                if (cursor) cursor.remove();
            }
        }, typingSpeed);
        
        const cursor = document.createElement('span');
        cursor.className = 'typing-cursor';
        lessonText.appendChild(cursor);
    }

    // UTILITIES
    showTeacherSpeech(text) {
        const cleanText = text.replace('Учитель: ', '');
        const teacherSpeech = document.getElementById('teacher-speech');
        teacherSpeech.textContent = cleanText;
        teacherSpeech.style.display = 'block';
        
        setTimeout(() => {
            teacherSpeech.style.display = 'none';
        }, 5000);
    }

    showStudentSpeech(text, sid) {
        const studentSpeech = document.getElementById('student-speech');
        studentSpeech.textContent = text;
        studentSpeech.style.display = 'block';
        
        setTimeout(() => {
            studentSpeech.style.display = 'none';
        }, 5000);
    }

    closeLesson() {
        document.getElementById('lesson-content').style.display = 'none';
        document.getElementById('text-tool').classList.remove('active');
        document.getElementById('answer-tool').classList.remove('active');
    }

    closeDrawing() {
        document.getElementById('drawing-window').style.display = 'none';
        this.isDrawingWindowOpen = false;
        document.getElementById('draw-tool').classList.remove('active');
    }

    closeTextWidget() {
        document.getElementById('main-text-widget').style.display = 'none';
    }

    clearText() {
        document.getElementById('lesson-text').textContent = '';
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    log(message) {
        console.log(message);
    }

    // iOS Audio Fix
    ensureAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
    }
}

// ИНИЦИАЛИЗАЦИЯ И GLOBAL EVENT LISTENERS
document.addEventListener('DOMContentLoaded', () => {
    window.conference = new Conference();
    
    // iOS Audio Context activation
    if (/iPad|iPhone|iPod/.test(navigator.userAgent) && !window.MSStream) {
        document.addEventListener('touchstart', function() {
            if (!window.conference.audioContext) {
                window.conference.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            }
        }, { once: true });
    }
    
    // Fullscreen change listener
    document.addEventListener('fullscreenchange', () => {
        window.conference.isFullscreen = !!document.fullscreenElement;
        document.getElementById('fullscreen-btn').querySelector('span').textContent = 
            window.conference.isFullscreen ? 'Обычный' : 'Полный';
    });
    
    // Message listener for embedded mode
    window.addEventListener('message', (event) => {
        if (event.data === 'end_conference') {
            window.conference.endCall();
        }
    });
    
    // Resize handler for background animation
    window.addEventListener('resize', () => {
        const backgroundAnimation = document.getElementById('background-animation');
        if (backgroundAnimation) {
            backgroundAnimation.innerHTML = '';
            window.conference.createBackgroundAnimation();
        }
    });
});
