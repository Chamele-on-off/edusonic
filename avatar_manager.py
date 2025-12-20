# avatar_manager.py
import zipfile
import shutil
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import logging
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)

class AvatarManager:
    def __init__(self, frames_dir: Path):
        self.frames_dir = frames_dir
        self.supported_formats = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    
    def validate_avatar_archive(self, zip_path: Path) -> Tuple[bool, str, List[str]]:
        """Проверяет zip-архив с аватаром"""
        try:
            if not zip_path.exists():
                return False, "Файл не найден", []
            
            frame_files = []
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Проверяем структуру архива
                file_list = zipf.namelist()
                
                # Ищем файлы изображений
                for filename in file_list:
                    if not filename.startswith('__') and not filename.startswith('.') and filename.lower().endswith(self.supported_formats):
                        frame_files.append(filename)
                
                if not frame_files:
                    return False, "В архиве не найдено файлов изображений", []
                
                # Проверяем, что файлы находятся в корне или в одной папке
                has_subfolders = any('/' in f for f in frame_files)
                if has_subfolders:
                    # Проверяем, что все файлы в одной папке
                    folders = set(os.path.dirname(f) for f in frame_files if '/' in f)
                    if len(folders) > 1:
                        return False, "Изображения должны быть в одной папке", []
                
                logger.info(f"Найдено {len(frame_files)} кадров в архиве")
                return True, "Архив валиден", frame_files
                
        except zipfile.BadZipFile:
            return False, "Некорректный ZIP архив", []
        except Exception as e:
            return False, f"Ошибка проверки архива: {str(e)}", []
    
    def extract_avatar(self, zip_path: Path, avatar_name: str, force_overwrite: bool = False) -> Tuple[bool, str, str]:
        """Извлекает аватар из архива"""
        try:
            avatar_dir = self.frames_dir / avatar_name
            
            # Проверяем, существует ли уже такой аватар
            if avatar_dir.exists():
                if not force_overwrite:
                    return False, f"Аватар '{avatar_name}' уже существует", ""
                # Создаем резервную копию
                backup_dir = self.frames_dir / f"{avatar_name}_backup_{int(time.time())}"
                shutil.move(avatar_dir, backup_dir)
                logger.info(f"Создана резервная копия: {backup_dir}")
            
            # Создаем папку для аватара
            avatar_dir.mkdir(parents=True, exist_ok=True)
            
            # Валидируем архив
            is_valid, message, frame_files = self.validate_avatar_archive(zip_path)
            if not is_valid:
                # Удаляем созданную папку
                if avatar_dir.exists():
                    shutil.rmtree(avatar_dir, ignore_errors=True)
                return False, message, ""
            
            # Извлекаем файлы
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Определяем базовую папку (если файлы находятся в подпапке)
                sample_file = frame_files[0]
                base_folder = os.path.dirname(sample_file) if '/' in sample_file else ""
                
                # Извлекаем файлы
                for filename in frame_files:
                    try:
                        # Безопасное имя файла
                        safe_filename = secure_filename(os.path.basename(filename))
                        if not safe_filename:
                            continue
                        
                        # Извлекаем файл
                        source = zipf.read(filename)
                        target_path = avatar_dir / safe_filename
                        
                        with open(target_path, 'wb') as f:
                            f.write(source)
                        
                    except Exception as e:
                        logger.error(f"Ошибка извлечения файла {filename}: {e}")
                        continue
            
            # Проверяем, что файлы были извлечены
            extracted_files = list(avatar_dir.glob("*"))
            if not extracted_files:
                shutil.rmtree(avatar_dir, ignore_errors=True)
                return False, "Не удалось извлечь файлы из архива", ""
            
            # Сортируем файлы для удобства
            self.sort_avatar_frames(avatar_dir)
            
            return True, f"Аватар '{avatar_name}' успешно установлен ({len(extracted_files)} кадров)", str(avatar_dir)
            
        except Exception as e:
            logger.error(f"Ошибка извлечения аватара: {e}")
            # Очищаем в случае ошибки
            if avatar_dir.exists():
                shutil.rmtree(avatar_dir, ignore_errors=True)
            return False, f"Ошибка извлечения: {str(e)}", ""
    
    def sort_avatar_frames(self, avatar_dir: Path):
        """Сортирует кадры аватара по имени"""
        try:
            frames = list(avatar_dir.iterdir())
            frames.sort(key=lambda x: x.name.lower())
            
            # Если файлы не имеют последовательных имен, можно их перенумеровать
            has_numbers = any(re.search(r'\d+', f.name) for f in frames)
            if not has_numbers and len(frames) > 1:
                # Перенумеровываем файлы
                for i, frame in enumerate(sorted(frames, key=lambda x: x.name.lower())):
                    ext = frame.suffix
                    new_name = f"frame_{i+1:04d}{ext}"
                    frame.rename(avatar_dir / new_name)
        except Exception as e:
            logger.error(f"Ошибка сортировки кадров: {e}")
    
    def delete_avatar(self, avatar_name: str) -> Tuple[bool, str]:
        """Удаляет аватар"""
        try:
            avatar_dir = self.frames_dir / avatar_name
            
            if not avatar_dir.exists():
                return False, f"Аватар '{avatar_name}' не найден"
            
            # Создаем резервную копию в корзину
            trash_dir = self.frames_dir.parent / "trash" / "avatars"
            trash_dir.mkdir(parents=True, exist_ok=True)
            backup_dir = trash_dir / f"{avatar_name}_{int(time.time())}"
            
            shutil.move(avatar_dir, backup_dir)
            
            return True, f"Аватар '{avatar_name}' перемещен в корзину"
            
        except Exception as e:
            logger.error(f"Ошибка удаления аватара: {e}")
            return False, f"Ошибка удаления: {str(e)}"
    
    def get_avatar_stats(self, avatar_name: str) -> Optional[dict]:
        """Получает статистику аватара"""
        try:
            avatar_dir = self.frames_dir / avatar_name
            
            if not avatar_dir.exists():
                return None
            
            frames = list(avatar_dir.iterdir())
            frames = [f for f in frames if f.is_file() and f.suffix.lower() in self.supported_formats]
            frames.sort(key=lambda x: x.name.lower())
            
            if not frames:
                return None
            
            # Анализируем файлы
            total_size = sum(f.stat().st_size for f in frames)
            formats = set(f.suffix.lower() for f in frames)
            dimensions = {}
            
            # Получаем размеры первого изображения
            if frames:
                try:
                    from PIL import Image
                    with Image.open(frames[0]) as img:
                        dimensions = {'width': img.width, 'height': img.height}
                except:
                    dimensions = {'width': 'unknown', 'height': 'unknown'}
            
            return {
                'name': avatar_name,
                'frames_count': len(frames),
                'total_size': total_size,
                'formats': list(formats),
                'dimensions': dimensions,
                'frame_names': [f.name for f in frames[:10]],  # Первые 10 имен
                'first_frame': f"/frames/{avatar_name}/{frames[0].name}" if frames else None
            }
            
        except Exception as e:
            logger.error(f"Ошибка получения статистики аватара: {e}")
            return None
    
    def list_avatars(self) -> List[dict]:
        """Список всех аватаров с информацией"""
        avatars = []
        
        for avatar_dir in self.frames_dir.iterdir():
            if avatar_dir.is_dir():
                stats = self.get_avatar_stats(avatar_dir.name)
                if stats:
                    avatars.append(stats)
        
        # Сортируем по имени
        avatars.sort(key=lambda x: x['name'].lower())
        return avatars
    
    def rename_avatar(self, old_name: str, new_name: str) -> Tuple[bool, str]:
        """Переименовывает аватар"""
        try:
            old_dir = self.frames_dir / old_name
            new_dir = self.frames_dir / new_name
            
            if not old_dir.exists():
                return False, f"Аватар '{old_name}' не найден"
            
            if new_dir.exists():
                return False, f"Аватар '{new_name}' уже существует"
            
            old_dir.rename(new_dir)
            return True, f"Аватар '{old_name}' переименован в '{new_name}'"
            
        except Exception as e:
            return False, f"Ошибка переименования: {str(e)}"
