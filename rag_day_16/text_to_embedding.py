import os
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

class YandexDocumentIndexer:
    """Класс для создания индекса документов с эмбеддингами YandexGPT"""
    
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Инициализация индексатора
        
        Args:
            folder_id: ID папки Yandex Cloud
            api_key: API ключ Yandex Cloud
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        print("Инициализация YandexGPT SDK...")
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        print("SDK успешно инициализирован!")
        
        # Статистика токенов
        self.total_tokens = 0
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 500, 
                               overlap: int = 50) -> List[str]:
        """
        Разбивает текст на чанки с перекрытием
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка в символах
            overlap: Размер перекрытия между чанками
            
        Returns:
            Список текстовых чанков
        """
        # Очистка текста от лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Определяем конец чанка
            end = start + chunk_size
            
            # Если это не последний чанк, пытаемся разбить по предложению
            if end < len(text):
                # Ищем конец предложения в окне
                sentence_end = max(
                    text.rfind('. ', start, end),
                    text.rfind('! ', start, end),
                    text.rfind('? ', start, end),
                    text.rfind('\n', start, end)
                )
                
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Сдвигаемся с учетом перекрытия
            start = end - overlap if end < len(text) else end
        
        return chunks
    
    def generate_embeddings(self, chunks: List[str], 
                          model: str = "text-search-doc") -> List[List[float]]:
        """
        Генерирует эмбеддинги для списка чанков через YandexGPT
        
        Args:
            chunks: Список текстовых чанков
            model: Модель для эмбеддингов (text-search-doc или text-search-query)
            
        Returns:
            Список эмбеддингов (векторов)
        """
        print(f"Генерация эмбеддингов для {len(chunks)} чанков...")
        
        embeddings = []
        
        try:
            # Получаем модель эмбеддингов
            embedder = self.sdk.models.text_embeddings(model)
            
            for i, chunk in enumerate(chunks):
                try:
                    # Генерируем эмбеддинг для чанка
                    result = embedder.run(chunk)
                    
                    # Извлекаем вектор эмбеддинга
                    embedding_vector = result.embedding
                    embeddings.append(embedding_vector)
                    
                    # Обновляем статистику токенов
                    if hasattr(result, 'usage') and hasattr(result.usage, 'total_tokens'):
                        self.total_tokens += result.usage.total_tokens
                    
                    if (i + 1) % 10 == 0:
                        print(f"  Обработано: {i + 1}/{len(chunks)} чанков")
                
                except Exception as e:
                    print(f"⚠️ Ошибка при генерации эмбеддинга для чанка {i}: {str(e)}")
                    # Добавляем пустой вектор в случае ошибки
                    embeddings.append([])
            
            print(f"✅ Генерация завершена! Использовано токенов: {self.total_tokens}")
            
        except Exception as e:
            print(f"❌ Критическая ошибка при генерации эмбеддингов: {str(e)}")
            raise
        
        return embeddings
    
    def create_index(self, text: str, chunk_size: int = 500, 
                    overlap: int = 50, model: str = "text-search-doc") -> Dict:
        """
        Создает полный индекс с чанками и эмбеддингами
        
        Args:
            text: Исходный текст
            chunk_size: Размер чанка
            overlap: Размер перекрытия
            model: Модель для эмбеддингов
            
        Returns:
            Словарь с индексом
        """
        # Разбивка на чанки
        print("Разбивка текста на чанки...")
        chunks = self.split_text_into_chunks(text, chunk_size, overlap)
        print(f"Создано {len(chunks)} чанков")
        
        # Генерация эмбеддингов
        embeddings = self.generate_embeddings(chunks, model)
        
        # Создание индекса
        index = {
            'metadata': {
                'total_chunks': len(chunks),
                'chunk_size': chunk_size,
                'overlap': overlap,
                'model': model,
                'embedding_dimension': len(embeddings[0]) if embeddings and embeddings[0] else 0,
                'total_tokens_used': self.total_tokens
            },
            'documents': [
                {
                    'id': i,
                    'text': chunk,
                    'embedding': embedding,
                    'char_start': sum(len(c) for c in chunks[:i]),
                    'char_end': sum(len(c) for c in chunks[:i+1])
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
        }
        
        return index
    
    def save_index(self, index: Dict, output_path: str):
        """
        Сохраняет индекс в JSON файл
        
        Args:
            index: Индекс для сохранения
            output_path: Путь к выходному файлу
        """
        print(f"Сохранение индекса в {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        print("✅ Индекс успешно сохранен!")
    
    def load_index(self, input_path: str) -> Dict:
        """
        Загружает индекс из JSON файла
        
        Args:
            input_path: Путь к файлу с индексом
            
        Returns:
            Загруженный индекс
        """
        print(f"Загрузка индекса из {input_path}...")
        with open(input_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        print("✅ Индекс успешно загружен!")
        return index
    
    def process_file(self, input_filename: str, chunk_size: int = 500, 
                    overlap: int = 50, model: str = "text-search-doc") -> str:
        """
        Обрабатывает файл: читает, создает индекс и сохраняет
        
        Args:
            input_filename: Имя входного текстового файла
            chunk_size: Размер чанка
            overlap: Размер перекрытия
            model: Модель для эмбеддингов
            
        Returns:
            Путь к сохраненному файлу индекса
        """
        # Формируем имя выходного файла
        base_name = os.path.splitext(input_filename)[0]
        output_filename = f"{base_name}_index.json"
        
        print(f"\n{'='*60}")
        print(f"📄 Обработка файла: {input_filename}")
        print(f"💾 Результат будет сохранен в: {output_filename}")
        print(f"{'='*60}\n")
        
        # Чтение файла
        try:
            with open(input_filename, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"✅ Загружено {len(text)} символов из {input_filename}\n")
        except FileNotFoundError:
            raise FileNotFoundError(f"Файл {input_filename} не найден")
        except Exception as e:
            raise Exception(f"Ошибка при чтении файла: {str(e)}")
        
        # Создание индекса
        index = self.create_index(text, chunk_size, overlap, model)
        
        # Сохранение индекса
        self.save_index(index, output_filename)
        
        # Вывод статистики
        print(f"\n{'='*60}")
        print("📊 СТАТИСТИКА ИНДЕКСА")
        print(f"{'='*60}")
        print(f"Всего чанков: {index['metadata']['total_chunks']}")
        print(f"Размерность эмбеддингов: {index['metadata']['embedding_dimension']}")
        print(f"Размер чанка: {index['metadata']['chunk_size']} символов")
        print(f"Перекрытие: {index['metadata']['overlap']} символов")
        print(f"Использовано токенов: {index['metadata']['total_tokens_used']}")
        print(f"Модель: {index['metadata']['model']}")
        
        if index['documents']:
            print(f"\nПример первого чанка:")
            first_text = index['documents'][0]['text']
            print(first_text[:200] + ("..." if len(first_text) > 200 else ""))
            
            if index['documents'][0]['embedding']:
                print(f"\nРазмер эмбеддинга: {len(index['documents'][0]['embedding'])} чисел")
                print(f"Первые 5 значений: {index['documents'][0]['embedding'][:5]}")
        
        print(f"{'='*60}\n")
        
        return output_filename


def main():
    """Пример использования"""
    
    # Имя входного файла
    input_file = 'text_to_test.txt'
    
    # Проверяем существование файла или создаем демо-файл
    if not os.path.exists(input_file):
        print(f"⚠️ Файл {input_file} не найден. Создаю демо-файл...")
        demo_text = """
        Искусственный интеллект - это область компьютерных наук, которая занимается 
        созданием интеллектуальных машин. Эти машины способны выполнять задачи, 
        которые обычно требуют человеческого интеллекта.
        """
        
        with open(input_file, 'w', encoding='utf-8') as f:
            f.write(demo_text)
        print(f"✅ Создан демо-файл {input_file}\n")
    
    try:
        # Создание индексатора
        indexer = YandexDocumentIndexer()
        
        # Обработка файла
        output_file = indexer.process_file(
            input_filename=input_file,
            chunk_size=500,      # Размер чанка в символах
            overlap=50,          # Перекрытие между чанками
            model="text-search-doc"  # Модель для эмбеддингов документов
        )
        
        print(f"🎉 Готово! Индекс сохранен в: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")


if __name__ == "__main__":
    main()