import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

class YandexRAGSystem:
    """RAG система: поиск релевантных чанков + генерация ответа через YandexGPT"""
    
    def __init__(self, index_path: str, folder_id: Optional[str] = None, 
                 api_key: Optional[str] = None):
        """
        Инициализация RAG системы
        
        Args:
            index_path: Путь к файлу с индексом
            folder_id: ID папки Yandex Cloud
            api_key: API ключ Yandex Cloud
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        print("🚀 Инициализация RAG системы...")
        print("Подключение к YandexGPT SDK...")
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        print("✅ SDK успешно инициализирован!\n")
        
        # Загрузка индекса
        self.index = self._load_index(index_path)
        self.documents = self.index['documents']
        
        print(f"📚 Загружено документов: {len(self.documents)}")
        print(f"📊 Размерность эмбеддингов: {self.index['metadata']['embedding_dimension']}\n")
    
    def _load_index(self, index_path: str) -> Dict:
        """Загружает индекс из JSON файла"""
        print(f"📂 Загрузка индекса из {index_path}...")
        with open(index_path, 'r', encoding='utf-8') as f:
            index = json.load(f)
        print("✅ Индекс успешно загружен!")
        return index
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Вычисляет косинусное сходство между двумя векторами
        
        Args:
            vec1: Первый вектор
            vec2: Второй вектор
            
        Returns:
            Косинусное сходство (от -1 до 1)
        """
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        # Нормализация и вычисление скалярного произведения
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1_np, vec2_np) / (norm1 * norm2))
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Генерирует эмбеддинг для поискового запроса
        
        Args:
            query: Текст запроса
            
        Returns:
            Вектор эмбеддинга запроса
        """
        print(f"🔍 Генерация эмбеддинга для запроса...")
        
        # Используем модель для запросов (text-search-query)
        embedder = self.sdk.models.text_embeddings("text-search-query")
        result = embedder.run(query)
        
        print("✅ Эмбеддинг запроса создан!")
        return result.embedding
    
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Ищет наиболее релевантные чанки для запроса
        
        Args:
            query: Поисковый запрос
            top_k: Количество топ результатов
            
        Returns:
            Список релевантных чанков с метаданными
        """
        print(f"\n{'='*60}")
        print(f"🔎 ПОИСК РЕЛЕВАНТНЫХ ЧАНКОВ")
        print(f"{'='*60}")
        print(f"Запрос: {query}")
        print(f"Ищем топ-{top_k} релевантных фрагментов...\n")
        
        # Генерируем эмбеддинг для запроса
        query_embedding = self.generate_query_embedding(query)
        
        # Вычисляем сходство со всеми документами
        similarities = []
        for doc in self.documents:
            if not doc['embedding']:  # Пропускаем пустые эмбеддинги
                continue
            
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            similarities.append({
                'id': doc['id'],
                'text': doc['text'],
                'similarity': similarity,
                'char_start': doc['char_start'],
                'char_end': doc['char_end'],
                'metadata': {}
            })
        
        # Сортируем по убыванию сходства
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Возвращаем топ-k результатов
        top_results = similarities[:top_k]
        
        print("📊 Найденные релевантные фрагменты:")
        print(f"{'='*60}\n")
        
        for i, result in enumerate(top_results, 1):
            print(f"🔹 Результат #{i}")
            print(f"   Релевантность: {result['similarity']:.4f}")
            print(f"   ID чанка: {result['id']}")
            print(f"   Текст: {result['text'][:150]}...")
            print()
        
        return top_results
    
    def generate_answer(self, query: str, context_chunks: List[Dict], 
                       model: str = "yandexgpt", temperature: float = 0.3,
                       max_tokens: int = 2000) -> Dict:
        """
        Генерирует ответ на основе запроса и контекста
        
        Args:
            query: Вопрос пользователя
            context_chunks: Релевантные чанки контекста
            model: Модель YandexGPT (yandexgpt, yandexgpt-lite)
            temperature: Температура генерации (0-1)
            max_tokens: Максимальное количество токенов в ответе
            
        Returns:
            Словарь с ответом и метаданными
        """
        print(f"{'='*60}")
        print(f"🤖 ГЕНЕРАЦИЯ ОТВЕТА")
        print(f"{'='*60}\n")
        
        # Формируем контекст из релевантных чанков
        context = "\n\n".join([
            f"Фрагмент {i+1} (релевантность: {chunk['similarity']:.4f}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Создаем промпт
        prompt = f"""На основе предоставленного контекста ответь на вопрос пользователя.

КОНТЕКСТ:
{context}

ВОПРОС:
{query}

ИНСТРУКЦИИ:
- Используй только информацию из контекста
- Если в контексте нет ответа, честно скажи об этом
- Отвечай четко и по существу
- Приводи цитаты из контекста, если это уместно

ОТВЕТ:"""
        
        print(f"📝 Промпт для LLM:")
        print(f"   Длина контекста: {len(context)} символов")
        print(f"   Количество фрагментов: {len(context_chunks)}")
        print(f"   Модель: {model}")
        print(f"   Temperature: {temperature}\n")
        
        # Получаем модель
        llm_model = self.sdk.models.completions(model)
        
        # Настройки генерации
        llm_model = llm_model.configure(temperature=temperature, max_tokens=max_tokens)
        
        # Генерируем ответ
        print("⏳ Генерация ответа...")
        result = llm_model.run(prompt)
        
        # Извлекаем текст ответа
        answer_text = result.alternatives[0].text
        
        # Собираем статистику
        usage_stats = {
            'prompt_tokens': result.usage.input_text_tokens if hasattr(result, 'usage') else 0,
            'completion_tokens': result.usage.completion_tokens if hasattr(result, 'usage') else 0,
            'total_tokens': result.usage.total_tokens if hasattr(result, 'usage') else 0
        }
        
        print(f"✅ Ответ получен!")
        print(f"   Токенов использовано: {usage_stats['total_tokens']}")
        print(f"   - Промпт: {usage_stats['prompt_tokens']}")
        print(f"   - Ответ: {usage_stats['completion_tokens']}\n")
        
        return {
            'answer': answer_text,
            'context_chunks': context_chunks,
            'usage': usage_stats,
            'model': model,
            'temperature': temperature
        }
    
    def ask(self, query: str, top_k: int = 3, model: str = "yandexgpt",
            temperature: float = 0.3, max_tokens: int = 2000) -> Dict:
        """
        Полный цикл RAG: вопрос → поиск → генерация ответа
        
        Args:
            query: Вопрос пользователя
            top_k: Количество релевантных чанков
            model: Модель YandexGPT
            temperature: Температура генерации
            max_tokens: Максимальное количество токенов
            
        Returns:
            Словарь с полным результатом
        """
        print(f"\n{'#'*60}")
        print(f"# RAG PIPELINE")
        print(f"{'#'*60}\n")
        
        # Шаг 1: Поиск релевантных чанков
        relevant_chunks = self.search_relevant_chunks(query, top_k)
        
        if not relevant_chunks:
            return {
                'answer': "Не удалось найти релевантную информацию в индексе.",
                'context_chunks': [],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'model': model
            }
        
        # Шаг 2: Генерация ответа
        result = self.generate_answer(query, relevant_chunks, model, temperature, max_tokens)
        
        return result
    
    def print_result(self, result: Dict):
        """
        Красиво выводит результат RAG запроса
        
        Args:
            result: Результат от метода ask()
        """
        print(f"\n{'='*60}")
        print(f"💬 ОТВЕТ")
        print(f"{'='*60}\n")
        print(result['answer'])
        
        print(f"\n{'='*60}")
        print(f"📊 МЕТАИНФОРМАЦИЯ")
        print(f"{'='*60}")
        print(f"Модель: {result['model']}")
        print(f"Temperature: {result.get('temperature', 'N/A')}")
        print(f"Использовано фрагментов: {len(result['context_chunks'])}")
        print(f"Токенов использовано: {result['usage']['total_tokens']}")
        print(f"  - Промпт: {result['usage']['prompt_tokens']}")
        print(f"  - Ответ: {result['usage']['completion_tokens']}")
        
        if result['context_chunks']:
            print(f"\n📚 Релевантные фрагменты:")
            for i, chunk in enumerate(result['context_chunks'], 1):
                print(f"\n  {i}. Релевантность: {chunk['similarity']:.4f}")
                print(f"     {chunk['text'][:100]}...")


def main():
    """Пример использования RAG системы"""
    
    # Путь к индексу (создается скриптом из первого документа)
    index_path = 'text_to_test_index.json'
    
    # Проверяем наличие индекса
    if not os.path.exists(index_path):
        print(f"❌ Файл индекса {index_path} не найден!")
        print(f"Сначала запустите скрипт индексации документов.")
        return
    
    try:
        # Создание RAG системы
        rag = YandexRAGSystem(index_path=index_path)
        
        # Примеры вопросов
        questions = [
            "О чем этот документ?",
            "Какие основные темы обсуждаются?",
            "Расскажи подробнее о ключевых концепциях",
        ]
        
        # Можно также запросить вопрос у пользователя
        print("\n" + "="*60)
        print("Введите свой вопрос или нажмите Enter для использования примеров")
        print("="*60)
        user_question = input("Ваш вопрос: ").strip()
        
        if user_question:
            questions = [user_question]
        
        # Обрабатываем каждый вопрос
        for question in questions:
            print(f"\n{'#'*60}")
            print(f"# ВОПРОС: {question}")
            print(f"{'#'*60}")
            
            # Запрос к RAG системе
            result = rag.ask(
                query=question,
                top_k=3,                    # Количество релевантных фрагментов
                model="yandexgpt",          # или "yandexgpt-lite" для быстрых ответов
                temperature=0.3,            # Креативность (0-1)
                max_tokens=2000             # Максимальная длина ответа
            )
            
            # Вывод результата
            rag.print_result(result)
            
            print("\n" + "#"*60 + "\n")
        
        print("🎉 Готово!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Ошибка: {str(e)}")
    except Exception as e:
        print(f"\n❌ Непредвиденная ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()