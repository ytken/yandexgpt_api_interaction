import os
import sys
from typing import Dict, List, Tuple
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

# Импорт классов с проверкой путей
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_17'))
    from rag_classes import YandexRAGSystem
    print("✅ YandexRAGSystem импортирован")
except ImportError as e:
    print(f"⚠️ Не удалось импортировать YandexRAGSystem: {e}")
    print("Убедитесь, что файл rag_classes.py существует")
    YandexRAGSystem = None

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_16'))
    from text_to_embedding import YandexDocumentIndexer
    print("✅ YandexDocumentIndexer импортирован")
except ImportError as e:
    print(f"⚠️ Не удалось импортировать YandexDocumentIndexer: {e}")
    print("Убедитесь, что файл text_to_embedding.py существует")
    YandexDocumentIndexer = None

load_dotenv()


class RAGWithRelevanceFilter(YandexRAGSystem):
    """RAG система с фильтром релевантности"""
    
    def __init__(self, index_path: str, relevance_threshold: float = 0.5,
                 folder_id: str = None, api_key: str = None):
        """
        Args:
            index_path: Путь к индексному файлу
            relevance_threshold: Минимальный порог релевантности (0-1)
            folder_id: ID папки Yandex Cloud
            api_key: API ключ Yandex Cloud
        """
        # Инициализируем родительский класс
        super().__init__(index_path, folder_id, api_key)
        self.relevance_threshold = relevance_threshold
        print(f"✅ Установлен порог релевантности: {relevance_threshold}\n")
    
    def filter_by_relevance(self, chunks_with_scores: List[Tuple]) -> List[Tuple]:
        """
        Фильтрует найденные фрагменты по порогу релевантности
        
        Args:
            chunks_with_scores: Список кортежей (chunk, score)
            
        Returns:
            Отфильтрованный список фрагментов
        """
        filtered = [(chunk, score) for chunk, score in chunks_with_scores 
                   if score >= self.relevance_threshold]
        
        print(f"🔍 Фильтрация релевантности:")
        print(f"   Найдено фрагментов: {len(chunks_with_scores)}")
        print(f"   Порог релевантности: {self.relevance_threshold}")
        print(f"   Прошло фильтр: {len(filtered)}")
        
        if filtered:
            scores = [score for _, score in filtered]
            print(f"   Средняя релевантность: {sum(scores)/len(scores):.3f}")
            print(f"   Мин/Макс релевантность: {min(scores):.3f} / {max(scores):.3f}")
        else:
            print(f"   ⚠️ Ни один фрагмент не прошел порог {self.relevance_threshold}")
        
        print()
        return filtered
    
    def ask_with_filter(self, query: str, top_k: int = 5, 
                       model: str = "yandexgpt", temperature: float = 0.3) -> Dict:
        """
        Запрос с двухэтапной обработкой:
        1. Поиск top_k наиболее похожих фрагментов
        2. Фильтрация по порогу релевантности
        
        Args:
            query: Вопрос пользователя
            top_k: Количество фрагментов для первичного поиска
            model: Модель YandexGPT
            temperature: Температура генерации
            
        Returns:
            Результат с ответом и метаданными
        """
        print(f"🔎 Двухэтапный поиск для вопроса: '{query}'\n")
        
        # ЭТАП 1: Поиск top_k наиболее похожих фрагментов
        print(f"📍 ЭТАП 1: Поиск top-{top_k} наиболее похожих фрагментов...")
        
        # Используем встроенный метод поиска из родительского класса
        query_embedding = self.generate_query_embedding(query)
        
        # Вычисляем сходство со всеми документами
        similar_chunks = []
        for doc in self.documents:
            if not doc['embedding']:
                continue
            
            similarity = self._cosine_similarity(query_embedding, doc['embedding'])
            similar_chunks.append((
                {
                    'id': doc['id'],
                    'text': doc['text'],
                    'char_start': doc['char_start'],
                    'char_end': doc['char_end']
                },
                similarity
            ))
        
        # Сортируем и берем top_k
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        similar_chunks = similar_chunks[:top_k]
        
        print(f"✅ Найдено {len(similar_chunks)} фрагментов")
        for i, (chunk, score) in enumerate(similar_chunks, 1):
            print(f"   {i}. Релевантность: {score:.3f} | "
                  f"Текст: {chunk['text'][:60]}...")
        print()
        
        # ЭТАП 2: Фильтрация по порогу релевантности
        print(f"📍 ЭТАП 2: Фильтрация по порогу релевантности...")
        filtered_chunks = self.filter_by_relevance(similar_chunks)
        
        # Если ничего не прошло фильтр - возвращаем ответ без контекста
        if not filtered_chunks:
            print("⚠️ Релевантного контекста не найдено. Отвечаю без RAG.\n")
            return {
                'answer': self._answer_without_context(query),
                'context_chunks': [],
                'relevance_scores': [],
                'filtered_out': len(similar_chunks),
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'warning': f'Все {len(similar_chunks)} фрагментов отфильтрованы '
                          f'(порог {self.relevance_threshold})'
            }
        
        # Генерируем ответ на основе отфильтрованного контекста
        print(f"📍 ЭТАП 3: Генерация ответа на основе {len(filtered_chunks)} фрагментов...\n")
        
        # Преобразуем в формат для generate_answer
        context_chunks = []
        for chunk, score in filtered_chunks:
            context_chunks.append({
                'id': chunk['id'],
                'text': chunk['text'],
                'similarity': score,
                'char_start': chunk['char_start'],
                'char_end': chunk['char_end']
            })
        
        # Генерируем ответ используя метод родительского класса
        result = self.generate_answer(query, context_chunks, model, temperature)
        
        # Добавляем информацию о фильтрации
        result['filtered_out'] = len(similar_chunks) - len(filtered_chunks)
        result['threshold_used'] = self.relevance_threshold
        result['relevance_scores'] = [score for _, score in filtered_chunks]
        
        return result
    
    def _answer_without_context(self, query: str) -> str:
        """Генерирует ответ без контекста (когда все отфильтровано)"""
        prompt = f"""Ответь на вопрос, но честно признай, что у тебя нет конкретной информации из документа.

ВОПРОС:
{query}

ОТВЕТ:"""
        
        llm_model = self.sdk.models.completions("yandexgpt")
        llm_model = llm_model.configure(temperature=0.3, max_tokens=500)
        
        result = llm_model.run(prompt)
        return result.alternatives[0].text


class RAGComparison:
    """Класс для сравнения ответов с RAG и без RAG"""
    
    def __init__(self, folder_id: str = None, api_key: str = None):
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        print("🚀 Инициализация системы сравнения...")
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        print("✅ SDK инициализирован!\n")
    
    def ask_without_rag(self, query: str, model: str = "yandexgpt", 
                       temperature: float = 0.3) -> Dict:
        """Запрос к LLM БЕЗ RAG"""
        print("🤖 Запрос БЕЗ RAG (только знания модели)...")
        
        prompt = f"""Ответь на следующий вопрос на основе своих знаний:

ВОПРОС:
{query}

ОТВЕТ:"""
        
        llm_model = self.sdk.models.completions(model)
        llm_model = llm_model.configure(temperature=temperature, max_tokens=2000)
        
        result = llm_model.run(prompt)
        answer_text = result.alternatives[0].text
        
        usage_stats = {
            'prompt_tokens': result.usage.input_text_tokens if hasattr(result, 'usage') else 0,
            'completion_tokens': result.usage.completion_tokens if hasattr(result, 'usage') else 0,
            'total_tokens': result.usage.total_tokens if hasattr(result, 'usage') else 0
        }
        
        print(f"✅ Ответ получен! Токенов: {usage_stats['total_tokens']}\n")
        
        return {
            'answer': answer_text,
            'usage': usage_stats,
            'method': 'WITHOUT_RAG'
        }
    
    def print_comparison_with_filter(self, query: str, rag_result: Dict, 
                                    no_rag_result: Dict):
        """Выводит сравнение с информацией о фильтрации"""
        print("\n" + "="*80)
        print(f"❓ ВОПРОС: {query}")
        print("="*80)
        
        print("\n🔵 ОТВЕТ БЕЗ RAG (только знания модели)")
        print("-"*80)
        print(no_rag_result['answer'])
        print(f"\n📊 Токенов: {no_rag_result['usage']['total_tokens']}")
        
        print("\n🟢 ОТВЕТ С RAG + ФИЛЬТР РЕЛЕВАНТНОСТИ")
        print("-"*80)
        print(rag_result['answer'])
        print(f"\n📊 Токенов: {rag_result['usage']['total_tokens']}")
        print(f"📚 Использовано фрагментов: {len(rag_result['context_chunks'])}")
        print(f"🔍 Отфильтровано фрагментов: {rag_result.get('filtered_out', 0)}")
        print(f"📏 Порог релевантности: {rag_result.get('threshold_used', 'N/A')}")
        
        if rag_result.get('relevance_scores'):
            scores = rag_result['relevance_scores']
            print(f"⭐ Средняя релевантность: {sum(scores)/len(scores):.3f}")
            print(f"📈 Диапазон: {min(scores):.3f} - {max(scores):.3f}")
        
        if rag_result.get('warning'):
            print(f"\n⚠️ {rag_result['warning']}")
        
        print("\n" + "="*80 + "\n")


def test_different_thresholds():
    """Тестирует работу с разными порогами релевантности"""
    
    print("="*80)
    print("🧪 ТЕСТИРОВАНИЕ ПОРОГОВ РЕЛЕВАНТНОСТИ")
    print("="*80 + "\n")
    
    # Проверяем доступность необходимых классов
    if YandexRAGSystem is None or YandexDocumentIndexer is None:
        print("❌ Не удалось импортировать необходимые классы!")
        print("\n📋 Инструкции:")
        print("1. Убедитесь, что следующие файлы существуют:")
        print("   - rag_day_17/rag_classes.py (содержит YandexRAGSystem)")
        print("   - rag_day_16/text_to_embedding.py (содержит YandexDocumentIndexer)")
        print("\n2. Или скопируйте класс YandexRAGSystem из предоставленного документа")
        return
    
    # Создаем тестовый документ и индекс
    doc_filename = "technomax_report.txt"
    
    # Проверяем существование файла
    if not os.path.exists(doc_filename):
        print(f"⚠️ Файл {doc_filename} не найден!")
        print("Создайте файл с тестовым содержимым или укажите существующий файл.")
        return
    
    try:
        print("🔧 Создание индекса документа...")
        indexer = YandexDocumentIndexer()
        index_path = indexer.process_file(doc_filename, chunk_size=400, overlap=50)
        
        # Тестовые вопросы разной релевантности
        test_cases = [
            {
                'question': "Какова выручка TechnoMax Solutions за 2024 год?",
                'type': 'highly_relevant',
                'description': 'Вопрос с высокой релевантностью (конкретный факт из документа)'
            },
            {
                'question': "Что такое машинное обучение в общем смысле?",
                'type': 'low_relevant',
                'description': 'Вопрос с низкой релевантностью (общие знания)'
            },
            {
                'question': "Какие продукты разрабатывает компания?",
                'type': 'medium_relevant',
                'description': 'Вопрос со средней релевантностью'
            }
        ]
        
        # Тестируем разные пороги
        thresholds = [0.3, 0.5, 0.7]
        
        for test_case in test_cases:
            print(f"\n{'='*80}")
            print(f"📋 ТЕСТ: {test_case['description']}")
            print(f"❓ Вопрос: {test_case['question']}")
            print(f"{'='*80}\n")
            
            for threshold in thresholds:
                print(f"\n{'─'*80}")
                print(f"🎚️ ПОРОГ РЕЛЕВАНТНОСТИ: {threshold}")
                print(f"{'─'*80}\n")
                
                # Создаем RAG с текущим порогом
                rag_filtered = RAGWithRelevanceFilter(index_path, threshold)
                
                # Получаем ответ с фильтром
                result = rag_filtered.ask_with_filter(test_case['question'], top_k=5)
                
                # Выводим результат
                print(f"\n💬 ОТВЕТ:")
                print(result['answer'])
                print(f"\n📊 Статистика:")
                print(f"   Фрагментов прошло фильтр: {len(result['context_chunks'])}")
                print(f"   Фрагментов отфильтровано: {result['filtered_out']}")
                
                if result.get('warning'):
                    print(f"\n⚠️ {result['warning']}")
        
        print("\n" + "="*80)
        print("✅ Тестирование завершено!")
        print("="*80 + "\n")
        
        # Рекомендации
        print("💡 РЕКОМЕНДАЦИИ ПО ВЫБОРУ ПОРОГА:")
        print("   • 0.3-0.4: Мягкий фильтр, больше контекста, риск нерелевантной информации")
        print("   • 0.5-0.6: Сбалансированный порог (рекомендуется)")
        print("   • 0.7+: Строгий фильтр, только высокорелевантный контекст")
        print()
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция с демонстрацией фильтра релевантности"""
    
    print("="*80)
    print("🧪 RAG С ФИЛЬТРОМ РЕЛЕВАНТНОСТИ")
    print("="*80 + "\n")
    
    # Запускаем тестирование разных порогов
    test_different_thresholds()


if __name__ == "__main__":
    main()