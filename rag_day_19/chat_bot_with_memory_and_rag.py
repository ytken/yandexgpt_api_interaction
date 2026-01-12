import os
import sys
from typing import Dict, List
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_16'))
from text_to_embedding import YandexDocumentIndexer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_17'))
from rag_classes import YandexRAGSystem

load_dotenv()

class RAGChatBot:
    """Чат-бот с RAG и историей диалога"""
    
    def __init__(self, index_path: str, folder_id: str = None, api_key: str = None):
        """
        Args:
            index_path: Путь к индексу документов
            folder_id: ID папки Yandex Cloud
            api_key: API ключ Yandex Cloud
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        print("🚀 Инициализация RAG чат-бота...")
        self.rag_system = YandexRAGSystem(index_path)
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # История диалога
        self.conversation_history: List[Dict[str, str]] = []
        
        print("✅ Чат-бот готов к работе!\n")
    
    def add_to_history(self, role: str, content: str):
        """Добавляет сообщение в историю"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })
    
    def get_history_context(self, max_messages: int = 6) -> str:
        """
        Формирует контекст из истории диалога
        
        Args:
            max_messages: Максимальное количество последних сообщений
            
        Returns:
            Текстовое представление истории
        """
        if not self.conversation_history:
            return ""
        
        # Берем последние N сообщений
        recent_history = self.conversation_history[-max_messages:]
        
        history_text = "ИСТОРИЯ ДИАЛОГА:\n"
        for msg in recent_history:
            role_label = "Пользователь" if msg['role'] == 'user' else "Ассистент"
            history_text += f"{role_label}: {msg['content']}\n"
        
        return history_text
    
    def format_sources(self, context_chunks: List[Dict]) -> str:
        """
        Форматирует источники для вывода
        
        Args:
            context_chunks: Список найденных фрагментов
            
        Returns:
            Отформатированная строка с источниками
        """
        if not context_chunks:
            return "Источники не найдены"
        
        sources = []
        for i, chunk in enumerate(context_chunks, 1):
            source = f"\n📄 Источник {i} (релевантность: {chunk['similarity']:.3f}):"
            source += f"\n   Файл: {chunk['metadata'].get('source', 'unknown')}"
            source += f"\n   Фрагмент: \"{chunk['text'][:150]}...\""
            sources.append(source)
        
        return "\n".join(sources)
    
    def ask(self, question: str, top_k: int = 3, temperature: float = 0.3) -> Dict:
        """
        Задает вопрос боту с учетом истории и RAG
        
        Args:
            question: Вопрос пользователя
            top_k: Количество релевантных фрагментов
            temperature: Температура генерации
            
        Returns:
            Словарь с ответом и метаданными
        """
        # Добавляем вопрос в историю
        self.add_to_history('user', question)
        
        # Ищем релевантный контекст в документах
        print("🔍 Поиск релевантной информации...")
        context_chunks = self.rag_system.search_relevant_chunks(question, top_k=top_k)
        
        # Формируем контекст из найденных фрагментов
        doc_context = "\n\n".join([
            f"Фрагмент {i+1} (релевантность: {chunk['similarity']:.3f}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Получаем историю диалога
        history_context = self.get_history_context()
        
        # Формируем промпт
        prompt = f"""Ты — helpful ассистент, отвечающий на вопросы на основе предоставленных документов и истории диалога.

{history_context}

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{doc_context}

НОВЫЙ ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ИНСТРУКЦИИ:
- Отвечай на основе предоставленного контекста
- Используй информацию из истории диалога, если она релевантна
- Если информации недостаточно, честно скажи об этом
- Будь конкретным и точным

ОТВЕТ:"""
        
        # Генерируем ответ
        print("💭 Генерация ответа...")
        llm_model = self.sdk.models.completions("yandexgpt")
        llm_model = llm_model.configure(temperature=temperature, max_tokens=2000)
        
        result = llm_model.run(prompt)
        answer = result.alternatives[0].text
        
        # Добавляем ответ в историю
        self.add_to_history('assistant', answer)
        
        # Собираем результат
        return {
            'answer': answer,
            'context_chunks': context_chunks,
            'history_length': len(self.conversation_history),
            'usage': {
                'prompt_tokens': result.usage.input_text_tokens if hasattr(result, 'usage') else 0,
                'completion_tokens': result.usage.completion_tokens if hasattr(result, 'usage') else 0,
                'total_tokens': result.usage.total_tokens if hasattr(result, 'usage') else 0
            }
        }
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        print("🗑️ История диалога очищена\n")
    
    def show_history(self):
        """Показывает текущую историю диалога"""
        if not self.conversation_history:
            print("📭 История диалога пуста\n")
            return
        
        print("\n" + "="*80)
        print("📚 ИСТОРИЯ ДИАЛОГА")
        print("="*80)
        
        for i, msg in enumerate(self.conversation_history, 1):
            role_label = "👤 Пользователь" if msg['role'] == 'user' else "🤖 Ассистент"
            print(f"\n{i}. {role_label}:")
            print(f"   {msg['content'][:200]}{'...' if len(msg['content']) > 200 else ''}")
        
        print("\n" + "="*80 + "\n")


def print_welcome():
    """Выводит приветственное сообщение"""
    print("\n" + "="*80)
    print("🤖 RAG ЧАТ-БОТ")
    print("="*80)
    print("""
Доступные команды:
  - Просто введите ваш вопрос для получения ответа
  - /history - показать историю диалога
  - /clear - очистить историю диалога
  - /exit или /quit - выход из программы
  - /help - показать эту справку
    """)
    print("="*80 + "\n")


def run_chat(index_path: str):
    """
    Запускает интерактивный чат
    
    Args:
        index_path: Путь к индексу документов
    """
    try:
        bot = RAGChatBot(index_path)
        print_welcome()
        
        while True:
            # Получаем ввод пользователя
            try:
                user_input = input("👤 Вы: ").strip()
            except EOFError:
                print("\n👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            # Обрабатываем команды
            if user_input.lower() in ['/exit', '/quit']:
                print("\n👋 До свидания!")
                break
            
            elif user_input.lower() == '/clear':
                bot.clear_history()
                continue
            
            elif user_input.lower() == '/history':
                bot.show_history()
                continue
            
            elif user_input.lower() == '/help':
                print_welcome()
                continue
            
            # Обрабатываем обычный вопрос
            print()
            result = bot.ask(user_input)
            
            # Выводим ответ
            print("\n" + "─"*80)
            print("🤖 Ассистент:")
            print("─"*80)
            print(result['answer'])
            
            # Выводим источники
            print("\n" + "─"*80)
            print("📚 ИСТОЧНИКИ:")
            print("─"*80)
            print(bot.format_sources(result['context_chunks']))
            
            # Выводим статистику
            print("\n" + "─"*80)
            print(f"📊 Статистика: Сообщений в истории: {result['history_length']} | "
                  f"Токенов: {result['usage']['total_tokens']}")
            print("─"*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем. До свидания!")
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Основная функция"""
    
    print("="*80)
    print("🔧 НАСТРОЙКА RAG ЧАТ-БОТА")
    print("="*80 + "\n")
    
    # Параметры документа
    doc_filename = "technomax_report.txt"
    index_filename = doc_filename.replace('.txt', '_index.json')
    
    # Проверяем наличие индекса
    if not os.path.exists(index_filename):
        print(f"📄 Индекс не найден. Создаем индекс для документа '{doc_filename}'...")
        
        try:
            indexer = YandexDocumentIndexer()
            index_path = indexer.process_file(doc_filename, chunk_size=400, overlap=50)
            print(f"✅ Индекс создан: {index_path}\n")
        except Exception as e:
            print(f"❌ Ошибка при создании индекса: {str(e)}")
            print("\nУбедитесь, что:")
            print(f"1. Файл '{doc_filename}' существует")
            print("2. Установлены необходимые зависимости")
            print("3. Настроены переменные окружения YANDEX_FOLDER_ID и YANDEX_API_KEY")
            return
    else:
        print(f"✅ Индекс найден: {index_filename}\n")
        index_path = index_filename
    
    # Запускаем чат
    run_chat(index_path)


if __name__ == "__main__":
    main()
