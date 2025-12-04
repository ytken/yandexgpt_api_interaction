import os
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()


class YandexGPTChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Инициализация клиента YandexGPT SDK
        
        Args:
            folder_id: ID каталога в Yandex Cloud
            api_key: API-ключ сервисного аккаунта
        """
        # Получаем данные из аргументов или переменных окружения
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id:
            raise ValueError("Не указан FOLDER_ID. Укажите в аргументе или переменной окружения YANDEX_FOLDER_ID")
        if not self.api_key:
            raise ValueError("Не указан API_KEY. Укажите в аргументе или переменной окружения YANDEX_API_KEY")
        
        # Инициализация SDK
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # История диалога
        self.messages = []
        
    def ask(self, question: str, model: str = "yandexgpt-lite") -> str:
        """
        Отправка запроса и получение ответа
        
        Args:
            question: Вопрос пользователя
            model: Модель для использования (yandexgpt-lite, yandexgpt)
            
        Returns:
            Ответ от YandexGPT
        """
        # Добавляем вопрос в историю
        messages = [
            {
            "role": "system",
            "text": "Представь результат в формате JSON."
            },
            {
            "role": "user",
            "text": question
            }
        ]
        self.messages.append(messages)
        
        try:
            # Получаем модель из SDK
            gpt_model = self.sdk.models.completions(model)
            
            gpt_model = gpt_model.configure(
                temperature=0.6,
                max_tokens=2000,
                response_format="json"
            )
            # Выполняем запрос
            for msg in self.messages:
                result = gpt_model.run(msg)
            
            # Извлекаем ответ
            answer = ""
            for alternative in result:
                answer = alternative.text
                break
            
            if answer:
                # Добавляем ответ в историю
                self.messages.append({
                    "role": "assistant",
                    "text": answer
                })
                return answer
            else:
                return "Ошибка: Не удалось получить ответ от модели"
                
        except Exception as e:
            error_msg = str(e)
            return f"Ошибка: {error_msg}"
    
    def clear_history(self):
        """Очистить историю диалога"""
        self.messages = []


def interactive_chat():
    """
    Интерактивный чат с YandexGPT
    """
    print("=" * 50)
    print("🤖 YANDEX GPT ЧАТ-БОТ (SDK)")
    print("=" * 50)
    print("Команды:")
    print("  /clear - очистить историю диалога")
    print("  /exit  - выход из программы")
    print("=" * 50)
    
    try:
        # Получаем данные из переменных окружения
        folder_id = os.getenv("YANDEX_FOLDER_ID")
        api_key = os.getenv("YANDEX_API_KEY")
        
        if not folder_id or not api_key:
            print("\n❌ Не найдены переменные окружения.")
            print("Создайте файл .env или введите данные вручную:")
            
            folder_id = input("Введите FOLDER_ID: ").strip()
            api_key = input("Введите API_KEY: ").strip()
            
            if not folder_id or not api_key:
                print("❌ Данные обязательны для работы")
                return
        
        # Создаем клиент
        client = YandexGPTChat(folder_id, api_key)
        
        print("\n✅ Подключение установлено!")
        print(f"📁 Каталог: {folder_id[:10]}...")
        print("💬 Начните диалог (для выхода введите /exit)\n")
        
        # Основной цикл диалога
        while True:
            try:
                # Получаем вопрос от пользователя
                question = input("👤 Вы: ").strip()
                
                # Проверяем команды
                if question.lower() in ['/exit', 'exit', 'выход', '/выход']:
                    print("\n👋 До свидания!")
                    break
                elif question.lower() == '/clear':
                    client.clear_history()
                    print("🗑️ История диалога очищена\n")
                    continue
                elif not question:
                    print("⚠️ Введите сообщение\n")
                    continue
                
                # Получаем ответ
                print("\n🤖 YandexGPT: ", end="", flush=True)
                answer = client.ask(question)
                print(f"{answer}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Выход по запросу пользователя")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {str(e)}\n")
                
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {str(e)}")

if __name__ == "__main__":
    interactive_chat()
    