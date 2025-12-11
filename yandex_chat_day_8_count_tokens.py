import os
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

PHASE_PROMPT = """
Оцени связность и интересность текста
"""

TEMPERATURE = 0.5

class YandexGPTChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Инициализация клиента YandexGPT SDK
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # История сообщений
        self.messages = [
            {"role": "system", "text": PHASE_PROMPT}
        ]
        
        # счётчик Q/A
        self.exchange_count = 0
        
        # Флаг, чтобы понимать, что модель завершила фазу вопрос-ответ
        self.ready_for_answer = False
        
        # Статистика токенов
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_tokens = 0

    def ask(self, question: str, model: str = "yandexgpt-lite", json: bool = False):
        """
        Отправляет вопрос модели и возвращает ответ
        """
        # Добавляем сообщение пользователя
        self.messages.append({"role": "user", "text": question})
        
        try:
            gpt_model = self.sdk.models.completions(model)
            gpt_model = gpt_model.configure(
                temperature=TEMPERATURE,
            )
        
            result = gpt_model.run(self.messages)
            
            answer_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            
            # Получаем ответ и статистику использования токенов
            for alternative in result:
                answer_text = alternative.text
                
                # Получаем информацию об использовании токенов из ответа
                if hasattr(result, 'usage'):
                    usage = result.usage
                    prompt_tokens = usage.input_text_tokens if hasattr(usage, 'input_text_tokens') else 0
                    completion_tokens = usage.completion_tokens if hasattr(usage, 'completion_tokens') else 0
                    reasoning_tokens = usage.reasoning_tokens if hasattr(usage, 'reasoning_tokens') else 0
                    total_tokens = usage.total_tokens if hasattr(usage, 'total_tokens') else 0
                
                break
            
            # Обновляем общую статистику
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_reasoning_tokens += reasoning_tokens
            self.total_tokens += total_tokens
            
            # Выводим статистику
            print(f"📊 Токены запроса: {prompt_tokens}")
            print(f"📊 Токены ответа: {completion_tokens}")
            print(f"📊 Токены reasoning: {reasoning_tokens}")
            print(f"📊 Всего токенов: {total_tokens}")
            print(f"📈 Накоплено за сессию: {self.total_tokens} токенов\n")
            
            # Добавляем ответ в историю
            self.messages.append({"role": "assistant", "text": answer_text})
            
            # Увеличиваем счётчик только после завершённой пары
            self.exchange_count += 1
            
            return answer_text
            
        except Exception as e:
            return f"❌ Ошибка при запросе: {str(e)}"

def interactive_chat():
    """
    Основной цикл общения в консоли
    """
    print("YandexGPT CLI (многошаговый режим). Нажмите CTRL+C для выхода.\n")
    
    try:
        client = YandexGPTChat()
        
        while True:
            question = input("Вы: ").strip()
            
            if not question:
                continue
                
            print()
            answer = client.ask(QUESTION)
            print(f"Модель: {answer}\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {str(e)}")

if __name__ == "__main__":
    interactive_chat()