import os
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()

PHASE_PROMPT = """
Ты личный ментор. Давай советы исходя из лучших практик личной эффективности.
"""

SUMMARIZATION_PROMPT = """
Суммаризируй предыдущий диалог, сохранив все ключевые моменты, контекст и важную информацию.
Создай краткое резюме, которое позволит продолжить разговор без потери контекста.
Формат ответа: краткая сводка в 3-5 предложениях.
"""

TEMPERATURE = 0.5
EXCHANGES_BEFORE_SUMMARY = 3

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
        
        # Текущая суммаризация (если есть)
        self.current_summary = None
        
        # счётчик Q/A
        self.exchange_count = 0
        
        # Флаг, чтобы понимать, что модель завершила фазу вопрос-ответ
        self.ready_for_answer = False
        
        # Статистика токенов
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_tokens = 0

    def _summarize_conversation(self):
        """
        Создает суммаризацию текущего диалога
        """
        print("🔄 Выполняется суммаризация диалога...\n")
        
        # Создаем временный список сообщений для суммаризации
        summary_messages = [
            {"role": "system", "text": SUMMARIZATION_PROMPT}
        ]
        
        # Добавляем предыдущую суммаризацию, если есть
        if self.current_summary:
            summary_messages.append({
                "role": "user", 
                "text": f"Предыдущая суммаризация:\n{self.current_summary}\n\nНовые сообщения для суммаризации:"
            })
        
        # Добавляем все сообщения, кроме системного промпта
        for msg in self.messages[1:]:
            summary_messages.append(msg)
        
        try:
            gpt_model = self.sdk.models.completions("yandexgpt-lite")
            gpt_model = gpt_model.configure(temperature=TEMPERATURE)
            
            result = gpt_model.run(summary_messages)
            
            summary_text = ""
            for alternative in result:
                summary_text = alternative.text
                break
            
            print(f"✅ Суммаризация выполнена:\n{summary_text}\n")
            
            return summary_text
            
        except Exception as e:
            print(f"⚠️ Ошибка при суммаризации: {str(e)}")
            return None

    def _apply_summarization(self):
        """
        Применяет суммаризацию: заменяет историю на системный промпт + суммаризацию
        """
        summary = self._summarize_conversation()
        
        if summary:
            # Сохраняем суммаризацию
            self.current_summary = summary
            
            # Очищаем историю, оставляя только системный промпт и суммаризацию
            self.messages = [
                {"role": "system", "text": PHASE_PROMPT},
                {"role": "system", "text": f"Контекст предыдущего диалога:\n{summary}"}
            ]
            
            # Сбрасываем счетчик
            self.exchange_count = 0

    def ask(self, question: str, model: str = "yandexgpt-lite", json: bool = False):
        """
        Отправляет вопрос модели и возвращает ответ
        """
        # Проверяем, нужна ли суммаризация
        if self.exchange_count >= EXCHANGES_BEFORE_SUMMARY:
            self._apply_summarization()
        
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
            reasoning_tokens = 0
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
            print(f"📈 Накоплено за сессию: {self.total_tokens} токенов")
            print(f"🔢 Обменов до суммаризации: {EXCHANGES_BEFORE_SUMMARY - self.exchange_count - 1}\n")
            
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
    print("YandexGPT CLI (многошаговый режим с суммаризацией). Нажмите CTRL+C для выхода.\n")
    
    try:
        client = YandexGPTChat()
        
        while True:
            question = input("Вы: ").strip()
            
            if not question:
                continue
                
            print()
            answer = client.ask(question)
            print(f"Модель: {answer}\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {str(e)}")

if __name__ == "__main__":
    interactive_chat()