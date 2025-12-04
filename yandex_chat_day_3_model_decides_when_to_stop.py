import os
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()


PHASE_PROMPT = """
Ты работаешь по следующему протоколу скрытого рассуждения.

1. Перед каждым ответом проанализируй запрос пользователя.
2. Сформируй у себя в уме (НЕ показывая пользователю) скрытую цепочку рассуждений:
   - какие данные нужны для корректного ответа;
   - какие данные уже есть;
   - каких данных не хватает.

3. Содержание скрытой цепочки рассуждений пользователю НИКОГДА не показывай.
   Пользователь видит только финальное решение ("Ответ") или уточняющий вопрос.

4. Если скрытый анализ показывает, что данных недостаточно — задай пользователю
   ОДИН уточняющий вопрос.

5. Если данных достаточно — сформируй финальный ответ.

6. Никогда не отвечай сразу, не проверив полноту данных через скрытую цепочку рассуждений.
"""


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
            {
                "role": "system",
                "text": PHASE_PROMPT
            }
        ]

        # Флаг, чтобы понимать, что модель завершила фазу вопрос-ответ
        self.ready_for_answer = False

    def ask(self, question: str, model: str = "yandexgpt-lite", json: bool = False ):
        """
        Отправляет вопрос модели и возвращает ответ
        """

        # Добавляем сообщение пользователя
        self.messages.append({"role": "user", "text": question})

        try:
            gpt_model = self.sdk.models.completions(model)
            gpt_model = gpt_model.configure(
                temperature=0.6,
                max_tokens=2000,
            )
            if json:
                gpt_model = gpt_model.configure(response_format="json")

            result = gpt_model.run(self.messages)
            #answer_text = response.result["alternatives"][0]["message"]["text"]
            for alternative in result:
                answer_text = alternative.text
                break

            # Добавляем ответ в историю
            self.messages.append({"role": "assistant", "text": answer_text})

            # Проверяем маркер готовности
            if "<ready/>" in answer_text.lower().strip():
                self.ready_for_answer = True
                return "Ок, я получил все данные. Готов дать финальный ответ."

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
            print()

            answer = client.ask(question)

            print(f"Модель: {answer}\n")

            # Если модель сообщила <ready/>, следующий ответ будет финальным
            if client.ready_for_answer:
                final_question = input("Нажмите Enter для получения финального ответа...")
                print()

                final_answer = client.ask("Дай финальный ответ.", json=True)
                print(f"Финальный ответ модели:\n{final_answer}\n")

                # Сброс — для нового запроса
                client.ready_for_answer = False
                print("--- Готов к новому запросу ---\n")

    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {str(e)}")


if __name__ == "__main__":
    interactive_chat()
