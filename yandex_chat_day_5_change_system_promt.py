import os
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

load_dotenv()


PHASE_PROMPT_1 = """
Ты веришь, что ИИ и технологии приведут к сингулярности — невообразимому скачку, который решит все глобальные проблемы и изменит саму природу человека. Твой тон восторженный и уверенный.
"""

PHASE_PROMPT_2 = """
Ты считаешь идею сингулярности мифом, а будущее — эрой "великого застоя", где прогресс замедлится из-за физических и социальных ограничений. Твой тон трезвый и осторожный.
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
            {"role": "system", "text": PHASE_PROMPT_1}
        ]

        # счётчик Q/A
        self.exchange_count = 0

        # Флаг, чтобы понимать, что модель завершила фазу вопрос-ответ
        self.ready_for_answer = False

    def ask(self, question: str, model: str = "yandexgpt-lite", json: bool = False ):
        """
        Отправляет вопрос модели и возвращает ответ
        """

        # Добавляем сообщение пользователя
        self.messages.append({"role": "user", "text": question})

        # 🔥 Если прошли 3 полных обмена — меняем системный промт
        if self.exchange_count == 3:
            # Заменяем только первый элемент истории (system)
            self.messages[0] = {"role": "system", "text": PHASE_PROMPT_2}
            # Чтобы не менять снова
            self.exchange_count += 1
            print('Изменяем систем промт с отпимистичного на осторожный')

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
            print()

            answer = client.ask(question)

            print(f"Модель: {answer}\n")

    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка инициализации: {str(e)}")


if __name__ == "__main__":
    interactive_chat()
