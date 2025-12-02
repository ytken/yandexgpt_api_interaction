import requests
import json
import os
from typing import Optional
# Данные из .env файла
from dotenv import load_dotenv
load_dotenv()


class YandexGPTChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None):
        """
        Инициализация клиента YandexGPT API
        
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
        
        # URL API
        self.api_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        # Заголовки с правильной кодировкой
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json; charset=utf-8"
        }
        
        # История диалога
        self.messages = []
        
    def ask(self, question: str, model: str = "yandexgpt-lite") -> str:
        """
        Отправка запроса и получение ответа
        
        Args:
            question: Вопрос пользователя
            model: Модель для использования
            
        Returns:
            Ответ от YandexGPT
        """
        # Добавляем вопрос в историю
        self.messages.append({
            "role": "user",
            "text": question
        })
        
        # Формируем данные запроса
        data = {
            "modelUri": f"gpt://{self.folder_id}/{model}",
            "completionOptions": {
                "stream": False,
                "temperature": 0.6,
                "maxTokens": "2000"
            },
            "messages": self.messages
        }
        
        try:
            # Отправляем запрос с явным указанием кодировки
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json=data,
                timeout=30
            )
            
            # Проверяем статус
            if response.status_code == 200:
                result = response.json()
                
                # Извлекаем ответ
                if ("result" in result and 
                    "alternatives" in result["result"] and 
                    result["result"]["alternatives"]):
                    
                    answer = result["result"]["alternatives"][0]["message"]["text"]
                    
                    # Добавляем ответ в историю
                    self.messages.append({
                        "role": "assistant",
                        "text": answer
                    })
                    
                    return answer
                else:
                    return "Ошибка: Не удалось получить ответ от модели"
                    
            elif response.status_code == 401:
                return "Ошибка 401: Неверный API ключ"
            elif response.status_code == 403:
                return "Ошибка 403: Недостаточно средств или нет доступа к модели"
            elif response.status_code == 404:
                return "Ошибка 404: Модель или каталог не найден"
            else:
                try:
                    error_data = response.json()
                    return f"Ошибка {response.status_code}: {error_data}"
                except:
                    return f"Ошибка {response.status_code}: {response.text}"
                    
        except requests.exceptions.ConnectionError:
            return "Ошибка подключения: Проверьте интернет-соединение"
        except requests.exceptions.Timeout:
            return "Ошибка: Таймаут запроса"
        except json.JSONDecodeError as e:
            return f"Ошибка обработки JSON: {str(e)}"
        except Exception as e:
            # Явно декодируем ошибку для кириллицы
            error_msg = str(e)
            try:
                error_msg = error_msg.encode('latin-1').decode('utf-8', 'ignore')
            except:
                pass
            return f"Ошибка: {error_msg}"
    
    def clear_history(self):
        """Очистить историю диалога"""
        self.messages = []


def interactive_chat():
    """
    Интерактивный чат с YandexGPT
    """
    print("=" * 50)
    print("🤖 YANDEX GPT ЧАТ-БОТ")
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


def simple_chat():
    """
    Упрощенная версия без классов
    """
    import requests
    
    print("=" * 50)
    print("🤖 YANDEX GPT ПРОСТОЙ ЧАТ")
    print("=" * 50)
    
    # Ввод данных
    folder_id = input("Введите FOLDER_ID: ").strip()
    api_key = input("Введите API_KEY: ").strip()
    
    if not folder_id or not api_key:
        print("❌ Данные обязательны")
        return
    
    print("\n✅ Готово! Введите сообщения (пустая строка для выхода)")
    
    while True:
        # Ввод сообщения
        question = input("\n👤 Вы: ").strip()
        
        if not question:
            print("👋 Выход")
            break
        
        try:
            # Формируем запрос
            url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
            headers = {
                "Authorization": f"Api-Key {api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "modelUri": f"gpt://{folder_id}/yandexgpt-lite",
                "messages": [{"role": "user", "text": question}],
                "completionOptions": {"stream": False}
            }
            
            # Отправляем запрос
            print("\n🤖 YandexGPT: ", end="", flush=True)
            response = requests.post(url, headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "Нет ответа")
                print(answer)
            else:
                print(f"Ошибка {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"Ошибка: {str(e)}")


def minimal_chat():
    """
    Минималистичная версия для быстрого запуска
    """
    import requests
    
    
    
    folder_id = os.getenv("YANDEX_FOLDER_ID")
    api_key = os.getenv("YANDEX_API_KEY")
    
    if not folder_id:
        folder_id = input("📁 Введите FOLDER_ID: ")
    if not api_key:
        api_key = input("🔑 Введите API_KEY: ")
    
    print("\n" + "="*40)
    print("🤖 YandexGPT Чат запущен!")
    print("Введите сообщение или 'выход' для завершения")
    print("="*40 + "\n")
    
    while True:
        user_input = input("Вы: ").strip()
        
        if user_input.lower() in ['выход', 'exit', 'quit']:
            print("👋 До свидания!")
            break
        
        if not user_input:
            continue
        
        try:
            # Формируем и отправляем запрос
            response = requests.post(
                "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                headers={"Authorization": f"Api-Key {api_key}"},
                json={
                    "modelUri": f"gpt://{folder_id}/yandexgpt-lite",
                    "messages": [{"role": "user", "text": user_input}],
                    "completionOptions": {"stream": False}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data["result"]["alternatives"][0]["message"]["text"]
                print(f"\n🤖 YandexGPT: {answer}\n")
            else:
                print(f"\n❌ Ошибка {response.status_code}\n")
                
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Ошибка сети: {e}\n")
        except KeyError:
            print(f"\n❌ Ошибка формата ответа\n")
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")


if __name__ == "__main__":
    # Выберите один из вариантов:
    
    # Вариант 1: Полнофункциональный чат с историей
    interactive_chat()
    
    # Вариант 2: Упрощенный чат
    # simple_chat()
    
    # Вариант 3: Минималистичный чат
    # minimal_chat()