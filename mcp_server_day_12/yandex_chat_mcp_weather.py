import os
import asyncio
import json
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SYSTEM_PROMPT = """Ты - полезный ассистент с доступом к инструменту прогноза погоды.

У тебя есть инструмент get_weather_forecast, который может получить прогноз погоды по координатам.

Когда пользователь спрашивает о погоде в каком-то месте:
1. Определи примерные координаты этого места (широта и долгота)
2. Используй инструмент для получения прогноза
3. Представь результат в удобочитаемом виде

Доступные инструменты:
- get_weather_forecast: получает прогноз погоды по координатам (lat, lon)
  Параметры:
  - lat (число): широта от -90 до 90
  - lon (число): долгота от -180 до 180
  - product (строка, опционально): тип прогноза - civil, civillight, astro
  - output (строка, опционально): формат - json или xml

Если пользователь спрашивает о погоде, ОБЯЗАТЕЛЬНО используй формат:
USE_TOOL: get_weather_forecast
ARGUMENTS: {"lat": <широта>, "lon": <долгота>}

После получения результата от инструмента, интерпретируй его для пользователя."""

TEMPERATURE = 0.7

# Известные координаты городов
CITY_COORDS = {
    "москва": {"lat": 55.7558, "lon": 37.6173},
    "moscow": {"lat": 55.7558, "lon": 37.6173},
    "санкт-петербург": {"lat": 59.9311, "lon": 30.3609},
    "петербург": {"lat": 59.9311, "lon": 30.3609},
    "saint petersburg": {"lat": 59.9311, "lon": 30.3609},
    "лондон": {"lat": 51.5074, "lon": -0.1278},
    "london": {"lat": 51.5074, "lon": -0.1278},
    "нью-йорк": {"lat": 40.7128, "lon": -74.0060},
    "new york": {"lat": 40.7128, "lon": -74.0060},
    "париж": {"lat": 48.8566, "lon": 2.3522},
    "paris": {"lat": 48.8566, "lon": 2.3522},
    "токио": {"lat": 35.6762, "lon": 139.6503},
    "tokyo": {"lat": 35.6762, "lon": 139.6503},
    "берлин": {"lat": 52.5200, "lon": 13.4050},
    "berlin": {"lat": 52.5200, "lon": 13.4050},
    "амстердам": {"lat": 52.3676, "lon": 4.9041},
    "amsterdam": {"lat": 52.3676, "lon": 4.9041},
}

class MCPWeatherClient:
    """Клиент для работы с MCP сервером погоды"""
    
    def __init__(self, server_script_path: str = "mcp_server_weather.py"):
        self.server_script_path = server_script_path
        self.session = None
        self.read_stream = None
        self.write_stream = None
        self._client_context = None
        self._session_context = None
        
    async def start(self):
        """Запуск MCP сервера"""
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=None
        )
        
        print("🚀 Запуск MCP сервера погоды...")
        self._client_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self._client_context.__aenter__()
        
        self._session_context = ClientSession(self.read_stream, self.write_stream)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
        print("✅ MCP сервер запущен и готов к работе\n")
        
    async def stop(self):
        """Остановка MCP сервера"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
        print("🛑 MCP сервер остановлен")
    
    async def get_weather(self, lat: float, lon: float, product: str = "civil"):
        """Получить прогноз погоды"""
        try:
            result = await self.session.call_tool(
                "get_weather_forecast",
                arguments={
                    "lat": lat,
                    "lon": lon,
                    "product": product,
                    "output": "json"
                }
            )
            
            # Извлекаем текст из результата
            weather_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    weather_text += content.text
            
            return weather_text
            
        except Exception as e:
            return f"❌ Ошибка при получении погоды: {str(e)}"

class YandexGPTChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None, 
                 mcp_server_path: str = "mcp_server_weather.py"):
        """
        Инициализация клиента YandexGPT SDK с поддержкой MCP
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # MCP клиент
        self.mcp_client = MCPWeatherClient(mcp_server_path)
        self.mcp_started = False
        
        # История сообщений
        self.messages = [
            {"role": "system", "text": SYSTEM_PROMPT}
        ]
        
        # Статистика
        self.exchange_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_reasoning_tokens = 0
        self.total_tokens = 0

    async def start_mcp(self):
        """Запуск MCP сервера"""
        if not self.mcp_started:
            await self.mcp_client.start()
            self.mcp_started = True
    
    async def stop_mcp(self):
        """Остановка MCP сервера"""
        if self.mcp_started:
            await self.mcp_client.stop()
            self.mcp_started = False

    def parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Парсит ответ модели на предмет вызова инструмента
        Ищет паттерн: USE_TOOL: ... ARGUMENTS: {...}
        """
        if "USE_TOOL:" not in text:
            return None
        
        try:
            # Ищем название инструмента
            tool_start = text.find("USE_TOOL:")
            args_start = text.find("ARGUMENTS:", tool_start)
            
            if tool_start == -1 or args_start == -1:
                return None
            
            tool_name = text[tool_start + 9:args_start].strip()
            
            # Ищем JSON с аргументами
            args_text = text[args_start + 10:].strip()
            
            # Пробуем извлечь JSON
            json_start = args_text.find("{")
            json_end = args_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            args_json = args_text[json_start:json_end]
            arguments = json.loads(args_json)
            
            return {
                "tool": tool_name,
                "arguments": arguments
            }
        except Exception as e:
            print(f"⚠️  Ошибка парсинга вызова инструмента: {e}")
            return None

    async def ask(self, question: str, model: str = "yandexgpt-lite"):
        """
        Отправляет вопрос модели и возвращает ответ
        Автоматически обрабатывает вызовы инструментов
        """
        # Убеждаемся, что MCP запущен
        if not self.mcp_started:
            await self.start_mcp()
        
        # Добавляем сообщение пользователя
        self.messages.append({"role": "user", "text": question})
        
        try:
            gpt_model = self.sdk.models.completions(model)
            gpt_model = gpt_model.configure(temperature=TEMPERATURE)
        
            result = gpt_model.run(self.messages)
            
            answer_text = ""
            
            # Получаем ответ
            for alternative in result:
                answer_text = alternative.text
                
                # Статистика токенов (если доступна)
                if hasattr(result, 'usage'):
                    usage = result.usage
                    prompt_tokens = getattr(usage, 'input_text_tokens', 0)
                    completion_tokens = getattr(usage, 'completion_tokens', 0)
                    reasoning_tokens = getattr(usage, 'reasoning_tokens', 0)
                    total_tokens = getattr(usage, 'total_tokens', 0)
                    
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_reasoning_tokens += reasoning_tokens
                    self.total_tokens += total_tokens
                    
                    print(f"📊 Токены: запрос={prompt_tokens}, ответ={completion_tokens}, "
                          f"reasoning={reasoning_tokens}, всего={total_tokens}")
                    print(f"📈 Накоплено: {self.total_tokens} токенов\n")
                
                break
            
            # Проверяем, нужно ли вызвать инструмент
            tool_call = self.parse_tool_call(answer_text)
            
            if tool_call:
                print(f"🔧 Модель запросила инструмент: {tool_call['tool']}")
                print(f"📦 Аргументы: {json.dumps(tool_call['arguments'], ensure_ascii=False)}\n")
                
                # Вызываем инструмент
                if tool_call['tool'] == 'get_weather_forecast':
                    weather_result = await self.mcp_client.get_weather(
                        lat=tool_call['arguments']['lat'],
                        lon=tool_call['arguments']['lon'],
                        product=tool_call['arguments'].get('product', 'civil')
                    )
                    
                    print(f"🌤️  Получен результат от инструмента:\n{weather_result}\n")
                    
                    # Добавляем результат инструмента в историю
                    tool_result_message = f"Результат от инструмента {tool_call['tool']}:\n{weather_result}"
                    self.messages.append({"role": "assistant", "text": answer_text})
                    self.messages.append({"role": "user", "text": tool_result_message})
                    
                    # Запрашиваем финальный ответ от модели
                    print("💬 Запрос финального ответа от модели...\n")
                    final_result = gpt_model.run(self.messages)
                    
                    for alternative in final_result:
                        final_answer = alternative.text
                        self.messages.append({"role": "assistant", "text": final_answer})
                        self.exchange_count += 1
                        return final_answer
            
            # Если инструмент не нужен, просто возвращаем ответ
            self.messages.append({"role": "assistant", "text": answer_text})
            self.exchange_count += 1
            
            return answer_text
            
        except Exception as e:
            return f"❌ Ошибка при запросе: {str(e)}"

async def interactive_chat():
    """
    Основной цикл общения в консоли с поддержкой MCP
    """
    print("=" * 70)
    print("🤖 YandexGPT CLI с поддержкой инструмента прогноза погоды")
    print("=" * 70)
    print("\nПримеры запросов:")
    print("  • Какая погода в Москве?")
    print("  • Расскажи о погоде в Лондоне")
    print("  • Что с погодой в Нью-Йорке?")
    print("\nНажмите CTRL+C для выхода.\n")
    
    client = None
    
    try:
        client = YandexGPTChat()
        
        while True:
            question = input("Вы: ").strip()
            
            if not question:
                continue
            
            print()
            answer = await client.ask(question)
            print(f"🤖 Модель: {answer}\n")
            print("-" * 70 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Выход по запросу пользователя")
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            await client.stop_mcp()

if __name__ == "__main__":
    asyncio.run(interactive_chat())