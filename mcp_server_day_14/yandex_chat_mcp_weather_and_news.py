import os
import asyncio
import json
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client

load_dotenv()

SYSTEM_PROMPT = """Ты - полезный ассистент с доступом к нескольким инструментам.

У тебя есть следующие инструменты:
1. get_weather_forecast: получает прогноз погоды по координатам (lat, lon)
2. Инструменты DeepWiki для поиска информации о GitHub репозиториях

ВАЖНО: Ты можешь вызывать только ОДИН инструмент за раз!

Когда пользователь запрашивает несколько действий:
1. Сначала вызови ПЕРВЫЙ нужный инструмент
2. Дождись результата
3. Затем вызови СЛЕДУЮЩИЙ инструмент
4. После получения всех результатов - создай финальный ответ

Формат вызова инструмента (ТОЛЬКО ОДИН за раз):
USE_TOOL: <имя_инструмента>
ARGUMENTS: {"параметр": "значение"}

Доступные инструменты weather:
- get_weather_forecast: lat (число), lon (число), product (строка, по умолчанию "civil"), output (строка, по умолчанию "json")

После получения списка инструментов DeepWiki ты узнаешь их параметры.

Пример правильного подхода для "Погода в Москве и как использовать React hooks":
1. Сначала: USE_TOOL: get_weather_forecast с координатами Москвы
2. После получения погоды: USE_TOOL: ask_question про React hooks  
3. После получения обоих результатов: создай объединённый ответ"""

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
    """Клиент для работы с локальным MCP сервером погоды"""
    
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
        
        print("🚀 Запуск локального MCP сервера погоды...")
        self._client_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self._client_context.__aenter__()
        
        self._session_context = ClientSession(self.read_stream, self.write_stream)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
        print("✅ MCP сервер погоды запущен\n")
        
    async def stop(self):
        """Остановка MCP сервера"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
        print("🛑 MCP сервер погоды остановлен")
    
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
            
            weather_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    weather_text += content.text
            
            return weather_text
            
        except Exception as e:
            return f"❌ Ошибка при получении погоды: {str(e)}"


class MCPDeepWikiClient:
    """Клиент для работы с удалённым MCP сервером DeepWiki"""
    
    def __init__(self, url: str = "https://mcp.deepwiki.com/sse"):
        self.url = url
        self.session = None
        self.streams = None
        self.available_tools = []
        
    async def start(self):
        """Подключение к удалённому MCP серверу"""
        print(f"🌐 Подключение к DeepWiki MCP ({self.url})...")
        
        try:
            # Используем AsyncExitStack для правильного управления контекстом
            self.exit_stack = AsyncExitStack()
            
            # Подключаемся к SSE серверу
            sse_context = sse_client(self.url)
            read_stream, write_stream = await self.exit_stack.enter_async_context(sse_context)
            
            # Создаём сессию
            session_context = ClientSession(read_stream, write_stream)
            self.session = await self.exit_stack.enter_async_context(session_context)
            
            await self.session.initialize()
            
            # Получаем список доступных инструментов
            tools_list = await self.session.list_tools()
            self.available_tools = tools_list.tools if hasattr(tools_list, 'tools') else []
            
            print(f"✅ DeepWiki MCP подключен, доступно инструментов: {len(self.available_tools)}")
            
            # Выводим список инструментов
            if self.available_tools:
                print("\n📋 Доступные инструменты DeepWiki:")
                for tool in self.available_tools:
                    print(f"   • {tool.name}: {tool.description if hasattr(tool, 'description') else 'нет описания'}")
                print()
            
        except Exception as e:
            print(f"❌ Ошибка подключения к DeepWiki: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
        
    async def stop(self):
        """Отключение от удалённого MCP сервера"""
        if hasattr(self, 'exit_stack'):
            await self.exit_stack.aclose()
        print("🛑 DeepWiki MCP отключен")
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Вызов инструмента DeepWiki"""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            result_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += content.text
            
            return result_text
            
        except Exception as e:
            return f"❌ Ошибка при вызове {tool_name}: {str(e)}"
    
    def get_tools_description(self) -> str:
        """Получить описание доступных инструментов для промпта"""
        if not self.available_tools:
            return "Инструменты DeepWiki недоступны"
        
        descriptions = []
        for tool in self.available_tools:
            desc = f"- {tool.name}"
            if hasattr(tool, 'description'):
                desc += f": {tool.description}"
            if hasattr(tool, 'inputSchema'):
                schema = tool.inputSchema
                if isinstance(schema, dict) and 'properties' in schema:
                    params = ", ".join(schema['properties'].keys())
                    desc += f" (параметры: {params})"
            descriptions.append(desc)
        
        return "\n".join(descriptions)


class YandexGPTChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None, 
                 weather_server_path: str = "mcp_server_weather.py"):
        """
        Инициализация клиента YandexGPT SDK с поддержкой нескольких MCP серверов
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # MCP клиенты
        self.weather_client = MCPWeatherClient(weather_server_path)
        self.deepwiki_client = MCPDeepWikiClient()
        self.mcp_started = False
        
        # История сообщений
        self.messages = [
            {"role": "system", "text": SYSTEM_PROMPT}
        ]
        
        # Статистика
        self.exchange_count = 0
        self.total_tokens = 0

    async def start_mcp(self):
        """Запуск всех MCP серверов"""
        if not self.mcp_started:
            await self.weather_client.start()
            await self.deepwiki_client.start()
            
            # Обновляем системный промпт с информацией об инструментах DeepWiki
            tools_desc = self.deepwiki_client.get_tools_description()
            updated_prompt = SYSTEM_PROMPT + f"\n\nИнструменты DeepWiki:\n{tools_desc}"
            self.messages[0] = {"role": "system", "text": updated_prompt}
            
            self.mcp_started = True
    
    async def stop_mcp(self):
        """Остановка всех MCP серверов"""
        if self.mcp_started:
            await self.weather_client.stop()
            await self.deepwiki_client.stop()
            self.mcp_started = False

    def parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Парсит ответ модели на предмет вызова инструмента.
        Если модель запрашивает несколько инструментов, возвращает только первый.
        """
        if "USE_TOOL:" not in text:
            return None
        
        try:
            # Ищем первый вызов инструмента
            tool_start = text.find("USE_TOOL:")
            if tool_start == -1:
                return None
            
            # Ищем ARGUMENTS для первого инструмента
            args_start = text.find("ARGUMENTS:", tool_start)
            if args_start == -1:
                return None
            
            # Извлекаем название инструмента
            tool_name = text[tool_start + 9:args_start].strip()
            
            # Ищем следующий USE_TOOL (если есть), чтобы знать где заканчивается первый
            next_tool = text.find("USE_TOOL:", args_start)
            
            # Определяем границы для парсинга JSON
            if next_tool != -1:
                # Есть ещё инструменты - берём текст только до следующего USE_TOOL
                args_text = text[args_start + 10:next_tool].strip()
            else:
                # Это единственный инструмент
                args_text = text[args_start + 10:].strip()
            
            # Извлекаем JSON
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

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Выполняет вызов инструмента в соответствующем MCP сервере
        """
        # Проверяем, это инструмент погоды
        if tool_name == "get_weather_forecast":
            print(f"🌤️  Вызов инструмента погоды: {tool_name}")
            print(f"📦 Аргументы: {json.dumps(arguments, ensure_ascii=False)}\n")
            
            result = await self.weather_client.get_weather(
                lat=arguments['lat'],
                lon=arguments['lon'],
                product=arguments.get('product', 'civil')
            )
            
            print(f"✅ Результат от погоды:\n{result[:300]}...\n")
            return result
        
        # Иначе это инструмент DeepWiki
        else:
            print(f"🔍 Вызов инструмента DeepWiki: {tool_name}")
            print(f"📦 Аргументы: {json.dumps(arguments, ensure_ascii=False)}\n")
            
            result = await self.deepwiki_client.call_tool(tool_name, arguments)
            
            print(f"✅ Результат от DeepWiki:\n{result[:300]}...\n")
            return result

    async def ask(self, question: str, model: str = "yandexgpt-lite"):
        """
        Отправляет вопрос модели и возвращает ответ
        Автоматически обрабатывает последовательные вызовы инструментов
        """
        if not self.mcp_started:
            await self.start_mcp()
        
        self.messages.append({"role": "user", "text": question})
        
        try:
            gpt_model = self.sdk.models.completions(model)
            gpt_model = gpt_model.configure(temperature=TEMPERATURE)
            
            # Итеративный процесс вызова инструментов
            max_iterations = 5
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                print(f"🔄 Итерация {iteration}\n")
                
                result = gpt_model.run(self.messages)
                
                answer_text = ""
                for alternative in result:
                    answer_text = alternative.text
                    break
                
                # Проверяем, нужно ли вызвать инструмент
                tool_call = self.parse_tool_call(answer_text)
                
                if not tool_call:
                    # Нет вызова инструмента - это финальный ответ
                    self.messages.append({"role": "assistant", "text": answer_text})
                    self.exchange_count += 1
                    return answer_text
                
                # Выполняем вызов инструмента
                tool_result = await self.execute_tool(
                    tool_call['tool'],
                    tool_call['arguments']
                )
                
                # Добавляем результат в историю
                self.messages.append({"role": "assistant", "text": answer_text})
                tool_message = f"Результат от инструмента {tool_call['tool']}:\n{tool_result}"
                self.messages.append({"role": "user", "text": tool_message})
                
                print("💬 Продолжаем обработку...\n")
            
            # Если достигли максимума итераций
            return "⚠️ Достигнуто максимальное количество вызовов инструментов"
            
        except Exception as e:
            return f"❌ Ошибка при запросе: {str(e)}"


async def interactive_chat():
    """
    Основной цикл общения в консоли с поддержкой нескольких MCP серверов
    """
    print("=" * 70)
    print("🤖 YandexGPT CLI с поддержкой Weather + DeepWiki MCP")
    print("=" * 70)
    print("\nПримеры запросов:")
    print("  • Какая погода в Москве и что сейчас происходит в городе?")
    print("  • Расскажи о погоде в Лондоне и найди информацию о React")
    print("  • Погода в Нью-Йорке + как работает async в Python")
    print("\n💡 DeepWiki специализируется на документации GitHub репозиториев")
    print("   Спрашивайте о библиотеках: React, FastAPI, Three.js и т.д.")
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