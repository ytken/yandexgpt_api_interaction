import os
import asyncio
import json
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

SYSTEM_PROMPT = """Ты - полезный ассистент с доступом к инструментам управления Docker контейнерами.

У тебя есть следующие инструменты:

1. start_container - запускает Docker контейнер
   Параметры:
   - image (строка, обязательно): имя образа (например: nginx:latest, postgres:15)
   - name (строка, опционально): имя контейнера
   - ports (объект, опционально): маппинг портов {"80/tcp": 8080}
   - environment (объект, опционально): переменные окружения {"KEY": "value"}
   - volumes (объект, опционально): монтирование volumes
   - detach (boolean, опционально): запуск в фоне (по умолчанию: true)
   - remove (boolean, опционально): авто-удаление после остановки

2. stop_container - останавливает контейнер
   Параметры:
   - container_id (строка, обязательно): ID или имя контейнера
   - timeout (число, опционально): таймаут в секундах

3. list_containers - показывает список контейнеров
   Параметры:
   - all (boolean, опционально): показать все включая остановленные

4. remove_container - удаляет контейнер
   Параметры:
   - container_id (строка, обязательно): ID или имя контейнера
   - force (boolean, опционально): принудительное удаление

5. container_logs - получает логи контейнера
   Параметры:
   - container_id (строка, обязательно): ID или имя контейнера
   - tail (число, опционально): количество последних строк

6. container_stats - получает статистику контейнера
   Параметры:
   - container_id (строка, обязательно): ID или имя контейнера

Когда пользователь просит выполнить операцию с Docker, используй формат:
USE_TOOL: <имя_инструмента>
ARGUMENTS: {<json с аргументами>}

Примеры:
- "Запусти nginx" -> USE_TOOL: start_container, ARGUMENTS: {"image": "nginx:latest", "ports": {"80/tcp": 8080}}
- "Останови контейнер nginx" -> USE_TOOL: stop_container, ARGUMENTS: {"container_id": "nginx"}
- "Покажи все контейнеры" -> USE_TOOL: list_containers, ARGUMENTS: {"all": true}

После получения результата от инструмента, интерпретируй его для пользователя."""

TEMPERATURE = 0.7

class MCPDockerClient:
    """Клиент для работы с MCP сервером Docker"""
    
    def __init__(self, server_script_path: str = "mcp_server_docker.py"):
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
        
        print("🐳 Запуск MCP Docker сервера...")
        self._client_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self._client_context.__aenter__()
        
        self._session_context = ClientSession(self.read_stream, self.write_stream)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
        print("✅ MCP Docker сервер запущен и готов к работе\n")
        
    async def stop(self):
        """Остановка MCP сервера"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
        print("🛑 MCP Docker сервер остановлен")
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Вызов инструмента Docker"""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # Извлекаем текст из результата
            result_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += content.text
            
            return result_text
            
        except Exception as e:
            return f"❌ Ошибка при вызове {tool_name}: {str(e)}"

class YandexGPTDockerChat:
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None, 
                 mcp_server_path: str = "mcp_server_docker.py"):
        """
        Инициализация клиента YandexGPT SDK с поддержкой Docker MCP
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # MCP клиент
        self.mcp_client = MCPDockerClient(mcp_server_path)
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
        Автоматически обрабатывает вызовы инструментов Docker
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
                
                # Вызываем инструмент Docker
                docker_result = await self.mcp_client.call_tool(
                    tool_name=tool_call['tool'],
                    arguments=tool_call['arguments']
                )
                
                print(f"🐳 Получен результат от Docker:\n{docker_result}\n")
                
                # Добавляем результат инструмента в историю
                tool_result_message = f"Результат от инструмента {tool_call['tool']}:\n{docker_result}"
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
    Основной цикл общения в консоли с поддержкой Docker MCP
    """
    print("=" * 70)
    print("🐳 YandexGPT CLI с управлением Docker контейнерами")
    print("=" * 70)
    print("\nПримеры запросов:")
    print("  • Запусти nginx контейнер на порту 8080")
    print("  • Покажи все запущенные контейнеры")
    print("  • Останови контейнер nginx")
    print("  • Запусти postgres с переменной POSTGRES_PASSWORD=secret")
    print("  • Покажи логи контейнера nginx")
    print("  • Удали контейнер nginx")
    print("\nНажмите CTRL+C для выхода.\n")
    
    client = None
    
    try:
        client = YandexGPTDockerChat()
        
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