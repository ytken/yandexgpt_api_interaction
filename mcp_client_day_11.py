import requests
import json
import uuid

class MCPHTTPClient:
    """Клиент для взаимодействия с MCP серверами по HTTP"""
    
    def __init__(self, endpoint_url):
        """
        Инициализация клиента
        
        Args:
            endpoint_url: URL MCP сервера (например, https://remote.mcpservers.org/fetch/mcp)
        """
        self.endpoint_url = endpoint_url
        self.session_id = None
        self.protocol_version = "2025-11-25"
        
    def _send_request(self, method, params=None):
        """
        Отправка JSON-RPC запроса на MCP сервер
        
        Args:
            method: Название метода JSON-RPC
            params: Параметры запроса (опционально)
            
        Returns:
            Ответ от сервера
        """
        # Формируем JSON-RPC запрос
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method
        }
        
        if params is not None:
            payload["params"] = params
        
        # Заголовки
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "MCP-Protocol-Version": self.protocol_version
        }
        
        # Добавляем session ID если есть
        if self.session_id:
            headers["MCP-Session-Id"] = self.session_id
        
        # Отправляем POST запрос
        response = requests.post(
            self.endpoint_url,
            json=payload,
            headers=headers,
            timeout=30
        )
        
        # Проверяем session ID в ответе (при инициализации)
        if "MCP-Session-Id" in response.headers:
            self.session_id = response.headers["MCP-Session-Id"]
        
        response.raise_for_status()
        
        # Обрабатываем JSON ответ
        if response.headers.get("Content-Type", "").startswith("application/json"):
            return response.json()
        
        # Если SSE, извлекаем JSON-RPC response из первого события
        if response.headers.get("Content-Type", "").startswith("text/event-stream"):
            # Простой парсинг SSE (для более сложных случаев используйте специальную библиотеку)
            for line in response.text.split('\n'):
                if line.startswith('data: '):
                    data = line[6:]  # Убираем "data: "
                    if data.strip():
                        return json.loads(data)
        
        raise ValueError(f"Неожиданный тип ответа: {response.headers.get('Content-Type')}")
    
    def initialize(self):
        """
        Инициализация соединения с MCP сервером
        
        Returns:
            Результат инициализации с информацией о сервере
        """
        params = {
            "protocolVersion": self.protocol_version,
            "capabilities": {
                "roots": {
                    "listChanged": False
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "python-mcp-client",
                "version": "1.0.0"
            }
        }
        
        response = self._send_request("initialize", params)
        
        if "error" in response:
            raise Exception(f"Ошибка инициализации: {response['error']}")
        
        return response.get("result", {})
    
    def list_tools(self):
        """
        Получение списка доступных инструментов (tools) от MCP сервера
        
        Returns:
            Список инструментов
        """
        response = self._send_request("tools/list")
        
        if "error" in response:
            raise Exception(f"Ошибка получения списка tools: {response['error']}")
        
        return response.get("result", {}).get("tools", [])


def main():
    """Пример использования"""
    # URL MCP сервера Fetch
    mcp_url = "https://remote.mcpservers.org/fetch/mcp"
    
    # Создаем клиента
    client = MCPHTTPClient(mcp_url)
    
    try:
        print("=" * 60)
        print("Подключение к MCP серверу Fetch")
        print(f"URL: {mcp_url}")
        print("=" * 60)
        
        # Инициализация
        print("\n1. Инициализация соединения...")
        init_result = client.initialize()
        print(f"✓ Сервер: {init_result.get('serverInfo', {}).get('name', 'N/A')}")
        print(f"✓ Версия: {init_result.get('serverInfo', {}).get('version', 'N/A')}")
        print(f"✓ Протокол: {init_result.get('protocolVersion', 'N/A')}")
        
        # Получаем список tools
        print("\n2. Получение списка tools...")
        tools = client.list_tools()
        
        print(f"\n✓ Найдено инструментов: {len(tools)}\n")
        
        # Выводим информацию о каждом tool
        for i, tool in enumerate(tools, 1):
            print(f"Tool #{i}:")
            print(f"  Название: {tool.get('name', 'N/A')}")
            print(f"  Описание: {tool.get('description', 'N/A')}")
            
            # Выводим схему параметров если есть
            if 'inputSchema' in tool:
                schema = tool['inputSchema']
                print(f"  Параметры:")
                if 'properties' in schema:
                    for prop_name, prop_info in schema['properties'].items():
                        required = prop_name in schema.get('required', [])
                        req_marker = " (обязательный)" if required else ""
                        print(f"    - {prop_name}{req_marker}: {prop_info.get('description', 'N/A')}")
            print()
        
        print("=" * 60)
        print("Готово!")
        
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Ошибка HTTP запроса: {e}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    main()