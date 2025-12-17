"""
Тестовый клиент для проверки MCP сервера 7Timer Weather
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp_server():
    """Основная функция тестирования"""
    
    # Параметры для запуска вашего MCP сервера
    server_params = StdioServerParameters(
        command="python3",  # или "python3" в зависимости от системы
        args=["/Users/aaovchin13/Documents/ai_advent/1_1_python_simple_api_call/mcp_server_day_12/mcp_server_weather.py"],  # путь к вашему серверу
        env=None
    )
    
    print("🚀 Запуск MCP сервера...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Инициализация
            await session.initialize()
            print("✅ Сервер инициализирован")
            
            # 1. Получаем список доступных инструментов
            print("\n📋 Получение списка инструментов...")
            tools = await session.list_tools()
            print(f"Найдено инструментов: {len(tools.tools)}")
            
            for tool in tools.tools:
                print(f"\n🔧 Инструмент: {tool.name}")
                print(f"   Описание: {tool.description}")
                print(f"   Схема: {json.dumps(tool.inputSchema, indent=2, ensure_ascii=False)}")
            
            # 2. Тестируем получение погоды для разных городов
            test_locations = [
                {"name": "Москва", "lat": 55.7558, "lon": 37.6173},
                {"name": "Санкт-Петербург", "lat": 59.9311, "lon": 30.3609},
                {"name": "Лондон", "lat": 51.5074, "lon": -0.1278},
                {"name": "Нью-Йорк", "lat": 40.7128, "lon": -74.0060},
            ]
            
            print("\n" + "="*60)
            print("🌍 ТЕСТИРОВАНИЕ ПРОГНОЗОВ ПОГОДЫ")
            print("="*60)
            
            for location in test_locations:
                print(f"\n📍 Тест для города: {location['name']}")
                print(f"   Координаты: ({location['lat']}, {location['lon']})")
                
                try:
                    result = await session.call_tool(
                        "get_weather_forecast",
                        arguments={
                            "lat": location["lat"],
                            "lon": location["lon"],
                            "product": "civil",
                            "output": "json"
                        }
                    )
                    
                    print("   ✅ Запрос выполнен успешно")
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print("\n" + content.text)
                    
                except Exception as e:
                    print(f"   ❌ Ошибка: {str(e)}")
                
                # Небольшая пауза между запросами
                await asyncio.sleep(1)
            
            # 3. Тест с разными типами прогнозов
            print("\n" + "="*60)
            print("🔬 ТЕСТИРОВАНИЕ РАЗНЫХ ТИПОВ ПРОГНОЗОВ")
            print("="*60)
            
            products = ["civil", "civillight", "astro"]
            test_location = {"name": "Москва", "lat": 55.7558, "lon": 37.6173}
            
            for product in products:
                print(f"\n📊 Тип прогноза: {product}")
                
                try:
                    result = await session.call_tool(
                        "get_weather_forecast",
                        arguments={
                            "lat": test_location["lat"],
                            "lon": test_location["lon"],
                            "product": product,
                            "output": "json"
                        }
                    )
                    
                    print(f"   ✅ Прогноз '{product}' получен успешно")
                    
                except Exception as e:
                    print(f"   ❌ Ошибка: {str(e)}")
                
                await asyncio.sleep(1)
            
            # 4. Тест граничных значений координат
            print("\n" + "="*60)
            print("🧪 ТЕСТИРОВАНИЕ ГРАНИЧНЫХ ЗНАЧЕНИЙ")
            print("="*60)
            
            edge_cases = [
                {"name": "Северный полюс", "lat": 90, "lon": 0},
                {"name": "Южный полюс", "lat": -90, "lon": 0},
                {"name": "Экватор", "lat": 0, "lon": 0},
            ]
            
            for case in edge_cases:
                print(f"\n🔍 Тест: {case['name']}")
                print(f"   Координаты: ({case['lat']}, {case['lon']})")
                
                try:
                    result = await session.call_tool(
                        "get_weather_forecast",
                        arguments={
                            "lat": case["lat"],
                            "lon": case["lon"]
                        }
                    )
                    print("   ✅ Запрос выполнен")
                    
                except Exception as e:
                    print(f"   ❌ Ошибка: {str(e)}")
                
                await asyncio.sleep(1)
            
            print("\n" + "="*60)
            print("✨ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
            print("="*60)

if __name__ == "__main__":
    print("🧪 MCP Server Test Suite")
    print("Убедитесь, что файл сервера называется '7timer_server.py'")
    print("и находится в текущей директории\n")
    
    try:
        asyncio.run(test_mcp_server())
    except KeyboardInterrupt:
        print("\n⚠️  Тестирование прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()