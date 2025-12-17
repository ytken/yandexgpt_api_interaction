import asyncio
import httpx
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

# Создаем экземпляр MCP сервера
app = Server("7timer-weather")

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Список доступных инструментов"""
    return [
        Tool(
            name="get_weather_forecast",
            description="Получить прогноз погоды от 7Timer по координатам (широта и долгота)",
            inputSchema={
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "Широта (latitude), например 55.7558",
                        "minimum": -90,
                        "maximum": 90
                    },
                    "lon": {
                        "type": "number",
                        "description": "Долгота (longitude), например 37.6173",
                        "minimum": -180,
                        "maximum": 180
                    },
                    "product": {
                        "type": "string",
                        "description": "Тип прогноза: civil (гражданский), civillight (упрощенный), astro (астрономический)",
                        "enum": ["civil", "civillight", "astro"],
                        "default": "civil"
                    },
                    "output": {
                        "type": "string",
                        "description": "Формат вывода",
                        "enum": ["json", "xml"],
                        "default": "json"
                    }
                },
                "required": ["lat", "lon"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Обработка вызова инструмента"""
    if name != "get_weather_forecast":
        raise ValueError(f"Неизвестный инструмент: {name}")
    
    lat = arguments["lat"]
    lon = arguments["lon"]
    product = arguments.get("product", "civil")
    output = arguments.get("output", "json")
    
    # Формируем URL для запроса к 7Timer API
    # ВАЖНО: используем правильный URL, который не делает редирект
    url = f"https://www.7timer.info/bin/{product}.php"
    params = {
        "lon": lon,
        "lat": lat,
        "output": output,
        "unit": "metric",
        "ac": 0,
        "tzshift": 0
    }
    
    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # Проверяем, что получили ответ
            if not response.text:
                return [TextContent(
                    type="text",
                    text=f"Ошибка: API вернул пустой ответ для координат ({lat}, {lon})"
                )]
            
            if output == "json":
                try:
                    data = response.json()
                    
                    # Форматируем красивый ответ
                    result = f"🌤️ Прогноз погоды для координат ({lat}, {lon})\n\n"
                    
                    if "dataseries" in data and len(data["dataseries"]) > 0:
                        result += "📊 Прогноз на ближайшие дни:\n\n"
                        
                        for i, forecast in enumerate(data["dataseries"][:5]):  # Первые 5 точек
                            timepoint = forecast.get("timepoint", i * 3)
                            result += f"⏰ +{timepoint} часов:\n"
                            result += f"   🌡️  Температура: {forecast.get('temp2m', 'н/д')}°C\n"
                            result += f"   ☁️  Облачность: {forecast.get('cloudcover', 'н/д')}\n"
                            result += f"   💧 Осадки: {forecast.get('prec_type', 'нет')}\n"
                            
                            wind = forecast.get('wind10m', {})
                            if wind:
                                result += f"   💨 Ветер: {wind.get('speed', 'н/д')} м/с, направление {wind.get('direction', 'н/д')}\n"
                            
                            result += "\n"
                    else:
                        result += "⚠️ Нет данных прогноза в ответе\n"
                        result += f"Полный ответ: {response.text[:500]}"
                    
                except Exception as e:
                    result = f"Ошибка парсинга JSON от API:\n{str(e)}\n\nОтвет API:\n{response.text[:500]}"
            else:
                result = response.text
            
            return [TextContent(
                type="text",
                text=result
            )]
    
    except httpx.HTTPError as e:
        return [TextContent(
            type="text",
            text=f"❌ Ошибка при запросе к API: {str(e)}\nURL: {url}\nParams: {params}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Неожиданная ошибка: {str(e)}\nURL: {url}"
        )]

async def main():
    """Запуск сервера"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())