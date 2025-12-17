import os
import asyncio
import json
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import aiohttp

load_dotenv()

# Константы
#WEATHER_CHECK_INTERVAL = 5 * 60  # 5 минут в секундах
WEATHER_CHECK_INTERVAL = 5
SUMMARY_INTERVAL = 30
#SUMMARY_INTERVAL = 5 * 60 * 60  # 5 часов в секундах
WEATHER_DATA_FILE = "weather_data.json"
MOSCOW_COORDS = {"lat": 55.7558, "lon": 37.6173}

SYSTEM_PROMPT = """Ты - аналитик погоды. Проанализируй данные о погоде за последние несколько часов 
и создай краткое саммари (3-5 предложений) с основными наблюдениями: 
- общая тенденция (теплеет/холодает)
- изменения в осадках
- другие важные изменения
Отвечай кратко и по делу."""

class MCPWeatherClient:
    """Клиент для работы с MCP сервером погоды"""
    
    def __init__(self, server_script_path: str = "/Users/aaovchin13/Documents/ai_advent/1_1_python_simple_api_call/mcp_server_day_12/mcp_server_weather.py"):
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
            
            weather_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    weather_text += content.text
            
            return weather_text
            
        except Exception as e:
            return f"❌ Ошибка при получении погоды: {str(e)}"


class TelegramNotifier:
    """Отправка уведомлений в Telegram"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    async def send_message(self, text: str):
        """Отправить сообщение в Telegram"""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML"
                }
                async with session.post(self.api_url, json=payload) as response:
                    if response.status == 200:
                        print("✅ Сообщение отправлено в Telegram")
                        return True
                    else:
                        print(f"❌ Ошибка отправки в Telegram: {response.status}")
                        return False
        except Exception as e:
            print(f"❌ Ошибка при отправке в Telegram: {str(e)}")
            return False


class WeatherDataManager:
    """Управление данными о погоде"""
    
    def __init__(self, filename: str = WEATHER_DATA_FILE):
        self.filename = filename
        """Создать файл, если его нет"""
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump([], f)
    
    def load_data(self) -> list:
        """Загрузить данные из файла"""
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Ошибка чтения файла: {e}")
            return []
    
    def save_entry(self, weather_data: str):
        """Сохранить новую запись о погоде"""
        data = self.load_data()
        entry = {
            "timestamp": datetime.now().isoformat(),
            "weather": weather_data
        }
        data.append(entry)
        
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"💾 Сохранено записей: {len(data)}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def clear_data(self):
        """Очистить данные после создания саммари"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump([], f)
            print("🗑️ Данные очищены")
        except Exception as e:
            print(f"❌ Ошибка очистки данных: {e}")


class YandexGPTSummarizer:
    """Создание саммари с помощью YandexGPT"""
    
    def __init__(self, folder_id: Optional[str] = None, api_key: Optional[str] = None):
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        if not self.folder_id or not self.api_key:
            raise ValueError("Не указан folder_id или api_key")
        
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
    
    async def create_summary(self, weather_data: list) -> str:
        """Создать саммари на основе данных о погоде"""
        if not weather_data:
            return "Нет данных для анализа"
        
        # Формируем промпт с данными
        data_text = "\n\n".join([
            f"Время: {entry['timestamp']}\nДанные: {entry['weather']}" 
            for entry in weather_data
        ])
        
        messages = [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user", "text": f"Проанализируй данные о погоде:\n\n{data_text}"}
        ]
        
        try:
            model = self.sdk.models.completions("yandexgpt-lite")
            model = model.configure(temperature=0.6)
            result = model.run(messages)
            
            for alternative in result:
                return alternative.text
            
            return "Не удалось создать саммари"
            
        except Exception as e:
            return f"❌ Ошибка при создании саммари: {str(e)}"


class WeatherMonitor:
    """Основной класс мониторинга погоды"""
    
    def __init__(self):
        self.mcp_client = MCPWeatherClient()
        self.data_manager = WeatherDataManager()
        self.summarizer = YandexGPTSummarizer()
        
        # Telegram
        bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not bot_token or not chat_id:
            raise ValueError("Не указан TELEGRAM_BOT_TOKEN или TELEGRAM_CHAT_ID в .env")
        
        self.telegram = TelegramNotifier(bot_token, chat_id)
        
        self.last_summary_time = datetime.now()
    
    async def fetch_and_save_weather(self):
        """Получить погоду и сохранить"""
        print(f"\n{'='*70}")
        print(f"🌤️  Запрос погоды в Москве - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        weather_data = await self.mcp_client.get_weather(
            lat=MOSCOW_COORDS["lat"],
            lon=MOSCOW_COORDS["lon"]
        )
        
        print(f"📡 Получено: {weather_data[:200]}...")
        self.data_manager.save_entry(weather_data)
    
    async def check_and_send_summary(self):
        """Проверить, нужно ли отправить саммари"""
        elapsed = (datetime.now() - self.last_summary_time).total_seconds()
        
        if elapsed >= SUMMARY_INTERVAL:
            print(f"\n{'='*70}")
            print(f"📊 Создание саммари - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*70}")
            
            data = self.data_manager.load_data()
            
            if data:
                summary = await self.summarizer.create_summary(data)
                print(f"\n📝 Саммари:\n{summary}\n")
                
                # Форматируем сообщение для Telegram
                message = f"<b>🌤️ Сводка по погоде в Москве</b>\n\n{summary}\n\n<i>Период: последние 5 часов</i>"
                
                await self.telegram.send_message(message)
                
                # Очищаем данные после отправки
                self.data_manager.clear_data()
            else:
                print("⚠️ Нет данных для создания саммари")
            
            self.last_summary_time = datetime.now()
    
    async def run(self):
        """Основной цикл мониторинга"""
        print("=" * 70)
        print("🤖 Мониторинг погоды в Москве")
        print("=" * 70)
        print(f"⏰ Проверка погоды: каждые {WEATHER_CHECK_INTERVAL // 60} минут")
        print(f"📊 Отправка саммари: каждые {SUMMARY_INTERVAL // 3600} часов")
        print("=" * 70)
        print("\nНажмите CTRL+C для остановки\n")
        
        try:
            # Запускаем MCP сервер
            await self.mcp_client.start()
            
            # Сразу получаем первые данные
            await self.fetch_and_save_weather()
            
            while True:
                # Ждем 5 минут
                await asyncio.sleep(WEATHER_CHECK_INTERVAL)
                
                # Получаем и сохраняем погоду
                await self.fetch_and_save_weather()
                
                # Проверяем, не пора ли отправить саммари
                await self.check_and_send_summary()
                
        except KeyboardInterrupt:
            print("\n\n👋 Остановка мониторинга...")
        except Exception as e:
            print(f"\n❌ Ошибка: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            await self.mcp_client.stop()


async def main():
    """Точка входа"""
    monitor = WeatherMonitor()
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())