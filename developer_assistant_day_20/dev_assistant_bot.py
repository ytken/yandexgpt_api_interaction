import os
import sys
import asyncio
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

# Импортируем RAG систему
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_16'))
from text_to_embedding import YandexDocumentIndexer
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_17'))
from rag_classes import YandexRAGSystem

SYSTEM_PROMPT = """Ты — ассистент разработчика с доступом к документации проекта и GitHub репозиторию.

ВОЗМОЖНОСТИ:
1. Отвечать на вопросы о проекте, используя документацию из базы знаний
2. Искать код в репозитории GitHub
3. Просматривать файлы проекта
4. Анализировать коммиты и Pull Requests
5. Создавать issues

ДОСТУПНЫЕ ИНСТРУМЕНТЫ GITHUB:
- search_code: поиск кода по ключевым словам в репозитории
- get_file_content: получить содержимое конкретного файла
- list_repository_files: просмотреть структуру директорий
- get_recent_commits: показать последние коммиты
- create_issue: создать новый issue
- get_pull_requests: посмотреть PR

КОГДА ИСПОЛЬЗОВАТЬ ИНСТРУМЕНТЫ:
- Если пользователь спрашивает о коде или просит найти функцию → search_code
- Если нужно показать конкретный файл → get_file_content
- Если спрашивают о структуре проекта → list_repository_files
- Если интересуют последние изменения → get_recent_commits
- Для создания задачи → create_issue

ФОРМАТ ВЫЗОВА ИНСТРУМЕНТА:
USE_TOOL: <имя_инструмента>
ARGUMENTS: {JSON с аргументами}

СТРАТЕГИЯ ОТВЕТА:
1. Сначала используй документацию из RAG для понимания контекста проекта
2. Если нужны примеры кода или детали реализации → используй GitHub инструменты
3. Комбинируй информацию из документации и кода для полного ответа
4. Всегда давай конкретные примеры и ссылки на код"""

class MCPGitHubClient:
    """Клиент для работы с MCP сервером GitHub"""
    
    def __init__(self, server_script_path: str = "mcp_server_github.py"):
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
        
        print("🚀 Запуск MCP GitHub сервера...")
        self._client_context = stdio_client(server_params)
        self.read_stream, self.write_stream = await self._client_context.__aenter__()
        
        self._session_context = ClientSession(self.read_stream, self.write_stream)
        self.session = await self._session_context.__aenter__()
        
        await self.session.initialize()
        print("✅ MCP GitHub сервер готов\n")
        
    async def stop(self):
        """Остановка MCP сервера"""
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._client_context:
            await self._client_context.__aexit__(None, None, None)
        print("🛑 MCP сервер остановлен")
    
    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        """Вызвать инструмент GitHub"""
        try:
            result = await self.session.call_tool(tool_name, arguments=arguments)
            
            # Извлекаем текст из результата
            result_text = ""
            for content in result.content:
                if hasattr(content, 'text'):
                    result_text += content.text
            
            return result_text
            
        except Exception as e:
            return f"❌ Ошибка вызова инструмента {tool_name}: {str(e)}"

class DevAssistantBot:
    """Ассистент разработчика с RAG и GitHub интеграцией"""
    
    def __init__(
        self, 
        index_path: str,
        github_repo: str,
        folder_id: str = None, 
        api_key: str = None,
        mcp_server_path: str = "mcp_server_github.py"
    ):
        """
        Args:
            index_path: Путь к индексу документации
            github_repo: Имя GitHub репозитория (owner/repo)
            folder_id: ID папки Yandex Cloud
            api_key: API ключ Yandex Cloud
            mcp_server_path: Путь к MCP серверу
        """
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        self.github_repo = github_repo
        
        print("🚀 Инициализация ассистента разработчика...")
        
        # RAG система для документации
        self.rag_system = YandexRAGSystem(index_path)
        
        # Yandex GPT
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        
        # MCP клиент для GitHub
        self.mcp_client = MCPGitHubClient(mcp_server_path)
        self.mcp_started = False
        
        # История диалога
        self.conversation_history: List[Dict[str, str]] = []
        
        print("✅ Ассистент готов к работе!\n")
    
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
    
    def add_to_history(self, role: str, content: str):
        """Добавляет сообщение в историю"""
        self.conversation_history.append({
            'role': role,
            'content': content
        })
    
    def get_history_context(self, max_messages: int = 6) -> str:
        """Формирует контекст из истории диалога"""
        if not self.conversation_history:
            return ""
        
        recent_history = self.conversation_history[-max_messages:]
        
        history_text = "ИСТОРИЯ ДИАЛОГА:\n"
        for msg in recent_history:
            role_label = "Пользователь" if msg['role'] == 'user' else "Ассистент"
            history_text += f"{role_label}: {msg['content']}\n"
        
        return history_text
    
    def parse_tool_call(self, text: str) -> Optional[dict]:
        """
        Парсит ответ модели на предмет вызова инструмента
        """
        if "USE_TOOL:" not in text:
            return None
        
        try:
            tool_start = text.find("USE_TOOL:")
            args_start = text.find("ARGUMENTS:", tool_start)
            
            if tool_start == -1 or args_start == -1:
                return None
            
            tool_name = text[tool_start + 9:args_start].strip()
            args_text = text[args_start + 10:].strip()
            
            json_start = args_text.find("{")
            json_end = args_text.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                return None
            
            args_json = args_text[json_start:json_end]
            arguments = json.loads(args_json)
            
            # Автоматически добавляем repo, если не указан
            if "repo" not in arguments and tool_name in [
                "search_code", "get_file_content", "list_repository_files",
                "get_recent_commits", "create_issue", "get_pull_requests"
            ]:
                arguments["repo"] = self.github_repo
            
            return {
                "tool": tool_name,
                "arguments": arguments
            }
        except Exception as e:
            print(f"⚠️  Ошибка парсинга: {e}")
            return None
    
    async def ask(
        self, 
        question: str, 
        top_k: int = 3, 
        temperature: float = 0.3,
        model: str = "yandexgpt"
    ) -> Dict:
        """
        Задает вопрос ассистенту с использованием RAG и GitHub
        """
        # Запускаем MCP если не запущен
        if not self.mcp_started:
            await self.start_mcp()
        
        # Добавляем вопрос в историю
        self.add_to_history('user', question)
        
        # Ищем релевантную документацию
        print("📚 Поиск в документации...")
        context_chunks = self.rag_system.search_relevant_chunks(question, top_k=top_k)
        
        # Формируем контекст из документации
        doc_context = "\n\n".join([
            f"Документ {i+1} (релевантность: {chunk['similarity']:.3f}):\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks)
        ])
        
        # Получаем историю
        history_context = self.get_history_context()
        
        # Формируем промпт
        prompt = f"""{SYSTEM_PROMPT}

{history_context}

ДОКУМЕНТАЦИЯ ПРОЕКТА:
{doc_context}

GITHUB РЕПОЗИТОРИЙ: {self.github_repo}

ВОПРОС ПОЛЬЗОВАТЕЛЯ:
{question}

ОТВЕТ:"""
        
        # Первый запрос к модели
        print("💭 Анализ вопроса...")
        llm_model = self.sdk.models.completions(model)
        llm_model = llm_model.configure(temperature=temperature, max_tokens=2000)
        
        result = llm_model.run(prompt)
        answer = result.alternatives[0].text
        
        # Проверяем, нужно ли вызвать GitHub инструменты
        tool_call = self.parse_tool_call(answer)
        
        github_results = []
        
        if tool_call:
            print(f"🔧 Вызов GitHub инструмента: {tool_call['tool']}")
            print(f"📦 Аргументы: {json.dumps(tool_call['arguments'], ensure_ascii=False, indent=2)}\n")
            
            # Вызываем инструмент
            tool_result = await self.mcp_client.call_tool(
                tool_call['tool'],
                tool_call['arguments']
            )
            
            print(f"✅ Результат получен\n")
            github_results.append({
                'tool': tool_call['tool'],
                'result': tool_result
            })
            
            # Формируем финальный промпт с результатами
            final_prompt = f"""{prompt}

РЕЗУЛЬТАТ МОДЕЛИ:
{answer}

РЕЗУЛЬТАТ ИЗ GITHUB ({tool_call['tool']}):
{tool_result}

Теперь дай финальный ответ пользователю, объединив информацию из документации и GitHub:"""
            
            print("💬 Генерация финального ответа...\n")
            final_result = llm_model.run(final_prompt)
            final_answer = final_result.alternatives[0].text
            
            self.add_to_history('assistant', final_answer)
            
            return {
                'answer': final_answer,
                'context_chunks': context_chunks,
                'github_results': github_results,
                'history_length': len(self.conversation_history),
                'used_github': True
            }
        
        # Если инструменты не нужны
        self.add_to_history('assistant', answer)
        
        return {
            'answer': answer,
            'context_chunks': context_chunks,
            'github_results': [],
            'history_length': len(self.conversation_history),
            'used_github': False
        }
    
    def clear_history(self):
        """Очищает историю диалога"""
        self.conversation_history = []
        print("🗑️  История очищена\n")
    
    def show_history(self):
        """Показывает историю диалога"""
        if not self.conversation_history:
            print("📭 История пуста\n")
            return
        
        print("\n" + "="*80)
        print("📚 ИСТОРИЯ ДИАЛОГА")
        print("="*80)
        
        for i, msg in enumerate(self.conversation_history, 1):
            role = "👤 Пользователь" if msg['role'] == 'user' else "🤖 Ассистент"
            content = msg['content'][:200] + ('...' if len(msg['content']) > 200 else '')
            print(f"\n{i}. {role}:")
            print(f"   {content}")
        
        print("\n" + "="*80 + "\n")

def print_welcome():
    """Приветственное сообщение"""
    print("\n" + "="*80)
    print("🤖 АССИСТЕНТ РАЗРАБОТЧИКА")
    print("="*80)
    print("""
Команды:
  Вопросы - просто спрашивай о проекте, коде или стиле
  /search <запрос> - поиск кода в репозитории
  /file <путь> - показать содержимое файла
  /structure [путь] - показать структуру директории
  /commits - показать последние коммиты
  /prs - показать открытые PR
  /history - история диалога
  /clear - очистить историю
  /help - эта справка
  /exit - выход

Примеры:
  • Как правильно обрабатывать ошибки в нашем проекте?
  • Покажи пример использования RAG класса
  • /search YandexRAGSystem
  • /file src/main.py
    """)
    print("="*80 + "\n")

async def run_chat(index_path: str, github_repo: str):
    """Запускает интерактивный чат"""
    bot = None
    
    try:
        bot = DevAssistantBot(index_path, github_repo)
        print_welcome()
        
        while True:
            try:
                user_input = input("👤 Вы: ").strip()
            except EOFError:
                print("\n👋 До свидания!")
                break
            
            if not user_input:
                continue
            
            # Команды выхода
            if user_input.lower() in ['/exit', '/quit']:
                print("\n👋 До свидания!")
                break
            
            # Команды управления
            elif user_input.lower() == '/clear':
                bot.clear_history()
                continue
            
            elif user_input.lower() == '/history':
                bot.show_history()
                continue
            
            elif user_input.lower() == '/help':
                print_welcome()
                continue
            
            # GitHub команды
            elif user_input.lower().startswith('/search '):
                query = user_input[8:].strip()
                question = f"Найди в коде: {query}"
            
            elif user_input.lower().startswith('/file '):
                file_path = user_input[6:].strip()
                question = f"Покажи содержимое файла: {file_path}"
            
            elif user_input.lower().startswith('/structure'):
                parts = user_input.split(maxsplit=1)
                path = parts[1] if len(parts) > 1 else ""
                question = f"Покажи структуру директории: {path}"
            
            elif user_input.lower() == '/commits':
                question = "Покажи последние коммиты в репозитории"
            
            elif user_input.lower() == '/prs':
                question = "Покажи открытые pull requests"
            
            else:
                question = user_input
            
            # Обрабатываем вопрос
            print()
            result = await bot.ask(question)
            
            # Выводим ответ
            print("\n" + "─"*80)
            print("🤖 Ассистент:")
            print("─"*80)
            print(result['answer'])
            
            # Показываем источники из документации
            if result['context_chunks']:
                print("\n" + "─"*80)
                print("📚 ИСТОЧНИКИ ИЗ ДОКУМЕНТАЦИИ:")
                print("─"*80)
                for i, chunk in enumerate(result['context_chunks'], 1):
                    print(f"\n{i}. Релевантность: {chunk['similarity']:.3f}")
                    print(f"   Источник: {chunk['metadata'].get('source', 'unknown')}")
                    print(f"   Фрагмент: \"{chunk['text'][:100]}...\"")
            
            # Показываем результаты GitHub
            if result['github_results']:
                print("\n" + "─"*80)
                print("🐙 РЕЗУЛЬТАТЫ ИЗ GITHUB:")
                print("─"*80)
                for gh_result in result['github_results']:
                    print(f"\n🔧 Инструмент: {gh_result['tool']}")
                    print(gh_result['result'][:500])
                    if len(gh_result['result']) > 500:
                        print("...")
            
            # Статистика
            print("\n" + "─"*80)
            status = "🐙 GitHub" if result['used_github'] else "📚 RAG"
            print(f"📊 Статистика: {status} | История: {result['history_length']} сообщений")
            print("─"*80 + "\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if bot:
            await bot.stop_mcp()

async def main():
    """Основная функция"""
    
    print("="*80)
    print("🔧 НАСТРОЙКА АССИСТЕНТА РАЗРАБОТЧИКА")
    print("="*80 + "\n")
    
    # Параметры
    doc_filename = "project_docs.txt"  # Ваша документация проекта
    index_filename = doc_filename.replace('.txt', '_index.json')
    github_repo = os.getenv("GITHUB_REPO", "your-username/your-repo")
    
    print(f"📁 Документация: {doc_filename}")
    print(f"🐙 GitHub репозиторий: {github_repo}\n")
    
    # Проверяем GitHub токен
    if not os.getenv("GITHUB_TOKEN"):
        print("⚠️  GITHUB_TOKEN не найден в переменных окружения!")
        print("Установите токен: export GITHUB_TOKEN=your_token\n")
        return
    
    # Проверяем/создаем индекс
    if not os.path.exists(index_filename):
        print(f"📄 Создаем индекс документации...")
        
        try:
            indexer = YandexDocumentIndexer()
            index_path = indexer.process_file(doc_filename, chunk_size=400, overlap=50)
            print(f"✅ Индекс создан: {index_path}\n")
        except Exception as e:
            print(f"❌ Ошибка создания индекса: {str(e)}")
            return
    else:
        print(f"✅ Индекс найден: {index_filename}\n")
        index_path = index_filename
    
    # Запускаем чат
    await run_chat(index_path, github_repo)

if __name__ == "__main__":
    asyncio.run(main())