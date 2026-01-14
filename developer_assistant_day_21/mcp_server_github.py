import asyncio
import os
from typing import Optional
from datetime import datetime
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server

try:
    from github import Github, GithubException
except ImportError:
    print("⚠️  Требуется установить PyGithub: pip install PyGithub")
    raise

# Создаем экземпляр MCP сервера
app = Server("github-assistant")

# GitHub клиент будет инициализирован при первом использовании
_github_client: Optional[Github] = None

def get_github_client() -> Github:
    """Получить или создать GitHub клиент"""
    global _github_client
    if _github_client is None:
        token = os.getenv("GITHUB_TOKEN")
        if not token:
            raise ValueError("GITHUB_TOKEN не найден в переменных окружения")
        _github_client = Github(token)
    return _github_client

@app.list_tools()
async def list_tools() -> list[Tool]:
    """Список доступных инструментов для работы с GitHub"""
    return [
        Tool(
            name="search_code",
            description="Поиск кода в репозитории по ключевым словам",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo), например 'username/project'"
                    },
                    "query": {
                        "type": "string",
                        "description": "Поисковый запрос (ключевые слова для поиска в коде)"
                    },
                    "language": {
                        "type": "string",
                        "description": "Язык программирования (опционально), например 'python', 'javascript'"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Максимальное количество результатов (по умолчанию 5)",
                        "default": 5
                    }
                },
                "required": ["repo", "query"]
            }
        ),
        Tool(
            name="get_file_content",
            description="Получить содержимое конкретного файла из репозитория",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Путь к файлу в репозитории, например 'src/main.py'"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Ветка (по умолчанию main)",
                        "default": "main"
                    }
                },
                "required": ["repo", "path"]
            }
        ),
        Tool(
            name="list_repository_files",
            description="Получить список файлов в директории репозитория",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "path": {
                        "type": "string",
                        "description": "Путь к директории (пустая строка для корня)",
                        "default": ""
                    },
                    "branch": {
                        "type": "string",
                        "description": "Ветка (по умолчанию main)",
                        "default": "main"
                    }
                },
                "required": ["repo"]
            }
        ),
        Tool(
            name="get_recent_commits",
            description="Получить последние коммиты в репозитории",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "branch": {
                        "type": "string",
                        "description": "Ветка (по умолчанию main)",
                        "default": "main"
                    },
                    "max_commits": {
                        "type": "number",
                        "description": "Максимальное количество коммитов (по умолчанию 10)",
                        "default": 10
                    }
                },
                "required": ["repo"]
            }
        ),
        Tool(
            name="create_issue",
            description="Создать новый issue в репозитории",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "title": {
                        "type": "string",
                        "description": "Заголовок issue"
                    },
                    "body": {
                        "type": "string",
                        "description": "Описание issue"
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Список меток (опционально)"
                    }
                },
                "required": ["repo", "title", "body"]
            }
        ),
        Tool(
            name="get_pull_requests",
            description="Получить список pull requests",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "state": {
                        "type": "string",
                        "enum": ["open", "closed", "all"],
                        "description": "Состояние PR (по умолчанию open)",
                        "default": "open"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Максимальное количество результатов",
                        "default": 10
                    }
                },
                "required": ["repo"]
            }
        ),
        Tool(
            name="get_pull_request_diff",
            description="Получить diff указанного pull request",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "number": {
                        "type": "number",
                        "description": "Номер pull request"
                    }
                },
                "required": ["repo", "number"]
            }
        ),
        Tool(
            name="get_pull_request_files",
            description="Получить список файлов и краткую сводку изменений из pull request",
            inputSchema={
                "type": "object",
                "properties": {
                    "repo": {
                        "type": "string",
                        "description": "Полное имя репозитория (owner/repo)"
                    },
                    "number": {
                        "type": "number",
                        "description": "Номер pull request"
                    }
                },
                "required": ["repo", "number"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Обработка вызова инструмента"""
    try:
        gh = get_github_client()
        
        if name == "search_code":
            return await search_code(gh, arguments)
        elif name == "get_file_content":
            return await get_file_content(gh, arguments)
        elif name == "list_repository_files":
            return await list_repository_files(gh, arguments)
        elif name == "get_recent_commits":
            return await get_recent_commits(gh, arguments)
        elif name == "create_issue":
            return await create_issue(gh, arguments)
        elif name == "get_pull_requests":
            return await get_pull_requests(gh, arguments)
        elif name == "get_pull_request_diff":
            return await get_pull_request_diff(gh, arguments)
        elif name == "get_pull_request_files":
            return await get_pull_request_files(gh, arguments)
        else:
            raise ValueError(f"Неизвестный инструмент: {name}")
            
    except GithubException as e:
        return [TextContent(
            type="text",
            text=f"❌ Ошибка GitHub API: {e.data.get('message', str(e))}"
        )]
    except Exception as e:
        return [TextContent(
            type="text",
            text=f"❌ Ошибка: {str(e)}"
        )]

async def search_code(gh: Github, args: dict) -> list[TextContent]:
    """Поиск кода в репозитории"""
    repo_name = args["repo"]
    query = args["query"]
    language = args.get("language")
    max_results = args.get("max_results", 5)
    
    # Формируем поисковый запрос
    search_query = f"{query} repo:{repo_name}"
    if language:
        search_query += f" language:{language}"
    
    results = gh.search_code(search_query)
    
    result_text = f"🔍 Результаты поиска '{query}' в {repo_name}:\n\n"
    
    count = 0
    for item in results:
        if count >= max_results:
            break
        
        result_text += f"📄 **{item.path}**\n"
        result_text += f"   🔗 {item.html_url}\n"
        
        # Получаем содержимое файла
        try:
            content = item.decoded_content.decode('utf-8')
            # Показываем фрагмент
            lines = content.split('\n')
            preview = '\n'.join(lines[:10])
            if len(lines) > 10:
                preview += f"\n... (еще {len(lines) - 10} строк)"
            result_text += f"```\n{preview}\n```\n\n"
        except:
            result_text += "   (не удалось загрузить содержимое)\n\n"
        
        count += 1
    
    if count == 0:
        result_text += "Ничего не найдено.\n"
    else:
        result_text += f"\nНайдено результатов: {count}"
    
    return [TextContent(type="text", text=result_text)]

async def get_file_content(gh: Github, args: dict) -> list[TextContent]:
    """Получить содержимое файла"""
    repo_name = args["repo"]
    file_path = args["path"]
    branch = args.get("branch", "main")
    
    repo = gh.get_repo(repo_name)
    
    try:
        file_content = repo.get_contents(file_path, ref=branch)
        content = file_content.decoded_content.decode('utf-8')
        
        result_text = f"📄 **{file_path}** (ветка: {branch})\n"
        result_text += f"🔗 {file_content.html_url}\n"
        result_text += f"📏 Размер: {file_content.size} байт\n\n"
        result_text += f"```\n{content}\n```"
        
        return [TextContent(type="text", text=result_text)]
    except GithubException as e:
        if e.status == 404:
            return [TextContent(
                type="text",
                text=f"❌ Файл '{file_path}' не найден в ветке '{branch}'"
            )]
        raise

async def list_repository_files(gh: Github, args: dict) -> list[TextContent]:
    """Получить список файлов в директории"""
    repo_name = args["repo"]
    path = args.get("path", "")
    branch = args.get("branch", "main")
    
    repo = gh.get_repo(repo_name)
    contents = repo.get_contents(path, ref=branch)
    
    if not isinstance(contents, list):
        contents = [contents]
    
    result_text = f"📁 Содержимое '{path or 'корневой директории'}' в {repo_name} (ветка: {branch}):\n\n"
    
    dirs = []
    files = []
    
    for item in contents:
        if item.type == "dir":
            dirs.append(f"📁 {item.name}/")
        else:
            size_kb = item.size / 1024
            files.append(f"📄 {item.name} ({size_kb:.1f} KB)")
    
    if dirs:
        result_text += "**Директории:**\n" + "\n".join(dirs) + "\n\n"
    
    if files:
        result_text += "**Файлы:**\n" + "\n".join(files) + "\n"
    
    if not dirs and not files:
        result_text += "Директория пуста.\n"
    
    return [TextContent(type="text", text=result_text)]

async def get_recent_commits(gh: Github, args: dict) -> list[TextContent]:
    """Получить последние коммиты"""
    repo_name = args["repo"]
    branch = args.get("branch", "main")
    max_commits = args.get("max_commits", 10)
    
    repo = gh.get_repo(repo_name)
    commits = repo.get_commits(sha=branch)
    
    result_text = f"📝 Последние коммиты в {repo_name} (ветка: {branch}):\n\n"
    
    count = 0
    for commit in commits:
        if count >= max_commits:
            break
        
        commit_date = commit.commit.author.date
        author = commit.commit.author.name
        message = commit.commit.message.split('\n')[0]  # Первая строка
        sha = commit.sha[:7]
        
        result_text += f"🔸 **{sha}** - {message}\n"
        result_text += f"   👤 {author} | 📅 {commit_date.strftime('%Y-%m-%d %H:%M')}\n"
        result_text += f"   🔗 {commit.html_url}\n\n"
        
        count += 1
    
    return [TextContent(type="text", text=result_text)]

async def create_issue(gh: Github, args: dict) -> list[TextContent]:
    """Создать issue"""
    repo_name = args["repo"]
    title = args["title"]
    body = args["body"]
    labels = args.get("labels", [])
    
    repo = gh.get_repo(repo_name)
    issue = repo.create_issue(title=title, body=body, labels=labels)
    
    result_text = f"✅ Issue создан успешно!\n\n"
    result_text += f"📋 **{issue.title}** (#{issue.number})\n"
    result_text += f"🔗 {issue.html_url}\n"
    if labels:
        result_text += f"🏷️  Метки: {', '.join(labels)}\n"
    
    return [TextContent(type="text", text=result_text)]

async def get_pull_requests(gh: Github, args: dict) -> list[TextContent]:
    """Получить список PR"""
    repo_name = args["repo"]
    state = args.get("state", "open")
    max_results = args.get("max_results", 10)
    
    repo = gh.get_repo(repo_name)
    pulls = repo.get_pulls(state=state)
    
    result_text = f"🔀 Pull Requests в {repo_name} (состояние: {state}):\n\n"
    
    count = 0
    for pr in pulls:
        if count >= max_results:
            break
        
        result_text += f"🔸 **#{pr.number}**: {pr.title}\n"
        result_text += f"   👤 {pr.user.login} | 📅 {pr.created_at.strftime('%Y-%m-%d')}\n"
        result_text += f"   🔗 {pr.html_url}\n"
        result_text += f"   📊 +{pr.additions} -{pr.deletions}\n\n"
        
        count += 1
    
    if count == 0:
        result_text += f"Нет {state} pull requests.\n"
    
    return [TextContent(type="text", text=result_text)]

async def get_pull_request_diff(gh: Github, args: dict) -> list[TextContent]:
    """Получить diff для конкретного PR"""
    repo_name = args["repo"]
    number = int(args["number"])
    
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(number)
    files = pr.get_files()
    
    result_text = f"🔀 Diff для PR #{pr.number}: {pr.title}\n"
    result_text += f"🔗 {pr.html_url}\n\n"
    
    for file in files:
        result_text += f"📄 {file.filename} (+{file.additions} -{file.deletions})\n"
        if file.patch:
            patch_lines = file.patch.splitlines()
            max_lines = 200
            shown_lines = patch_lines[:max_lines]
            result_text += "```diff\n" + "\n".join(shown_lines) + "\n```"
            if len(patch_lines) > max_lines:
                result_text += f"\n... (diff обрезан, еще {len(patch_lines) - max_lines} строк)\n"
        else:
            result_text += "(diff недоступен)\n"
        result_text += "\n"
    
    return [TextContent(type="text", text=result_text)]

async def get_pull_request_files(gh: Github, args: dict) -> list[TextContent]:
    """Получить список файлов и краткую сводку изменений для PR"""
    repo_name = args["repo"]
    number = int(args["number"])
    
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(number)
    files = pr.get_files()
    
    result_text = f"📂 Файлы в PR #{pr.number}: {pr.title}\n"
    result_text += f"🔗 {pr.html_url}\n\n"
    
    for file in files:
        status = file.status  # modified, added, removed, renamed, etc.
        result_text += f"📄 {file.filename}\n"
        result_text += f"   Статус: {status}\n"
        result_text += f"   Изменения: +{file.additions} -{file.deletions}, всего {file.changes}\n\n"
    
    if not files:
        result_text += "Файлы не найдены.\n"
    
    return [TextContent(type="text", text=result_text)]

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