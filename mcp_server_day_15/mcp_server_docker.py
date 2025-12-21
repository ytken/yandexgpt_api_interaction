import asyncio
import json
from typing import Any
from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
import mcp.server.stdio
import docker
from docker.errors import DockerException, NotFound, APIError

# Инициализация Docker клиента
try:
    docker_client = docker.from_env()
    docker_client.ping()
    DOCKER_AVAILABLE = True
except Exception as e:
    DOCKER_AVAILABLE = False
    print(f"⚠️  Docker недоступен: {e}", flush=True)

# Создание MCP сервера
server = Server("docker-manager")

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """Список доступных инструментов для работы с Docker"""
    return [
        types.Tool(
            name="start_container",
            description="Запускает Docker контейнер по имени образа. Поддерживает настройку портов, переменных окружения и volume.",
            inputSchema={
                "type": "object",
                "properties": {
                    "image": {
                        "type": "string",
                        "description": "Имя Docker образа (например: nginx:latest, postgres:15)"
                    },
                    "name": {
                        "type": "string",
                        "description": "Имя контейнера (опционально)"
                    },
                    "ports": {
                        "type": "object",
                        "description": "Маппинг портов в формате {'container_port/protocol': host_port}, например: {'80/tcp': 8080}",
                        "additionalProperties": True
                    },
                    "environment": {
                        "type": "object",
                        "description": "Переменные окружения в формате {key: value}",
                        "additionalProperties": {"type": "string"}
                    },
                    "volumes": {
                        "type": "object",
                        "description": "Монтирование volume в формате {host_path: {'bind': container_path, 'mode': 'rw'}}",
                        "additionalProperties": True
                    },
                    "detach": {
                        "type": "boolean",
                        "description": "Запустить в фоновом режиме (по умолчанию: true)"
                    },
                    "remove": {
                        "type": "boolean",
                        "description": "Автоматически удалить контейнер после остановки (по умолчанию: false)"
                    }
                },
                "required": ["image"]
            }
        ),
        types.Tool(
            name="stop_container",
            description="Останавливает работающий Docker контейнер",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "ID или имя контейнера"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Таймаут в секундах для graceful shutdown (по умолчанию: 10)"
                    }
                },
                "required": ["container_id"]
            }
        ),
        types.Tool(
            name="list_containers",
            description="Показывает список Docker контейнеров",
            inputSchema={
                "type": "object",
                "properties": {
                    "all": {
                        "type": "boolean",
                        "description": "Показать все контейнеры, включая остановленные (по умолчанию: false)"
                    }
                }
            }
        ),
        types.Tool(
            name="remove_container",
            description="Удаляет Docker контейнер",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "ID или имя контейнера"
                    },
                    "force": {
                        "type": "boolean",
                        "description": "Принудительно удалить, даже если контейнер работает (по умолчанию: false)"
                    }
                },
                "required": ["container_id"]
            }
        ),
        types.Tool(
            name="container_logs",
            description="Получает логи контейнера",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "ID или имя контейнера"
                    },
                    "tail": {
                        "type": "integer",
                        "description": "Количество последних строк (по умолчанию: 100)"
                    }
                },
                "required": ["container_id"]
            }
        ),
        types.Tool(
            name="container_stats",
            description="Получает статистику использования ресурсов контейнера",
            inputSchema={
                "type": "object",
                "properties": {
                    "container_id": {
                        "type": "string",
                        "description": "ID или имя контейнера"
                    }
                },
                "required": ["container_id"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Обработка вызовов инструментов"""
    
    if not DOCKER_AVAILABLE:
        return [types.TextContent(
            type="text",
            text="❌ Docker недоступен. Убедитесь, что Docker запущен и доступен."
        )]
    
    try:
        if name == "start_container":
            return await start_container(arguments or {})
        elif name == "stop_container":
            return await stop_container(arguments or {})
        elif name == "list_containers":
            return await list_containers(arguments or {})
        elif name == "remove_container":
            return await remove_container(arguments or {})
        elif name == "container_logs":
            return await get_container_logs(arguments or {})
        elif name == "container_stats":
            return await get_container_stats(arguments or {})
        else:
            return [types.TextContent(
                type="text",
                text=f"❌ Неизвестный инструмент: {name}"
            )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при выполнении {name}: {str(e)}"
        )]

async def start_container(args: dict) -> list[types.TextContent]:
    """Запуск Docker контейнера"""
    try:
        image = args.get("image")
        name = args.get("name")
        ports = args.get("ports", {})
        environment = args.get("environment", {})
        volumes = args.get("volumes", {})
        detach = args.get("detach", True)
        remove = args.get("remove", False)
        
        # Проверяем наличие образа, если нет - скачиваем
        try:
            docker_client.images.get(image)
        except NotFound:
            result_text = f"📥 Образ {image} не найден локально, начинаю загрузку...\n"
            docker_client.images.pull(image)
            result_text += f"✅ Образ {image} успешно загружен\n\n"
        else:
            result_text = ""
        
        # Запускаем контейнер
        container = docker_client.containers.run(
            image=image,
            name=name,
            ports=ports,
            environment=environment,
            volumes=volumes,
            detach=detach,
            remove=remove
        )
        
        result_text += f"✅ Контейнер успешно запущен!\n"
        result_text += f"   ID: {container.short_id}\n"
        result_text += f"   Имя: {container.name}\n"
        result_text += f"   Образ: {image}\n"
        
        if ports:
            result_text += f"   Порты: {json.dumps(ports, ensure_ascii=False)}\n"
        
        return [types.TextContent(type="text", text=result_text)]
        
    except APIError as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка Docker API: {e.explanation}"
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при запуске контейнера: {str(e)}"
        )]

async def stop_container(args: dict) -> list[types.TextContent]:
    """Остановка контейнера"""
    try:
        container_id = args.get("container_id")
        timeout = args.get("timeout", 10)
        
        container = docker_client.containers.get(container_id)
        container.stop(timeout=timeout)
        
        return [types.TextContent(
            type="text",
            text=f"✅ Контейнер {container.name} ({container.short_id}) успешно остановлен"
        )]
        
    except NotFound:
        return [types.TextContent(
            type="text",
            text=f"❌ Контейнер {args.get('container_id')} не найден"
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при остановке контейнера: {str(e)}"
        )]

async def list_containers(args: dict) -> list[types.TextContent]:
    """Список контейнеров"""
    try:
        show_all = args.get("all", False)
        containers = docker_client.containers.list(all=show_all)
        
        if not containers:
            return [types.TextContent(
                type="text",
                text="📦 Контейнеры не найдены"
            )]
        
        result = f"📦 Найдено контейнеров: {len(containers)}\n\n"
        
        for container in containers:
            result += f"{'🟢' if container.status == 'running' else '🔴'} {container.name}\n"
            result += f"   ID: {container.short_id}\n"
            result += f"   Образ: {container.image.tags[0] if container.image.tags else container.image.short_id}\n"
            result += f"   Статус: {container.status}\n"
            
            # Порты
            if container.ports:
                result += f"   Порты: {json.dumps(container.ports, ensure_ascii=False)}\n"
            
            result += "\n"
        
        return [types.TextContent(type="text", text=result)]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при получении списка контейнеров: {str(e)}"
        )]

async def remove_container(args: dict) -> list[types.TextContent]:
    """Удаление контейнера"""
    try:
        container_id = args.get("container_id")
        force = args.get("force", False)
        
        container = docker_client.containers.get(container_id)
        container_name = container.name
        container.remove(force=force)
        
        return [types.TextContent(
            type="text",
            text=f"✅ Контейнер {container_name} успешно удален"
        )]
        
    except NotFound:
        return [types.TextContent(
            type="text",
            text=f"❌ Контейнер {args.get('container_id')} не найден"
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при удалении контейнера: {str(e)}"
        )]

async def get_container_logs(args: dict) -> list[types.TextContent]:
    """Получение логов контейнера"""
    try:
        container_id = args.get("container_id")
        tail = args.get("tail", 100)
        
        container = docker_client.containers.get(container_id)
        logs = container.logs(tail=tail).decode('utf-8')
        
        result = f"📋 Логи контейнера {container.name} (последние {tail} строк):\n\n"
        result += logs
        
        return [types.TextContent(type="text", text=result)]
        
    except NotFound:
        return [types.TextContent(
            type="text",
            text=f"❌ Контейнер {args.get('container_id')} не найден"
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при получении логов: {str(e)}"
        )]

async def get_container_stats(args: dict) -> list[types.TextContent]:
    """Получение статистики контейнера"""
    try:
        container_id = args.get("container_id")
        
        container = docker_client.containers.get(container_id)
        stats = container.stats(stream=False)
        
        # Вычисляем процент использования CPU
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                    stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        cpu_percent = (cpu_delta / system_delta) * 100.0 if system_delta > 0 else 0.0
        
        # Использование памяти
        memory_usage = stats['memory_stats']['usage'] / (1024 * 1024)  # MB
        memory_limit = stats['memory_stats']['limit'] / (1024 * 1024)  # MB
        memory_percent = (memory_usage / memory_limit) * 100 if memory_limit > 0 else 0
        
        result = f"📊 Статистика контейнера {container.name}:\n\n"
        result += f"CPU: {cpu_percent:.2f}%\n"
        result += f"Память: {memory_usage:.2f} MB / {memory_limit:.2f} MB ({memory_percent:.2f}%)\n"
        result += f"Статус: {container.status}\n"
        
        return [types.TextContent(type="text", text=result)]
        
    except NotFound:
        return [types.TextContent(
            type="text",
            text=f"❌ Контейнер {args.get('container_id')} не найден"
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=f"❌ Ошибка при получении статистики: {str(e)}"
        )]

async def main():
    """Запуск MCP сервера"""
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="docker-manager",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())