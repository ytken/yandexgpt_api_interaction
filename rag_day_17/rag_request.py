import os
import sys
from typing import Dict
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rag_day_16'))
from rag_classes import YandexRAGSystem
from text_to_embedding import YandexDocumentIndexer

load_dotenv()


class RAGComparison:
    """Класс для сравнения ответов с RAG и без RAG"""
    
    def __init__(self, folder_id: str = None, api_key: str = None):
        self.folder_id = folder_id or os.getenv("YANDEX_FOLDER_ID")
        self.api_key = api_key or os.getenv("YANDEX_API_KEY")
        
        print("🚀 Инициализация системы сравнения...")
        self.sdk = YCloudML(folder_id=self.folder_id, auth=self.api_key)
        print("✅ SDK инициализирован!\n")
    
    def ask_without_rag(self, query: str, model: str = "yandexgpt", 
                       temperature: float = 0.3) -> Dict:
        """
        Запрос к LLM БЕЗ RAG (только на основе знаний модели)
        
        Args:
            query: Вопрос пользователя
            model: Модель YandexGPT
            temperature: Температура генерации
            
        Returns:
            Словарь с ответом и метаданными
        """
        print("🤖 Запрос БЕЗ RAG (только знания модели)...")
        
        # Простой промпт без контекста
        prompt = f"""Ответь на следующий вопрос на основе своих знаний:

ВОПРОС:
{query}

ОТВЕТ:"""
        
        llm_model = self.sdk.models.completions(model)
        llm_model = llm_model.configure(temperature=temperature, max_tokens=2000)
        
        result = llm_model.run(prompt)
        answer_text = result.alternatives[0].text
        
        usage_stats = {
            'prompt_tokens': result.usage.input_text_tokens if hasattr(result, 'usage') else 0,
            'completion_tokens': result.usage.completion_tokens if hasattr(result, 'usage') else 0,
            'total_tokens': result.usage.total_tokens if hasattr(result, 'usage') else 0
        }
        
        print(f"✅ Ответ получен! Токенов: {usage_stats['total_tokens']}\n")
        
        return {
            'answer': answer_text,
            'usage': usage_stats,
            'model': model,
            'method': 'WITHOUT_RAG'
        }
    
    def compare_responses(self, question: str, rag_result: Dict, 
                         no_rag_result: Dict) -> Dict:
        """
        Сравнивает ответы с RAG и без RAG
        
        Args:
            question: Заданный вопрос
            rag_result: Результат с RAG
            no_rag_result: Результат без RAG
            
        Returns:
            Словарь с анализом
        """
        comparison = {
            'question': question,
            'with_rag': {
                'answer': rag_result['answer'],
                'tokens': rag_result['usage']['total_tokens'],
                'context_chunks': len(rag_result.get('context_chunks', [])),
                'relevance_scores': [
                    chunk['similarity'] 
                    for chunk in rag_result.get('context_chunks', [])
                ]
            },
            'without_rag': {
                'answer': no_rag_result['answer'],
                'tokens': no_rag_result['usage']['total_tokens']
            }
        }
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Красиво выводит сравнение"""
        print("\n" + "="*80)
        print(f"❓ ВОПРОС: {comparison['question']}")
        print("="*80)
        
        print("\n" + "🔵 ОТВЕТ БЕЗ RAG (только знания модели)")
        print("-"*80)
        print(comparison['without_rag']['answer'])
        print(f"\n📊 Токенов использовано: {comparison['without_rag']['tokens']}")
        
        print("\n" + "🟢 ОТВЕТ С RAG (на основе документа)")
        print("-"*80)
        print(comparison['with_rag']['answer'])
        print(f"\n📊 Токенов использовано: {comparison['with_rag']['tokens']}")
        print(f"📚 Использовано фрагментов: {comparison['with_rag']['context_chunks']}")
        if comparison['with_rag']['relevance_scores']:
            print(f"🎯 Релевантность фрагментов: {[f'{s:.3f}' for s in comparison['with_rag']['relevance_scores']]}")
        
        print("\n" + "="*80 + "\n")
    
    def analyze_difference(self, comparison: Dict) -> str:
        """
        Анализирует разницу между ответами
        
        Returns:
            Текстовый анализ
        """
        rag_answer = comparison['with_rag']['answer'].lower()
        no_rag_answer = comparison['without_rag']['answer'].lower()
        
        analysis = []
        
        # Проверяем наличие конкретных деталей
        if len(rag_answer) > len(no_rag_answer) * 1.3:
            analysis.append("✅ RAG дал более детальный ответ")
        elif len(no_rag_answer) > len(rag_answer) * 1.3:
            analysis.append("⚠️ Ответ без RAG оказался более развернутым")
        
        # Проверяем упоминание конкретных фактов
        concrete_indicators = ['согласно', 'в документе', 'указано', 'фрагмент', 'говорится']
        if any(indicator in rag_answer for indicator in concrete_indicators):
            analysis.append("✅ RAG ссылается на конкретные источники")
        
        # Проверяем общие фразы (признак отсутствия информации)
        vague_indicators = ['в общем', 'как правило', 'обычно', 'может быть', 'возможно']
        no_rag_vague = sum(1 for indicator in vague_indicators if indicator in no_rag_answer)
        rag_vague = sum(1 for indicator in vague_indicators if indicator in rag_answer)
        
        if no_rag_vague > rag_vague:
            analysis.append("✅ RAG дал более конкретный ответ")
        
        # Проверяем релевантность найденных чанков
        if comparison['with_rag']['relevance_scores']:
            avg_relevance = sum(comparison['with_rag']['relevance_scores']) / len(comparison['with_rag']['relevance_scores'])
            if avg_relevance > 0.7:
                analysis.append(f"✅ Высокая релевантность контекста ({avg_relevance:.3f})")
            elif avg_relevance > 0.5:
                analysis.append(f"⚠️ Средняя релевантность контекста ({avg_relevance:.3f})")
            else:
                analysis.append(f"❌ Низкая релевантность контекста ({avg_relevance:.3f})")
        
        return "\n".join(analysis) if analysis else "Существенных различий не обнаружено"

def get_test_questions():
    """Возвращает набор тестовых вопросов разных типов"""
    
    questions = {
        'specific_facts': [
            # Вопросы о конкретных фактах из документа
            "Какова выручка TechnoMax Solutions за 2024 год?",
            "Назовите точность системы VisionControl.",
            "Сколько патентов получила компания в 2024 году?",
        ],
        'complex_analysis': [
            # Вопросы, требующие анализа нескольких фактов
            "Сравните финансовые результаты 2024 года с предыдущим периодом. В чем основные успехи?",
            "Проанализируйте портфель продуктов: для каких отраслей и задач они предназначены?",
            "Каковы ключевые направления инвестиций в развитие компании (R&D, команда, география)?",
        ],
        'general_knowledge': [
            # Вопросы на общие знания (модель может ответить и без RAG)
            "Что такое машинное обучение?",
            "Для чего используется компьютерное зрение в промышленности?",
            "Что такое EBITDA?",
        ],
        'outside_context': [
            # Вопросы, информации для которых нет в документе
            "Какие рыночные тренды в промышленном ИИ могут повлиять на TechnoMax?",
            "С какими основными рисками может столкнуться компания при выходе на новые рынки?",
            "Какие еще компании в России разрабатывают аналогичные решения?",
        ]
    }
    
    return questions


def main():
    """Основная функция для проведения тестирования"""
    
    print("="*80)
    print("🧪 СРАВНЕНИЕ RAG vs БЕЗ RAG")
    print("="*80 + "\n")
    
    # Создаем тестовый документ
    print("📄 Подготовка тестового документа...")
    doc_filename = "technomax_report.txt"
    
    # Импортируем необходимые классы
    print("📦 Импорт модулей для индексации...")
    try:
        from __main__ import YandexDocumentIndexer, YandexRAGSystem
    except:
        print("⚠️ Импортируйте классы YandexDocumentIndexer и YandexRAGSystem")
        print("   Этот скрипт должен работать совместно с основными модулями.\n")
        
        # Предоставляем инструкции
        print("📋 Инструкции по запуску:")
        print("1. Сначала создайте индекс:")
        print(f"   indexer = YandexDocumentIndexer()")
        print(f"   indexer.process_file('{doc_filename}')")
        print()
        print("2. Затем запустите сравнение:")
        print(f"   rag = YandexRAGSystem('{doc_filename.replace('.txt', '_index.json')}')")
        print(f"   comparison = RAGComparison()")
        return
    
    try:
        # Создаем индекс документа
        print("🔧 Создание индекса документа...")
        indexer = YandexDocumentIndexer()
        index_path = indexer.process_file(doc_filename, chunk_size=400, overlap=50)
        
        # Инициализируем системы
        print("\n🚀 Инициализация систем...")
        rag_system = YandexRAGSystem(index_path)
        comparison_system = RAGComparison()
        
        # Получаем тестовые вопросы
        test_questions = get_test_questions()
        
        # Результаты по категориям
        results_by_category = {}
        
        # Тестируем каждую категорию вопросов
        for category, questions in test_questions.items():
            print(f"\n{'='*80}")
            print(f"📋 КАТЕГОРИЯ: {category.upper().replace('_', ' ')}")
            print(f"{'='*80}\n")
            
            category_results = []
            
            for question in questions:
                print(f"\n{'─'*80}")
                
                # Получаем ответ БЕЗ RAG
                no_rag_result = comparison_system.ask_without_rag(question)
                
                # Получаем ответ С RAG
                rag_result = rag_system.ask(question, top_k=3)
                
                # Сравниваем результаты
                comparison = comparison_system.compare_responses(
                    question, rag_result, no_rag_result
                )
                
                # Выводим сравнение
                comparison_system.print_comparison(comparison)
                
                # Анализируем разницу
                analysis = comparison_system.analyze_difference(comparison)
                print("🔍 АНАЛИЗ РАЗНИЦЫ:")
                print("-"*80)
                print(analysis)
                print("\n" + "─"*80)
                
                category_results.append({
                    'question': question,
                    'comparison': comparison,
                    'analysis': analysis
                })
            
            results_by_category[category] = category_results
        
        # Итоговый отчет
        print("\n" + "="*80)
        print("📊 ИТОГОВЫЙ ОТЧЕТ")
        print("="*80 + "\n")
        
        for category, results in results_by_category.items():
            print(f"\n🔹 {category.upper().replace('_', ' ')}")
            print(f"   Вопросов протестировано: {len(results)}")
            
            # Подсчитываем, где RAG помог
            rag_helped = sum(1 for r in results if '✅' in r['analysis'])
            print(f"   RAG был полезен: {rag_helped}/{len(results)}")
            
            # Средняя релевантность
            relevances = [
                sum(r['comparison']['with_rag']['relevance_scores']) / len(r['comparison']['with_rag']['relevance_scores'])
                for r in results 
                if r['comparison']['with_rag']['relevance_scores']
            ]
            if relevances:
                avg_rel = sum(relevances) / len(relevances)
                print(f"   Средняя релевантность контекста: {avg_rel:.3f}")
        
        print("\n" + "="*80)
        print("🎯 ВЫВОДЫ")
        print("="*80)
        print("""
✅ RAG НАИБОЛЕЕ ПОЛЕЗЕН ДЛЯ:
   - Вопросов о конкретных фактах и цифрах из документа
   - Запросов, требующих точной информации
   - Случаев, когда модель не знает специфической информации

⚠️ RAG МЕНЕЕ ПОЛЕЗЕН ДЛЯ:
   - Вопросов на общие знания
   - Задач, требующих внешних знаний (не из документа)
   - Запросов, требующих рассуждений без опоры на факты

🔑 КЛЮЧЕВОЙ ФАКТОР: Релевантность найденных фрагментов
   - При релевантности > 0.7: RAG значительно улучшает ответ
   - При релевантности < 0.5: RAG может добавить шума
        """)
        
        print("✅ Тестирование завершено!\n")
        
    except Exception as e:
        print(f"\n❌ Ошибка: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()