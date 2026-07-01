from __future__ import annotations

import os
from typing import Any, Mapping


def generate_human_explanation(report: Mapping[str, Any], *, audience: str = 'doctor') -> str:
    """Deterministic text generator for reproducible demo explanations."""
    risk = report.get('risk', report.get('input_risk', 'unknown'))
    memberships = report.get('memberships', {}) or {}
    selected = report.get('selected_class', report.get('selected_representation', 'not selected'))
    uncertainty = report.get('uncertainty', report.get('u', None))
    delta = report.get('reduction_loss', report.get('Delta', None))
    high = memberships.get('high', None)
    medium = memberships.get('medium', None)

    if audience == 'patient':
        intro = 'Система объясняет результат простыми словами.'
        tail = 'Такое объяснение не заменяет решение врача, но показывает, какие элементы вызвали повышенное внимание.'
    elif audience == 'engineer':
        intro = 'Техническое объяснение фиксирует состояние объяснительного объекта и выбранного представления неопределённости.'
        tail = 'Все численные величины сохраняются в трассируемом отчёте и могут быть пересчитаны.'
    else:
        intro = 'Клиническое объяснение показывает риск, степень уверенности и ограничения интерпретации.'
        tail = 'При повышенной неопределённости требуется дополнительная проверка или аудит источников.'

    parts = [intro]
    parts.append(f'Нормированный риск равен {risk}.')
    if high is not None or medium is not None:
        parts.append(f'Степень принадлежности терму "высокий риск" равна {high}, терму "средний риск" — {medium}.')
    parts.append(f'Для представления неопределённости выбран класс {selected}.')
    if uncertainty is not None:
        parts.append(f'Агрегированная неопределённость объяснения равна {uncertainty}.')
    if delta is not None:
        parts.append(f'Потеря при редукции сложного представления составляет {delta}.')
    parts.append(tail)
    return ' '.join(str(p) for p in parts)


def build_llm_prompt(report: Mapping[str, Any], *, audience: str = 'doctor') -> str:
    """Build a reproducible prompt for an optional LLM explanation."""
    return (
        'Сформулируй краткое объяснение результата ИИ-системы на русском языке. '
        'Не добавляй медицинских рекомендаций, которых нет в данных. '
        'Сохрани оговорку, что объяснение не заменяет экспертное решение.\n\n'
        f'Аудитория: {audience}.\n'
        f'Данные объяснения: {dict(report)}\n'
    )


def generate_explanation_with_optional_llm(
    report: Mapping[str, Any],
    *,
    audience: str = 'doctor',
    use_llm: bool = False,
    model: str = 'gpt-4o-mini',
) -> tuple[str, Mapping[str, Any]]:
    """Generate text either deterministically or through OpenAI if explicitly enabled.

    The function never requires OpenAI. If ``use_llm`` is False, if the package is
    not installed, or if ``OPENAI_API_KEY`` is absent, a deterministic template is
    returned. In all cases, the prompt metadata is returned for the trace.
    """
    prompt = build_llm_prompt(report, audience=audience)
    trace = {'mode': 'template', 'audience': audience, 'prompt': prompt, 'model': None}
    if not use_llm or not os.getenv('OPENAI_API_KEY'):
        return generate_human_explanation(report, audience=audience), trace

    try:  # pragma: no cover - optional external API path
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {'role': 'system', 'content': 'Ты пишешь краткие, аккуратные объяснения ИИ-решений для научной демонстрации.'},
                {'role': 'user', 'content': prompt},
            ],
        )
        text = response.choices[0].message.content or ''
        trace.update({'mode': 'openai', 'model': model})
        return text.strip(), trace
    except Exception as exc:
        trace.update({'mode': 'template_fallback', 'error': str(exc)[:300]})
        return generate_human_explanation(report, audience=audience), trace
