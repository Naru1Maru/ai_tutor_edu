import re
from typing import Dict, List, Optional

VERDICT_RE = re.compile(r"^вердикт\s*:\s*(верно|неверно)\b", re.IGNORECASE | re.MULTILINE)

SYSTEM_PROMPT = (
    "Ты — эксперт ЕГЭ по математике. Твоя задача — проверить решение ученика.\n"
    "Отвечай СТРОГО в формате:\n"
    "Вердикт: верно|неверно\n"
    "Пояснение: <1-3 предложения, конкретно где ошибка или почему всё верно>\n"
    "Не добавляй ничего кроме этих двух строк."
)

USER_TEMPLATE = (
    "Условие задачи:\n"
    "{condition}\n\n"
    "Решение ученика:\n"
    "{student_solution}\n"
)

USER_TEMPLATE_WITH_REF = (
    "Условие задачи:\n"
    "{condition}\n\n"
    "Решение ученика:\n"
    "{student_solution}\n\n"
    "Эталонное решение (для сверки):\n"
    "{reference_solution}\n"
)

def build_messages(payload: Dict) -> List[Dict[str, str]]:
    condition = (payload.get("condition") or "").strip()
    student_solution = (payload.get("student_solution") or "").strip()
    reference_solution = payload.get("reference_solution")

    if reference_solution is not None:
        reference_solution = str(reference_solution).strip()
        user_text = USER_TEMPLATE_WITH_REF.format(
            condition=condition,
            student_solution=student_solution,
            reference_solution=reference_solution,
        )
    else:
        user_text = USER_TEMPLATE.format(condition=condition, student_solution=student_solution)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]


def extract_verdict(text: str) -> Optional[int]:
    m = VERDICT_RE.search(text or "")
    if not m:
        return None
    return 1 if m.group(1).lower() == "верно" else 0
