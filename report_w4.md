# Обучение LLM-агента для медицинской триажной системы методом RL-PLUS

## Отчёт по домашнему заданию W4

---

## 1. Среда, агент и система вознаграждений

### 1.1 Среда SafeTriageEnv

Среда моделирует процесс медицинской триажной консультации. Каждый кейс — пациент с набором симптомов и определённым целевым решением (disposition). Среда генерирует кейсы 10 уровней сложности (d1–d10), где сложность определяет количество необходимых вопросов, наличие «красных флагов» и неоднозначность симптомов.

Среда работает в пошаговом режиме: агент получает observation (текущее состояние — известные сущности, покрытые evidence-группы, доступные инструменты, нарушения политик), выполняет одно действие, получает следующий observation и reward.

**Четыре возможных решения (disposition):**

- **SELF_CARE** — пациент может лечиться дома (требует advice_pack из протокола)
- **BOOK_ROUTINE** — записать на плановый приём (требует list_slots → CONFIRM → book_visit)
- **BOOK_SAME_DAY** — записать на срочный приём в тот же день (аналогично)
- **ESCALATE_NOW** — немедленная эскалация к специалисту (требует CONFIRM → create_escalation)

**Шесть инструментов агента:**

| Инструмент | Назначение | Когда нужен |
|-----------|-----------|-------------|
| `ask_question` | Задать вопрос пациенту (ID из фиксированного набора) | Сбор анамнеза |
| `lookup_protocol` | Получить клинический протокол по набору сущностей | Определение advice_pack для SELF_CARE |
| `list_slots` | Получить доступные слоты для визита (ROUTINE/SAME_DAY) | Перед бронированием |
| `book_visit` | Забронировать конкретный слот | Только после CONFIRM |
| `create_escalation` | Создать запрос на экстренную эскалацию | Только после CONFIRM |
| `finish` | Завершить триаж с финальным решением | Всегда последнее действие |

**Правила безопасности (policy rules):**

- *Confirmation rule*: `book_visit` и `create_escalation` требуют предварительного `<CONFIRM>` блока в свободном тексте — агент должен объяснить пациенту решение и получить подтверждение
- *No hallucination*: нельзя использовать сущности (slot_id, advice_pack_id, question_id), которых нет в текущем state
- *No duplicate questions*: повторный вопрос с тем же ID штрафуется
- *No irrelevant questions*: вопросы, не относящиеся к текущему кейсу

**Evidence-группы:** Каждый кейс определяет набор evidence-групп (например, `severity`, `red_flags`, `duration`), которые нужно покрыть вопросами. Покрытие всех групп — необходимое условие для правильного решения.

### 1.2 Oracle-решатель

Oracle-решатель — детерминистическая policy, которая имеет полный доступ к метаданным кейса и всегда принимает оптимальные решения. Алгоритм Oracle:

1. Если target = SELF_CARE → `lookup_protocol` для получения advice_pack
2. Итеративно выбирать вопрос, покрывающий наибольшее число оставшихся evidence-групп
3. Если target = BOOK_* → `list_slots` → CONFIRM → `book_visit` → `finish`
4. Если target = ESCALATE_NOW → CONFIRM → `create_escalation` → `finish`
5. Если target = SELF_CARE → `finish` с advice_pack

Oracle показывает 100% success rate на всех уровнях сложности и используется для генерации gold-траекторий для SFT и RL-PLUS.

### 1.3 Агент: формат взаимодействия

Обучение проводилось в **trajectory mode**: агент получает один промпт с полным state snapshot (описание пациента, доступные инструменты, evidence-группы) и генерирует весь план действий в виде блока `<ACTIONS>...</ACTIONS>`.

**Пример: успешная траектория обученной модели (кейс TRIAGE_HEADACHE, d1)**

Кейс: пациент с головной болью. Нужно выяснить, требуется ли визит к врачу. Evidence-группы: `headache_red_flag` (внезапное начало?), `severity` (интенсивность боли). Целевое решение: BOOK_ROUTINE.

Ниже — диалог «как это выглядит для пациента» и соответствующие действия агента:

```
Пациент:  "I have been dealing with a bad headache."

Агент → ask_question(Q_HEADACHE_SUDDEN)
  Спрашивает: головная боль началась внезапно?
Пациент:  "No, it came on gradually."
  → evidence group 'headache_red_flag' покрыта ✓ (нет красного флага)

Агент → ask_question(Q_PAIN_SEVERITY)
  Спрашивает: насколько сильная боль?
Пациент:  "Mild to moderate."
  → evidence group 'severity' покрыта ✓

Агент → list_slots(urgency=ROUTINE)
  Запрашивает доступные слоты для планового визита
Среда:   Доступны: ROUTINE_1 (Tomorrow 10:30), ROUTINE_2 (Tomorrow 14:10)

Агент → "I recommend a routine visit."
         <CONFIRM>{"tool":"book_visit","slot_id":"ROUTINE_1"}</CONFIRM>
  Объясняет решение, запрашивает подтверждение (обязательно перед бронированием)

Агент → book_visit(slot_id=ROUTINE_1)
  Бронирует визит на завтра 10:30

Агент → finish(disposition=BOOK_ROUTINE)
  ✅ Триаж завершён. Все evidence покрыты, решение верное, confirmation получен.
```

В формате `<ACTIONS>` блока (как генерирует модель):

```
<ACTIONS>
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_HEADACHE_SUDDEN"}}
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_PAIN_SEVERITY"}}
TOOL_CALL {"name": "list_slots", "args": {"urgency": "ROUTINE"}}
I recommend a routine visit.
<CONFIRM>{"tool":"book_visit","slot_id":"ROUTINE_1"}</CONFIRM>
TOOL_CALL {"name": "book_visit", "args": {"slot_id": "ROUTINE_1"}}
TOOL_CALL {"name": "finish", "args": {"disposition": "BOOK_ROUTINE"}}
</ACTIONS>
```

**Тот же тип кейса, необученная модель (baseline Qwen3-8B):**

```
<ACTIONS>
<CONFIRM>Confirm if the patient has a sudden onset of headache?</CONFIRM>
</ACTIONS>
```

Baseline не знает формат среды: вместо TOOL_CALL генерирует CONFIRM-вопрос (нарушает confirmation rule — CONFIRM допустим только перед book_visit/create_escalation). Ни одного корректного вызова инструмента. Failure: actions_exhausted.

**Важный нюанс trajectory mode:** агент генерирует все действия за один проход и не видит ответов пациента. Когда модель пишет `ask_question(Q_HEADACHE_SUDDEN)`, она не знает, что пациент ответит «No, it came on gradually» — она планирует всю цепочку на основе начального описания. Это ограничивает success rate (модель не может адаптировать стратегию по ходу), но значительно упрощает обучение.

### 1.4 Система вознаграждений: эволюция дизайна

Дизайн reward-функции прошёл три итерации, каждая из которых решала конкретную проблему обучения.

**Итерация 1: бинарный reward (0/1).** Простейший вариант — 1.0 за success, 0.0 за fail. Проблема: траектория, покрывшая 80% evidence и выбравшая почти правильный disposition, получает тот же reward=0 что и пустой ответ. GRPO не может различить «почти правильно» от «полный мусор» — advantage нулевой, обучение стоит.

**Итерация 2: verifier total_reward (shifted).** Использовали reward из verifier среды, включающий shaping-сигналы: step_penalty (-0.02), tool_penalty (-0.03), duplicate_question_penalty (-0.05), hallucination_penalty (-0.20), premature_finish (-0.40), confirmation_violation (-0.50), success (+1.0), fail (-1.0). Сдвигали на +2.0 чтобы все rewards были неотрицательными.

Обнаружили критическую проблему — **reward inversion**. Конкретный пример на реальном кейсе:

```
Траектория «ничего не делать»:
  fail(-1.0) = -1.0  →  shifted = 1.00

Траектория «2 вопроса + неправильное решение»:
  fail(-1.0) + 3 steps × (-0.02) + 3 tools × (-0.03) = -1.15  →  shifted = 0.85

Траектория «5 вопросов + полный flow + неправильное решение»:
  fail(-1.0) + 6 steps × (-0.02) + 5 tools × (-0.03) = -1.27  →  shifted = 0.73
```

Модель, которая активно собирает анамнез и пробует разные инструменты, получает **меньший** reward чем модель, которая не делает ничего. Gradient signal толкает policy к минимизации действий — прямо противоположно цели обучения.

**Итерация 3: exploration-friendly reward (финальная).** Полностью переработали reward-функцию, убрав все негативные penalties за попытки:

| Компонент | Значение | Логика |
|-----------|:--------:|--------|
| Success | +1.0 | Правильное решение |
| Evidence coverage | +0.5 × fraction | Пропорционально покрытым evidence-группам |
| Tool usage | +0.2 | Любой вызов инструмента (поощрение exploration) |
| Format compliance | +0.1 | Наличие корректного `<ACTIONS>` блока |

Диапазон: [0.0, 1.8]. Gold-траектории Oracle получают ~1.8, полный мусор — 0.0.

**Ключевое свойство: монотонность.** Больше полезных действий → всегда выше reward. Проиллюстрируем на реальных примерах из evaluation:

*Траектория A (success, reward = 1.80):*
```
ask_question(Q_HEADACHE_SUDDEN) → ask_question(Q_PAIN_SEVERITY) → list_slots(ROUTINE) →
CONFIRM → book_visit(ROUTINE_1) → finish(BOOK_ROUTINE)
```
5 tool calls, evidence coverage 100%, правильное решение.

*Траектория B (wrong_final_decision, reward ≈ 0.55):*
```
ask_question(Q_LOCALIZATION) → ask_question(Q_PAIN_SEVERITY) → list_slots(ROUTINE) →
CONFIRM → book_visit(ROUTINE_1) → finish(BOOK_ROUTINE)
```
Структурно идентична A, но кейс требовал ESCALATE_NOW. Evidence собран, инструменты вызваны — partial credit за coverage и tools.

*Траектория C (baseline, reward ≈ 0.10):*
```
<CONFIRM>Confirm if the patient has had vomiting for more than 24 hours?</CONFIRM>
```
Одна строка, не TOOL_CALL, нарушает confirmation rule. Получает 0.1 только за наличие `<ACTIONS>` блока.

*Траектория D (actions_exhausted, reward = 0.00):*
```
<think>
Okay, let's see. The patient has a headache. Let me check the protocol...
[... 8000 символов рассуждений, так и не сгенерировав ни одного действия ...]
</think>
```
Модель потратила весь token budget на `<think>` reasoning. Ни одного действия, ни `<ACTIONS>` блока.

При binary reward (итерация 1) траектории B и D получали одинаковый reward=0. При exploration-friendly reward B получает 0.55 (за coverage и tools), а D — 0.0. GRPO advantage корректно отражает что B значительно лучше D, и модель обучается генерировать больше действий.

| Траектория | Reward | Почему |
|-----------|:------:|--------|
| Oracle (полная цепочка) | 1.80 | success + coverage + tools + format |
| 2 вопроса + неправильное решение | 0.55 | partial coverage + tools + format |
| 1 вопрос + неправильное решение | 0.30 | tools + format |
| Пустой `<ACTIONS>` блок | 0.10 | только format |
| Мусор (нет действий) | 0.00 | ничего |

При этом reward spread 1.80 (gold vs garbage) — даже в группе из 4 одинаково неуспешных траекторий разные уровни evidence coverage дают разные rewards, и GRPO advantages ненулевые.

### 1.5 Verifier и метрики качества

TriageTrajectoryVerifier проверяет каждую траекторию по следующим критериям:

- **success**: правильный disposition + все evidence-группы покрыты + нет критических нарушений
- **evidence_coverage**: доля покрытых evidence-групп (0.0–1.0)
- **tool_calls**: количество вызовов инструментов
- **policy_violations**: нарушения правил безопасности (confirmation, hallucination, duplicate)
- **failure_reason**: конкретная причина неудачи (wrong_final_decision, hallucinated_slot, actions_exhausted, premature_finish, max_steps_exceeded)

Verifier используется и при evaluation (для подсчёта метрик), и при training (в reward_fn для расчёта evidence coverage).

## 2. Baseline: диагностика проблем

### 2.1 Первые попытки запуска

При запуске базовой модели Qwen3-8B через vLLM обнаружились три критические проблемы, из-за которых модель показывала 0% tool calls:

**Проблема 1: бюджет токенов.** Qwen3-8B при каждой генерации создаёт блок `<think>...</think>` для внутреннего рассуждения, который потребляет 200-300 токенов. При дефолтном `max_tokens=256` весь бюджет уходил на thinking, и модель не успевала сгенерировать ни одного действия.

**Проблема 2: формат вызова инструментов.** Промпт показывал инструменты в компактном формате `ask_question {"question_id":"<ID>"}`, и модель копировала именно этот формат. Но парсер действий требовал префикс `TOOL_CALL` — без него все вызовы трактовались как свободный текст.

**Проблема 3: Qwen3 thinking mode.** Параметр `enable_thinking=False` в `apply_chat_template` не полностью подавлял генерацию `<think>` блоков. Решение — добавление суффикса `/no_think` к user-сообщению, после чего модель генерировала пустой `<think></think>` и переходила к действиям.

После исправления этих проблем модель начала генерировать корректные TOOL_CALL (60/60 правильный формат), среднее число вызовов инструментов выросло с 0 до 4.8, но success rate остался 0% — модель делала агентные ошибки: дублировала вопросы, игнорировала протоколы, всегда выбирала одно и то же решение.

## 3. Чистый GRPO: мёртвый gradient

### 3.1 Попытка обучения

Первая попытка обучения использовала стандартный GRPO (Group Relative Policy Optimization). Конфигурация: batch=4, group_size=8, lr=5e-6, 8 completions на каждый промпт.

### 3.2 Коллапс обучения

На первых 114 шагах обучения наблюдалась полная стагнация:

- `completion_length` = 29-34 токена — модель генерировала один-единственный вызов `finish`
- `reward_std = 0.0` в 73% шагов — все 8 completions идентичны
- `loss = 0.0` в 73% шагов — нулевой advantage → нулевой gradient
- `reward ≈ -1.05` стабильно (fail + shaping penalties)

Типичный output модели в этом состоянии — все 8 completions в группе выглядели одинаково:

```
<ACTIONS>
TOOL_CALL {"name": "finish", "args": {"disposition": "BOOK_SAME_DAY"}}
</ACTIONS>
```

Одно действие, 29 токенов. Модель сколлапсировала в кратчайшую возможную траекторию с одним и тем же disposition для всех кейсов.

Это классический «death spiral» GRPO: модель сколлапсировала в детерминистическую стратегию, все completions в группе одинаковые, GRPO не может вычислить advantage, обучение замирает.

## 4. Решение: SFT + RL-PLUS с curriculum

### 4.1 Архитектура пайплайна

Для преодоления cold-start проблемы был разработан двухфазный пайплайн:

**Stage 0: SFT (Supervised Fine-Tuning)** на Oracle gold-траекториях. Oracle-решатель генерирует идеальные цепочки действий для каждого кейса. SFT обучает модель формату `<ACTIONS>...</ACTIONS>` с корректными `TOOL_CALL` вызовами, включая confirmation flow (`<CONFIRM>` → `book_visit`/`create_escalation`). 50 шагов при lr=1e-4 достаточно — loss достигает 0 уже на 25-м шаге.

**Stage 1: RL-PLUS** на лёгких кейсах (d1-d3). RL-PLUS расширяет стандартный GRPO добавлением expert (gold) term:

$$J(\theta) = E_{\text{on-policy}}[r(\theta) \cdot A] + \lambda \cdot E_{\text{expert}}[r_m(\theta) \cdot A_c]$$

Первый член — стандартный GRPO (group-normalized advantages). Второй — MIS (Multiple Importance Sampling) ratio по gold-траекториям с focal weighting. Ключевое преимущество: даже когда все on-policy completions fail (reward_std=0), gold term предоставляет ненулевой gradient signal.

### 4.2 Критические находки в процессе разработки

В ходе итеративной разработки были обнаружены и исправлены несколько глубоких проблем, наиболее значимые из которых связаны с reward-функцией (подробная эволюция описана в разделе 1.4):

**Reward inversion.** Verifier-штрафы за шаги и инструменты делали «ничего не делать» выгоднее чем «попробовать и ошибиться». Решение — exploration-friendly reward без негативных penalties (раздел 1.4, итерация 3).

**Binary → gradual reward.** Бинарный 0/1 reward не различал «почти правильно» и «полный мусор». Замена на компонентный reward дала spread 1.80 между gold и garbage.

**Key order mismatch.** `render_tool_call()` использовала `json.dumps(sort_keys=True)`, что ставило `"args"` перед `"name"` в JSON. SFT обучал модель неправильному порядку ключей, отличному от промпта и от того, что модель видела при inference.

**Per-row gold rewards.** Код использовал `batch_rows[0]["gold_reward"]` для всего batch из 4 кейсов с разными gold_reward — каждый кейс имеет свою длину Oracle-траектории и свой evidence coverage. Исправлено на per-row вычисление.

### 4.3 Оптимизации памяти

Qwen3-8B (~16 GB в bf16) с тремя копиями (policy + ref + behavior) не влезала в 96 GB при адекватном batch size. Были реализованы три оптимизации:

**disable_adapter как ref.** Вместо загрузки отдельной копии модели как reference для KL penalty, LoRA-адаптеры временно отключаются на policy model: `model.disable_adapter_layers()` → forward pass (= base model logprobs) → `model.enable_adapter_layers()`. Экономия: 4-16 GB.

**Chunked backward.** `_compute_loss_v2` разбивает rollouts на чанки по 4 sequence. Каждый чанк: forward → loss → `.backward()` → del activations → `empty_cache()`. Градиенты accumulate автоматически. Экономия: ~75% activation memory.

**Разделение generation и loss.** `torch.cuda.empty_cache()` между фазами. KV cache и grad activations никогда не сосуществуют.

Итоговый VRAM: ~50 GB при batch=8, group=4 (было 123 GB при batch=4, group=8 без оптимизаций).

## 5. Эксперименты с моделями разного размера

### 5.1 Qwen2.5-1.5B-Instruct

Первая попытка — самая маленькая модель для максимальной скорости итераций (~3 GB weights, batch=4×group=16 влезал в 96 GB без оптимизаций).

Результат: модель быстро выучила формат (format=100%), но сколлапсировала в одну стратегию — ESCALATE_NOW для всех кейсов. Debug samples на step 50: оба примера — escalation, даже для случаев с лёгкой сыпью или головной болью. acc скачет 4-65% не потому что модель учится, а потому что рандомно попадает в батчи, где target действительно ESCALATE_NOW.

Диагноз: 1.5B не хватает reasoning capacity чтобы различать четыре disposition. Модель выучила самый длинный шаблон (с CONFIRM + escalation) и применяет его везде.

### 5.2 Qwen2.5-3B-Instruct

Вторая попытка — компромисс между скоростью и качеством (~6 GB weights).

Улучшения по сравнению с 1.5B:
- Disposition diversity: модель генерирует SELF_CARE, BOOK_ROUTINE и ESCALATE_NOW (не коллапсирует в одну стратегию)
- Debug samples показывают разные flows: lookup_protocol → ask × 2 → finish(SELF_CARE) и ask × 3 → CONFIRM → escalate
- acc стабилизировался на ~25-35%

Но reward curve вышел на плато уже к step 30 (reward 0.9→1.1, затем flat). 3B хватает для формата и базового reasoning, но недостаточно для тонкого различения disposition на основе собранного evidence.

Дополнительный плюс Qwen2.5: отсутствие `<think>` блока, что экономит 300+ токенов на каждой генерации.

### 5.3 Qwen3-8B

Финальный выбор. Качественный скачок:

| Метрика (step 50) | 1.5B | 3B | 8B |
|-------------------|:----:|:---:|:---:|
| acc | 8% | 35% | 35% |
| Disposition diversity | только ESCALATE | 3 типа | **4 типа** |
| Confirmation flow | шаблонный | частичный | **полный** |
| list_slots → book_visit | никогда | редко | **часто** |
| Evidence coverage | 0.40 | 0.65 | **0.78** |

8B первой научилась полной цепочке booking: ask_question × 3 → list_slots(ROUTINE) → CONFIRM → book_visit → finish(BOOK_ROUTINE). 3B и 1.5B никогда не генерировали `list_slots`.

Цена: ~16 GB weights, потребовались все memory оптимизации (disable_adapter, chunked loss), и модель более чувствительна к learning rate (gradient explosion при lr=1e-5 после step 60).

### 5.4 Выводы по размеру модели

Для tool-use агента с четырьмя типами решений и confirmation flow:
- **1.5B**: выучит формат, но не policy (disposition collapse)
- **3B**: базовый reasoning, но плато на ~25% success
- **8B**: полноценный triage workflow, 20% eval success (ограничен trajectory mode и hallucinations, не capacity)

Интересное наблюдение: разрыв между 3B и 8B проявляется не в overall accuracy, а в **качестве траекторий** — 8B генерирует структурно более сложные и корректные цепочки действий.

## 6. Результаты обучения

### 6.1 SFT Phase

SFT на 50 шагах при lr=1e-4 полностью решил проблему формата. Loss упал с 1.5 до ~0 за 25 шагов. После SFT модель генерирует идеальные `<ACTIONS>` блоки с корректным JSON, включая confirmation flow.

### 6.2 RL-PLUS Stage 1 (easy, d1-d3)

Обучение: 60 шагов при lr=1e-5, batch=8, group=4, λ_ext=1.0.

Наблюдения из training логов:
- `format_rate = 100%` — формат стабильно корректный
- `reward_int_mean` вырос с 0.78 до ~1.05 за первые 30 шагов, затем плато
- KL стабилизировался на 0.6-0.8 — policy не убегала от base model
- Debug samples показали все четыре типа disposition и полный confirmation flow

Пример debug sample (step 50) — модель генерирует полный booking flow для кейса с сыпью:

```
<ACTIONS>
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_SWELLING"}}
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_RASH_SPREAD"}}
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_DURATION"}}
TOOL_CALL {"name": "list_slots", "args": {"urgency": "ROUTINE"}}
I recommend a routine visit.
<CONFIRM>{"tool":"book_visit","slot_id":"ROUTINE_1"}</CONFIRM>
TOOL_CALL {"name": "book_visit", "args": {"slot_id": "ROUTINE_1"}}
TOOL_CALL {"name": "finish", "args": {"disposition": "BOOK_ROUTINE"}}
</ACTIONS>
```

Для сравнения, до обучения (чистый GRPO) модель на аналогичном кейсе генерировала:
```
<ACTIONS>
TOOL_CALL {"name": "finish", "args": {"disposition": "BOOK_SAME_DAY"}}
</ACTIONS>
```

При продолжении обучения (шаги 70-90) наблюдался gradient explosion (grad вырос до 20+) и format collapse (fmt упал до 43%), что указывает на чувствительность 8B модели к learning rate. Лучший checkpoint: step 60.

### 6.3 Evaluation: Trained vs Baseline

Evaluation на 100 кейсах в trajectory mode:

| Метрика | Baseline d1 | Trained d1 | Baseline d3 | Trained d3 |
|---------|:-----------:|:----------:|:-----------:|:----------:|
| Success rate | 0% | **20%** | 0% | **19%** |
| Tool calls | 0.4 | **3.5** | 0.6 | **4.3** |
| Evidence coverage | 14% | **78%** | 25% | **81%** |
| Policy violations | 0.03 | 0.73 | 0.00 | 0.79 |

Baseline (необученная Qwen3-8B) показала 0% success — модель не знает формат triage environment и не может сгенерировать корректный `<ACTIONS>` блок. Все 100 кейсов завершились с `actions_exhausted`.

Trained модель достигла 20% success rate. Основные причины ошибок:

**hallucinated_slot (28%)** — самая частая ошибка. Модель генерирует корректную структуру booking flow, но использует slot_id «ROUTINE_1», которого может не быть в конкретном кейсе. Пример (кейс TRIAGE_RASH, сыпь):

```
ask_question(Q_SWELLING) → ask_question(Q_RASH_SPREAD) → list_slots(ROUTINE) →
CONFIRM{"slot_id":"ROUTINE_1"} → book_visit("ROUTINE_1") → finish(BOOK_ROUTINE)
```

Структура идеальна, evidence coverage 100%, но slot_id захардкожен из SFT gold-траекторий. В trajectory mode модель не видит реальный ответ list_slots и не может узнать актуальные ID.

**actions_exhausted (20%)** — модель тратит весь token budget (2048 токенов) на `<think>` reasoning и не успевает сгенерировать `<ACTIONS>` блок. Qwen3-8B генерирует 300-600 токенов рассуждений даже с `/no_think`.

**wrong_final_decision (8%)** — модель собирает evidence правильно, но выбирает неправильный disposition. Например, кейс требует ESCALATE_NOW, а модель выбирает BOOK_ROUTINE.

Evidence coverage 78-81% показывает, что модель научилась собирать анамнез — она задаёт релевантные вопросы и покрывает большую часть evidence-групп.

## 7. Попытка Stage 2 и коллапс

Stage 2 (easy + medium кейсы с replay) при lr=5e-6 привёл к немедленному gradient explosion: grad вырос до 14+ за 50 шагов, format_rate упал до 28%. Debug sample на step 50 показал дегенерацию — вместо корректного CONFIRM блока модель генерировала бессвязный текст:

```
<ACTIONS>
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_TEMPERATURE"}}
TOOL_CALL {"name": "ask_question", "args": {"question_id": "Q_BLOOD_PRESENT"}}
This case should be escalated urgently.
<CONFIRM>{"tool":"create_1","tool_type":"create_escalation","level":"ROUTINE Trusted
Pharmacist Visit, then ROUTINE Routine Blood Test Visit within 5 Working Days, then
if Blood Cultures show E. coli growth then SAME_DAY Emergency Antibiotic IV Consult
Visit Achieved in Time-Sensitive Emergency Support Framework Study Abstract Achieved
by Emergency Support This Urgent Bloodstream Infection Antibody Cascade Kit KitKit...
```

Снижение lr до 1e-6 замедлило коллапс (format удерживался на ~90% первые 30 шагов), но к step 40 gradient снова начал расти.

Причина: distribution shift при переходе от easy к medium кейсам в сочетании с высоким LoRA rank (r=64) делает policy слишком пластичной. Medium кейсы требуют более длинных траекторий с другими паттернами (ESCALATE_NOW с confirmation flow), и policy пытается резко перестроиться.

## 8. Анализ и дальнейшие направления

### 8.1 Почему 20% — потолок текущего подхода

**Trajectory vs Interactive mode.** Модель обучалась в trajectory mode (один промпт → полный `<ACTIONS>` блок), но реальная triage система работает интерактивно (один шаг за раз, с обратной связью от среды). Модель не видит ответов на вопросы — она «угадывает» всю цепочку действий за один проход.

**Hallucination.** 28% ошибок — hallucinated slot IDs. Модель выучила шаблон `ROUTINE_1`, но не знает конкретные ID слотов для каждого кейса (они доступны только через `list_slots`).

**Think block overhead.** Qwen3-8B тратит 300-600 токенов на `<think>` reasoning при inference, оставляя мало budget для действий.

### 8.2 Как двигаться дальше

**Interactive RL.** Обучение в том же формате, в котором модель будет использоваться — step-by-step с обратной связью от среды. Это устранит проблему hallucinated slots (модель увидит реальные ID из `list_slots`).

**Модель без thinking.** Qwen2.5-Instruct не генерирует `<think>` блоки, что освобождает 300+ токенов. Однако 3B версия оказалась слишком слабой для disposition reasoning — нужна 7B.

**Градуальное увеличение сложности.** Stage 2 коллапсировал из-за резкого distribution shift. Более плавный переход (10% medium → 20% → 30%...) с очень маленьким lr (3e-7) может стабилизировать обучение.

**LoRA rank.** r=64 делает policy слишком пластичной. Снижение до r=16-32 может улучшить стабильность при curriculum переходах, за счёт более медленного обучения.

**Reward engineering.** Добавление штрафа за hallucinated entities (которые не присутствуют в state snapshot) напрямую в reward function.

## 9. Итоги

Работа демонстрирует полный цикл RL-обучения LLM-агента для tool-use задачи:

1. Диагностика baseline → выявление cold-start проблемы GRPO
2. SFT warm-start → модель выучила формат и базовые паттерны
3. RL-PLUS с gradual reward → модель научилась triage policy
4. Системные оптимизации памяти → обучение 8B модели на 96 GB GPU

Итоговый результат: **0% → 20% success rate**, **0.4 → 3.5 tool calls**, **14% → 78% evidence coverage**. Модель из полностью неработоспособной превратилась в функционирующего triage-агента, владеющего полным workflow: сбор анамнеза, проверка протоколов, confirmation flow, бронирование визитов и эскалация.
