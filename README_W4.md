# Week 4 — RL-PLUS Safe Medical Triage (Инструкция по запуску)

Ниже — инструкция по воспроизведению решения для ДЗ «Неделя 4 / Избранные темы в ИИ» (обучение LLM-агента методом SFT + RL-PLUS для медицинской триажной системы).

Репозиторий рассчитан на запуск **из корня проекта** (рядом лежат `.ipynb`, `triage/`, `runtimes/`).

---

## 1) Что делает решение

**Задача:** по описанию симптомов пациента провести полный цикл медицинского триажа — собрать анамнез через инструменты (ask_question, lookup_protocol, list_slots, book_visit, create_escalation, finish) и вынести одно из четырёх решений: SELF_CARE, BOOK_ROUTINE, BOOK_SAME_DAY или ESCALATE_NOW.

Пайплайн состоит из двух частей:

1) **Обучение (TRAIN)**: SFT warm-start на Oracle gold-траекториях → RL-PLUS (GRPO + expert gold term) с curriculum по сложности. Реализованы memory-оптимизации: disable_adapter как ref model, chunked backward, exploration-friendly reward.
2) **Оценка (EVAL)**: инференс через **vLLM** с LoRA-адаптером и расчёт метрик (success rate, evidence coverage, tool calls, policy violations) на eval-наборах для baseline и trained модели.

---

## 2) Структура проекта и важные файлы

Минимально важное (то, что нужно кураторам):

- **`04_curriculum_rl_plus.ipynb`** — основной ноутбук обучения (SFT → RL-PLUS / curriculum)
- **`05_eval_quick.ipynb`** — основной ноутбук оценки (vLLM: baseline vs trained на d1, d3)

Ноутбуки-предыстория (для контекста, не обязательны):

- `01_env_smoke_and_oracle.ipynb` — smoke-тест среды и Oracle-решателя
- `02_baseline_vllm_eval_fixed_notebook_native.ipynb` — baseline eval (v5.2, исторический)
- `03_grpo_training.ipynb` — чистый GRPO без SFT (демонстрация death spiral)
- `05_eval_vllm.ipynb` — полная оценка d1..d10

Среда и агент (`triage/`):

- `env.py` — `SafeTriageEnv` — пошаговая среда (reset, step, reward)
- `schema.py` — `TriageData` — dataclass кейса (patient, QA, slots, evidence groups)
- `generator.py` — `CaseGenerator` — генерация кейсов d1–d10
- `oracle.py` — `OracleSolver` — детерминистическая 100%-policy для gold-траекторий
- `verifier.py` — `TriageTrajectoryVerifier` — проверка траекторий (success, violations)
- `prompting.py` — промпты (system, user), trajectory/interactive mode, `/no_think` для Qwen3
- `action_parser.py` — парсинг TOOL_CALL, render_tool_call
- `trajectory_text.py` — extract_actions, extract_single_action, normalize
- `rule_engine.py` — правила безопасности (confirmation, hallucination, duplicates)
- `metrics.py` — агрегация метрик, failure reasons
- `catalogs.py`, `constants.py`, `text_templates.py` — справочники симптомов, вопросов, шаблонов
- `io_utils.py`, `logging_utils.py`, `artifacts.py` — JSONL IO, логирование, manifest

Обучение (`runtimes/train_unsloth/`):

- `rl_plus_trainer.py` — **главный**: `RLPlusTrainer` (GRPO + gold expert term + chunked loss)
- `sft_trainer.py` — `train_sft_lora` — SFT на Oracle gold-траекториях
- `triage_rl_plus_compat.py` — адаптер RL-PLUS → triage env (reward_fn, verify_fn, gold_fn)
- `config.py` — `UnslothGRPOConfig`
- `export_merged_for_vllm.py` — экспорт base+LoRA → merged model

Оценка (`runtimes/eval_vllm/`):

- `api.py` — `run_vllm_rollouts` — запуск eval (interactive/trajectory mode)
- `config.py` — `VLLMEvalConfig` — конфигурация eval (model, lora, max_tokens...)
- `vllm_utils.py` — обёртки над vLLM (generate, LoRA request)

Скрипты/прочее:

- `tests/` — 45 unit-тестов (env, oracle, verifier, parser, prompting)
- `scripts/` — CLI-скрипты (generate_train, generate_eval, run_oracle)
- `configs/` — примеры JSON-конфигов
- `requirements/` — зависимости для train и eval окружений

Папки-артефакты:

- `data/` — сгенерированные датасеты (`train.jsonl`, `eval/eval_d1.jsonl`...`eval_d10.jsonl`)
- `artifacts/` — результаты обучения (адаптеры, checkpoints, метрики, manifest)

---

## 3) Окружения (env) и требования

Рекомендовано держать **2 отдельных окружения**:

- **TRAIN-env** (Unsloth + Transformers + PEFT)
- **EVAL-env** (vLLM + Transformers)

Причина: vLLM и Unsloth предъявляют разные требования к версиям torch/CUDA и удобнее не смешивать.

### Железо
- Для **обучения** Qwen3-8B нужен NVIDIA GPU с ≥48 GB VRAM (использовалась RTX 6000 Pro, 96 GB). Для моделей 1.5B–3B достаточно 24 GB.
- Для **оценки** (vLLM) рекомендован GPU с ≥24 GB VRAM.

---

## 4) Установка зависимостей

### 4.1 TRAIN-env
```bash
cd <project_root>

python -m venv .train-env
source .train-env/bin/activate
pip install -U pip

pip install -r requirements/train-unsloth.txt
pip install -r requirements/base-core.txt

# Опционально (для онлайн-графиков обучения):
pip install comet-ml
```

### 4.2 EVAL-env
```bash
cd <project_root>

python -m venv .eval-env
source .eval-env/bin/activate
pip install -U pip

pip install -r requirements/eval-vllm.txt
pip install -r requirements/base-core.txt
```

> Примечание: базовая модель (`Qwen/Qwen3-8B`) скачивается с Hugging Face при первом запуске (если не закэширована локально).

---

## 5) Запуск обучения (TRAIN)

1) Активируйте окружение обучения:
```bash
cd <project_root>
source .train-env/bin/activate
```

2) Запустите Jupyter:
```bash
jupyter lab
```

3) Откройте и выполните **Run All**:
- `04_curriculum_rl_plus.ipynb`

### 5.1 Где настраивать параметры

В первой code-ячейке ноутбука — **флаги управления**:
```python
RUN_SFT         = True    # Stage 0: SFT на Oracle gold
FORCE_SFT       = False   # True → перетренировать даже если адаптер есть

RUN_STAGE1_EASY = True    # Stage 1: RL-PLUS на d1-d3
FORCE_STAGE1    = False

RUN_STAGE2_MEDIUM = True  # Stage 2: RL-PLUS на d1-d7 (replay)
FORCE_STAGE2    = False

RUN_STAGE3_HARD = True    # Stage 3: RL-PLUS на d1-d10 (replay)
FORCE_STAGE3    = False

RUN_EXPORT      = True    # Экспорт merged model для vLLM
FORCE_EXPORT    = False
```

Каждый этап идемпотентен — при повторном Run All пропускает уже обученные стадии (если не выставлен `FORCE_*`).

В ячейке `STAGES` / `BASE_RL_CFG` — ключевые гиперпараметры:

| Параметр | Значение | Описание |
|----------|:--------:|----------|
| `TRAIN_MODEL` | `Qwen/Qwen3-8B` | Базовая модель |
| `batch_size_prompts` | 8 | Промптов в батче |
| `group_size` | 4 | Completions на промпт (diversity для GRPO) |
| `lr` | 1e-5 (stage1), 5e-6 (stage2) | Learning rate |
| `max_steps` | 150 | Шагов на стадию |
| `lambda_ext` | 1.0 | Вес gold (expert) term в RL-PLUS |
| `loss_chunk_size` | 4 | Chunked backward (0 = off), экономит activation memory |
| `ref_use_disable_adapter` | True | Ref model = base без LoRA (экономит 16 GB) |
| `save_every_steps` | 30 | Частота checkpoint'ов |
| `comet_project` | `"triage-rl-plus"` | Comet.ml проект (None = off) |

### 5.2 Comet.ml (онлайн-визуализация, опционально)

Если установлен `comet-ml`:

1) Создайте файл `~/.comet.config`:
```ini
[comet]
api_key=YOUR_API_KEY_HERE
```
(API key берётся на https://www.comet.com/account-settings/apiKeys)

2) В ноутбуке задайте:
```python
COMET_PROJECT = "triage-rl-plus"
```

Ключевые графики в Comet (рекомендуется включить Smoothing=0.8):
- `train/reward_int_mean` — главный индикатор обучения
- `train/kl` — здоровье policy (если > 2.0 — policy убегает)
- `train/correct_rate` — success rate на training batch (шумный, но показывает тренд)

### 5.3 Какие артефакты появляются после train

```
artifacts/runs/curriculum_rl_plus_8b/
├── sft_adapter/                    # Stage 0: SFT LoRA адаптер
├── sft_history.json                # SFT loss history
├── gold_train_rows.jsonl           # Cached Oracle gold-траектории
├── stage1_easy/
│   └── adapter/                    # LoRA адаптер
│       ├── checkpoint-30/          # Промежуточные checkpoints
│       ├── checkpoint-60/          # ← обычно лучший
│       ├── checkpoint-best/        # Авто-лучший по smoothed reward
│       ├── adapter_model.safetensors
│       └── adapter_config.json
├── stage2_medium/                  # (если запускалось)
│   └── adapter/
├── merged_16bit/                   # Merged model для vLLM (если экспортирован)
└── manifest.json                   # Метаданные для eval
```

### 5.4 Где смотреть метрики обучения

В логах ноутбука печатаются строки вида:
```
step 10/150 | loss=-0.0315 int=0.0011 ext=-0.0326 kl=1.3443
  r_m=1.115 r=0.925±0.391 acc=0.156 parse=1.000 fmt=1.000
  len=162.1 grad=0.53 lr=1.00e-05 t=44.03s
```

| Метрика | Что значит | Хорошо |
|---------|-----------|--------|
| `loss` | Суммарный loss (int + λ·ext) | Отрицательный |
| `kl` | KL-divergence от ref | < 2.0 |
| `acc` | Success rate на batch (шумный!) | Растёт |
| `fmt` | Format compliance | = 1.000 |
| `grad` | Gradient norm | < 5.0 |
| `r` | Mean reward ± std | Растёт |

Каждые 50 шагов печатается debug sample — пример генерации модели.

В конце обучения:
```
[best] step 42, smoothed reward = 1.1234
[best] saved → .../checkpoint-best
```

### 5.5 Resume обучения с checkpoint

Если обучение прервано или нужно продолжить с лучшего checkpoint — в ячейке training loop перед `for stage_idx...` добавьте:
```python
current_adapter = str(ARTIFACTS_DIR / "stage1_easy" / "adapter" / "checkpoint-60")
STAGES[0]["max_steps"] = 90  # оставшиеся шаги (150 - 60)
```

### 5.6 Gradient explosion

Если наблюдается `grad > 5` и `format_rate < 0.9` — модель дестабилизировалась:

1) Останавливайте обучение
2) Берите предыдущий checkpoint (`checkpoint-best` или последний стабильный)
3) Снижайте `lr` в 2–5 раз
4) Опционально: `max_grad_norm=0.5` (жёстче клиппинг)

---

## 6) Запуск финальной оценки (EVAL)

1) Активируйте окружение оценки:
```bash
cd <project_root>
source .eval-env/bin/activate
```

2) Запустите Jupyter и выполните **Run All**:
- `05_eval_quick.ipynb`

### 6.1 Что делает eval-ноутбук

- Загружает Qwen3-8B как base model
- Подключает LoRA-адаптер из `checkpoint-60` (через vLLM `enable_lora`)
- Прогоняет 100 кейсов на d1 и d3 в trajectory mode
- Считает метрики: success rate, reward, tool calls, evidence coverage
- Повторяет без адаптера (baseline)
- Печатает summary-таблицу

### 6.2 Где настраивать параметры eval

В ячейке с конфигурацией:
```python
BASE_MODEL = "Qwen/Qwen3-8B"
ADAPTER_DIR = str(PROJECT_ROOT / "artifacts" / "runs" / "curriculum_rl_plus_8b"
                  / "stage1_easy" / "adapter" / "checkpoint-60")
N_CASES = 20  # кейсов на difficulty (увеличить для точной оценки)
```

Важные параметры в `COMMON_CFG`:

| Параметр | Значение | Описание |
|----------|:--------:|----------|
| `rollout_mode` | `"trajectory"` | Должен совпадать с режимом обучения |
| `max_model_len` | 4096 | Максимальная длина контекста vLLM |
| `max_tokens` | 2048 | Бюджет генерации (Qwen3 тратит ~300 на `<think>`) |
| `enforce_eager` | True | Пропустить CUDA graph compilation (быстрый старт) |
| `max_lora_rank` | 64 | Должен совпадать с `lora_r` при обучении |

### 6.3 Результаты eval (куда сохраняются)

После выполнения:
```
artifacts/eval_stage1/
├── trained_d1.traj.jsonl       # Полные траектории trained модели
├── trained_d1.metrics.jsonl    # Метрики per-case
├── baseline_d1.traj.jsonl
├── baseline_d1.metrics.jsonl
├── trained_d3.traj.jsonl
├── trained_d3.metrics.jsonl
├── baseline_d3.traj.jsonl
└── baseline_d3.metrics.jsonl
```

В ноутбуке печатается summary:
```
              success_rate  avg_reward  avg_tool_calls  evidence_coverage
Baseline d1          0.00       -1.09             0.4               0.14
Trained  d1          0.20       -0.98             3.5               0.78
Baseline d3          0.00       -1.08             0.6               0.25
Trained  d3          0.19       -1.05             4.3               0.81
```

### 6.4 Если не хватает GPU памяти в eval

- Уменьшить `max_model_len` до 2048
- `gpu_memory_utilization=0.85`
- `enforce_eager=True` (экономит memory на CUDA graphs)
- Уменьшить `max_tokens` до 1024 (но для Qwen3 нужно ≥1024 из-за `<think>`)

### 6.5 Полная оценка d1–d10

Для полной оценки используйте `05_eval_vllm.ipynb`, раскомментировав:
```python
EVAL_DIFFICULTIES = list(range(1, 11))
N_CASES = 100
```

---

## 7) Тесты

```bash
source .train-env/bin/activate  # или .eval-env
cd <project_root>
PYTHONPATH=. python -m pytest tests/ -q
```

Ожидается 45 тестов, все PASS. Тесты покрывают: env, oracle, verifier, action_parser, trajectory_text, prompting, generator, config, workflows.

---

## 8) Контрольный список для проверки

1) После TRAIN:
- [ ] существует `artifacts/runs/curriculum_rl_plus_8b/sft_adapter/`
- [ ] существует `artifacts/runs/curriculum_rl_plus_8b/stage1_easy/adapter/checkpoint-60/`
- [ ] существует `data/train.jsonl` и `data/eval/eval_d1.jsonl`
- [ ] в логах ноутбука: `format_rate = 1.000`, `reward > 0.9`

2) После EVAL:
- [ ] существует `artifacts/eval_stage1/trained_d1.metrics.jsonl`
- [ ] существует `artifacts/eval_stage1/baseline_d1.metrics.jsonl`
- [ ] trained success rate > 0% (ожидается ~20%)
- [ ] baseline success rate = 0% (ожидаемо)
- [ ] в ноутбуке напечатана summary-таблица

---

## 9) Известные ограничения

- **Trajectory mode**: модель не видит ответов пациента — планирует всю цепочку за один проход. Ограничивает success rate (~20%).
- **Qwen3 `<think>` overhead**: модель тратит 300–600 токенов на reasoning. При `max_tokens < 1024` часть completions обрезается.
- **Hallucinated slot IDs**: модель заучивает `ROUTINE_1` из gold-траекторий, но в реальных кейсах ID могут отличаться.
- **Stage 2 gradient explosion**: при переходе easy→medium policy дестабилизируется при lr > 1e-6. Рекомендуется использовать Stage 1 checkpoint.
- **LoRA rank sensitivity**: r=64 делает policy пластичной — хорошо для обучения, но нестабильно при distribution shift.
