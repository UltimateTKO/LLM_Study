###############################################
# ファインチューニング
###############################################
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import transformers
from accelerate import infer_auto_device_map
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
# import deepspeed
# from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"
# MODEL_NAME = "rinna/llama-3-youko-8b"
# DATASET_NAME = "izumi-lab/llm-japanese-dataset"
DATASET_NAME = "kunishou/databricks-dolly-15k-ja"
PEFT_NAME = "lora-rinna-3.6b"
OUTPUT_DIR = "lora-rinna-3.6b-results"

# トークナイザー作成
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# トークナイザーのスペシャルトークンの確認。
# print(tokenizer.special_tokens_map)
# print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
# print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
# print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
# print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)

# model = AutoModelForSequenceClassification.from_pretrained("rinna/japanese-gpt-neox-3.6b-instruction-sft-v2", num_labels=5)

# training_args = TrainingArguments(output_dir="test_trainer")

# コンテキスト長
CUTOFF_LEN = 256
# トークナイズ
def tokenize(prompt, tokenizer):
	return tokenizer(
		prompt,
		truncation=True,
		max_length=CUTOFF_LEN,
		padding=False,
	)

# トークナイズの動作確認
# tokenize_temp = tokenize("hi there", tokenizer)
# print(tokenize_temp)

dataset = load_dataset(DATASET_NAME)
# print(dataset["train"][0])
# print(dataset["train"][100])

# プロンプトテンプレートの準備
def generate_prompt(data_point):
	if data_point["input"]:
		result = f"""### 指示:
{data_point["instruction"]}

### 入力:
{data_point["input"]}

### 回答:
{data_point["output"]}"""
	else:
		result = f"""### 指示:
{data_point["instruction"]}

### 回答:
{data_point["output"]}"""

	# 改行→<NL>
	result = result.replace('\n', '<NL>')
	return result

# プロンプトテンプレートの確認
DATASET_LENGTH = 0
print(generate_prompt(dataset["train"][DATASET_LENGTH]))

# 学習データと検証データの準備peft
VAL_SET_SIZE = 2000

train_val = dataset["train"].train_test_split(
	test_size = VAL_SET_SIZE, shuffle=True, seed=42
)
train_data = train_val["train"]
val_data = train_val["test"]
train_data = train_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))
val_data = val_data.shuffle().map(lambda x: tokenize(generate_prompt(x), tokenizer))

# モデルの準備
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
	MODEL_NAME,
	# torch_dtype=torch.bfloat16,
	# int8bit量子化
	# load_in_8bit=True,
	# device_map = "auto"
)

# LoRAのパラメータ
lora_config = LoraConfig(
	r = 8,
	lora_alpha = 16,
	target_modules = ["query_key_value"],
	# target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
	lora_dropout = 0.05,
	bias = "none",
	task_type = TaskType.CAUSAL_LM
)

# モデルの前処理 int8bit量子化
# model = prepare_model_for_int8_training(model)
model = prepare_model_for_kbit_training(model)

# LoRAモデルの準備
model = get_peft_model(model, lora_config)

# 学習可能パラメータの確認
model.print_trainable_parameters()

# トレーナーの準備(コンスタント)
EVAL_STEPS = 1000
SAVE_STEPS = 1000
LOGGING_STEPS = 20

# トレーナーの準備
trainer = transformers.Trainer(
	model = model,
	train_dataset = train_data,
	eval_dataset = val_data,
	args = transformers.TrainingArguments(
		num_train_epochs = 3,
		learning_rate = 3e-4,
		logging_steps = LOGGING_STEPS,
		evaluation_strategy = "steps",
		save_strategy = "steps",
		bf16_full_eval=True,
		eval_steps = EVAL_STEPS,
		save_steps = SAVE_STEPS,
		output_dir = OUTPUT_DIR,
		report_to = "none",
		save_total_limit = 3,
		push_to_hub = False,
		auto_find_batch_size = True,
		# per_device_train_batch_size=1,
		# deepspeed = "./deepspeed_config.json",
	),
	data_collator = transformers.DataCollatorForLanguageModeling(tokenizer, mlm = False),
)

print("train_batch_size:", trainer._train_batch_size)
print("deepspeed_enabled:", trainer.is_deepspeed_enabled)

# 学習の実行
model.config.use_cache = False
trainer.train()
model.config.use_cache = True

# LoRAモデルの保存
trainer.model.save_pretrained(PEFT_NAME)
