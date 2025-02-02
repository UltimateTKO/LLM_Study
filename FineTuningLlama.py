import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import deepspeed
from deepspeed_config import zero3_config, zero2_config
import transformers
from datasets import load_dataset
from transformers.integrations import HfDeepSpeedConfig
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, DataCollatorForLanguageModeling, Trainer
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import traceback
import evaluate

# 1コア1スレッド化
import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import MyLogging
my_logging = MyLogging.FileLogger()

MODEL_NAME = "rinna/llama-3-youko-8b-instruct"
# DATASET_NAME = "izumi-lab/llm-japanese-dataset"
DATASET_NAME = "kunishou/databricks-dolly-15k-ja"
# OUTPUT_DIR = "lora-llama-3-youko-8b-results-pt"
OUTPUT_DIR = "lora-llama-3-youko-8b-instruct-results-pt-test2"
PEFT_NAME = "lora-llama-3-youko-8b"

torch.cuda.empty_cache()

# Datasetの作成
class SimpleDataset(Dataset):
	def __init__(self, tokenizer, datalists):
		self.tokenizer = tokenizer
		self.datalists = datalists

	def __len__(self):
		return len(self.datalists)

	def __getitem__(self, idx):
		encoded = self.tokenizer(
			self.tokenizer.apply_chat_template(self.generate_prompt(self.datalists[idx]), tokenize=False),
			return_tensors="pt",
			padding="max_length",
			truncation=True,
			max_length=256)
		my_logging.debug(encoded)
		input_ids = encoded["input_ids"].squeeze(0)  # (1, seq_len) → (seq_len)
		attention_mask = encoded["attention_mask"].squeeze(0)  # (1, seq_len) → (seq_len)
		labels = input_ids.clone()  # ラベルとしてinput_idsそのものを使用
		return input_ids, attention_mask, labels

	# プロンプトテンプレートの準備
	def generate_prompt(self, data_point):
		if data_point["input"]:
			template = [
				{"role": "system", "content": "あなたは誠実で優秀なアシスタントです。どうか、簡潔かつ正直に答えてください。"},
				{"role": "user", "content": data_point["instruction"] + "\n" + data_point["input"]},
				{"role": "assistant", "content": f"""{data_point["output"]} by平良"""},
			]
		else:
			template = [
				{"role": "system", "content": "あなたは誠実で優秀なアシスタントです。どうか、簡潔かつ正直に答えてください。"},
				{"role": "user", "content": data_point["instruction"]},
				{"role": "assistant", "content": f"""{data_point["output"]} by平良"""},
			]
		return template

try:
	tokenizer = AutoTokenizer.from_pretrained(
		MODEL_NAME,
		# use_fast=False,
		# trust_remote_code=True,
	)

	# eos_tokenの設定
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		# tokenizer.pad_token = "<|eot_id|>"
		tokenizer.pad_token_id = tokenizer.eos_token_id

	# max_memory={i: "6GB" for i in range(torch.cuda.device_count())}

	# device_map = {
	# 	"model.embed_tokens": 0,
	# }
	# for i in range(0, 32, 1):
	# 	layers_gpu_num = i % 4
	# 	device_map[f"model.layers.{i}"] = layers_gpu_num
	# device_map["model.norm"] = 1
	# device_map["model.rotary_emb"] = 2
	# device_map["lm_head"] = 3

	# model = AutoModelForCausalLM.from_pretrained(
	# 	MODEL_NAME,
	# 	torch_dtype=torch.bfloat16,
	# 	low_cpu_mem_usage=True,
	# 	# trust_remote_code=True,
	# 	# device_map=device_map,
	# )

	local_model_path = snapshot_download(MODEL_NAME)
	with init_empty_weights():
		model = AutoModelForCausalLM.from_pretrained(
			MODEL_NAME,
			torch_dtype=torch.bfloat16,
			low_cpu_mem_usage=True
		)

	model = load_checkpoint_and_dispatch(
		model,
		local_model_path,
	)

	# zero3_config_param = zero3_config(2, 4)
	# dschf = HfDeepSpeedConfig(zero3_config_param)

	model.gradient_checkpointing_enable()

	# LoRA設定
	lora_config = LoraConfig(
		r = 8,
		lora_alpha = 16,
		target_modules = [
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
		],
		# target_modules = ['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj'],
		lora_dropout = 0.05,
		bias = "none",
		task_type = TaskType.CAUSAL_LM
	)

	# LoRA モデルに適用
	model = get_peft_model(model, lora_config)

	# pad_tokenの設定
	if tokenizer.pad_token is None:
		# eos_token を代用
		# tokenizer.pad_token = tokenizer.eos_token
		# tokenizer.pad_token_id = tokenizer.eos_token_id
		model.config.pad_token_id = model.config.eos_token_id

	# データセットのインスタンス作成
	loadDataset = load_dataset(DATASET_NAME)

	# print(loadDataset["train"][0])
	dataset = SimpleDataset(tokenizer, loadDataset["train"].select(range(3000)))
	# dataloader = DataLoader(dataset)

	#deepspeed
	epochs=1

	# metric = evaluate.load("accuracy")
	# def compute_metrics(eval_pred):
	# 	logits, labels = eval_pred
	# 	predictions = np.argmax(logits, axis=-1)
	# 	return metric.compute(predictions=predictions, references=labels)

	training_args = TrainingArguments(
		output_dir=OUTPUT_DIR,
		evaluation_strategy="no",
		num_train_epochs=epochs,  # エポック数
		logging_steps=100,  # 途中経過を表示する間隔
		per_device_train_batch_size=1,
		bf16=True,
		gradient_accumulation_steps=4,
		deepspeed="ds_zero3_config.json",
	)

	print(f"ああああああああああああああああああああdist.get_rank:{dist.get_rank()}")
	print(dist.get_world_size())
	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=dataset,
		# compute_metrics=compute_metrics,
	)
	model.config.use_cache = False
	trainer.train()
	model.config.use_cache = True

	# LoRAモデルの保存
	trainer.model.save_pretrained(PEFT_NAME)

except Exception as e:
	print("エラーが出た！！！！")
	# my_logging.error(e.)
	with open("log/error.log", 'w') as f:
		traceback.print_exc(file=f)

	raise e

