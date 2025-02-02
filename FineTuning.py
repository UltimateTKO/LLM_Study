import torch
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
import deepspeed
from deepspeed_config import zero3_config, zero2_config
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, AutoConfig
from huggingface_hub import snapshot_download
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import traceback

# 1コア1スレッド化
import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import MyLogging
my_logging = MyLogging.FileLogger()

MODEL_NAME = "rinna/llama-3-youko-8b-instruct"
# DATASET_NAME = "izumi-lab/llm-japanese-dataset"
DATASET_NAME = "kunishou/databricks-dolly-15k-ja"
# OUTPUT_DIR = "lora-llama-3-youko-8b-results-pt"
OUTPUT_DIR = "lora-llama-3-youko-8b-instruct-results-pt-test2"

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

zero3_config_param = zero3_config(2, 4)
zero2_config_param = zero2_config(2, 4)

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

	# device_map = {
	# 	"model.embed_tokens": 0,
	# }
	# for i in range(0, 32, 1):
	# 	layers_gpu_num = i % 4
	# 	device_map[f"model.layers.{i}"] = layers_gpu_num
	# device_map["model.norm"] = 1
	# device_map["model.rotary_emb"] = 2
	# device_map["lm_head"] = 3

	# local_model_path = snapshot_download(MODEL_NAME)
	# with init_empty_weights():
	# 	model = AutoModelForCausalLM.from_pretrained(
	# 		MODEL_NAME,
	# 		torch_dtype=torch.bfloat16,
	# 		low_cpu_mem_usage=True,
	# 		device_map=device_map,
	# 	)

	# model = load_checkpoint_and_dispatch(
	# 	model,
	# 	local_model_path,
	# )
	model = AutoModelForCausalLM.from_pretrained(
		MODEL_NAME,
		torch_dtype=torch.bfloat16,
		low_cpu_mem_usage=True,
		device_map="auto",
	)
	# model = prepare_model_for_kbit_training(model)

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

	# 大きなモデルのパラメータを効率的に分散初期化
	with deepspeed.zero.Init(config=zero3_config_param, remote_device=None, pin_memory=True):

		# データセットのインスタンス作成
		loadDataset = load_dataset(DATASET_NAME)

		# print(loadDataset["train"][0])
		dataset = SimpleDataset(tokenizer, loadDataset["train"].select(range(1000)))
		# dataloader = DataLoader(dataset)

		# DeepSpeedの初期化
		model_engine, optimizer, data_loader, _ = deepspeed.initialize(
			model = model,
			config_params = zero3_config_param,
			training_data = dataset  # Datasetを渡す
		)

	# トレーニングループ
	epochs = 1
	for epoch in range(epochs):
		for step, batch in enumerate(data_loader):
		# for step, batch in enumerate(dataloader):
			# my_logging.debug(f"Step: {step}, batch_size: {batch[0].shape[0]}")  # 例として input_ids の先頭次元を出す
			print(f"Step: {step}, batch_size: {batch[0].shape[0]}")

			# バッチデータを分割 (例: 特徴量とラベル)
			input_ids, attention_mask, labels = batch
			input_ids, attention_mask, labels = (
				input_ids.cuda(),
				attention_mask.cuda(),
				labels.cuda(),
			)  # GPUに転送

			# フォワードパス
			outputs = model_engine(
				input_ids=input_ids,
				attention_mask=attention_mask,
				labels=labels  # ラベルを渡すと自動的に損失も計算される
			)
			# logits = outputs.logits  # テンソルを取り出す
			# loss = torch.nn.functional.cross_entropy(outputs, labels)
			# これでもいい？
			# loss = torch.nn.functional.cross_entropy(logits, labels)
			# loss = torch.nn.functional.cross_entropy(
			# 	logits.view(-1, logits.size(-1)),  # [バッチサイズ, シーケンス長, ボキャブラリ数] → [-1, ボキャブラリ数]
			# 	labels.view(-1)  # ラベルと次元を合わせる
			# )
			loss = outputs.loss

			# バックプロパゲーション
			model_engine.backward(loss)

			# パラメータ更新
			model_engine.step()

			# ログ出力
			if step % 10 == 0:
				print(f"Epoch [{epoch+1}/{epochs}], Step [{step}], Loss: {loss.item():.4f}")
				torch.cuda.empty_cache()

	model_engine.save_pretrained(OUTPUT_DIR)

except Exception as e:
	print("エラーが出た！！！！")
	# my_logging.error(e.)
	with open("log/error.log", 'w') as f:
		traceback.print_exc(file=f)
	raise e
finally:
	if dist.is_initialized():
		dist.destroy_process_group()
