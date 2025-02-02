from transformers import AutoModelForCausalLM

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# モデルの準備
# MODEL_NAME = "rinna/japanese-gpt-neox-3.6b"  # 適切なモデル名を指定してください
MODEL_NAME = "rinna/llama-3-youko-8b-instruct"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")

def get_module_string(name):
	name_split = name.split(".")
	return ".".join(name_split[(name_split.index("layers") + 2):])

def get_model_named_module(model_named_modules):
	name_modules = []
	for name, module in model_named_modules:
		if name == "":
			continue
		name_split = name.split(".")
		if len(name_split) < 2 or name_split[1] != "layers":
			continue
		name_modules.append((name, module))
	return name_modules

target_name_strings = []

for name, module in model.named_modules():
	print(name)

# # モジュール名の確認
# for name, module in get_model_named_module(model.named_modules()):
# 	target_name_strings.append(get_module_string(name))
# 	print(name)

# print(MODEL_NAME)
# for name in set(target_name_strings):
# 	print(name)
