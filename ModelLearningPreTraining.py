###############################################
# データの前処理
###############################################
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("rinna/llama-3-youko-8b-instruct")
# tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt-neox-3.6b")

batchs_sentences = [
	"魔法使いのことに干渉するな。彼らは微妙で怒りっぽい。",
	"怒りが限界を超えた！悟空が超サイヤ人になる",
]

# ①基本
encoded_input = tokenizer(batchs_sentences)
# ②最大に合わせたシーケンスの長さに合わせてパディング
# encoded_input = tokenizer(batchs_sentences, padding=True)
# ③モデルが受け入れる最大のシーケンス長さに合わせて切る。
# encoded_input = tokenizer(batchs_sentences, padding=True, truncation=True)
# トークナイザがモデルに供給される実際のテンソルを返すように設定 pt（PyTorch用）,tf（TensorFlow用）
# encoded_input = tokenizer(batchs_sentences, padding=True, truncation=True, return_tensors="pt")

print(encoded_input.keys)
print("")
print(encoded_input.input_ids[0])
print(encoded_input.input_ids[1])
print("")
print(encoded_input.attention_mask[0])
print(encoded_input.attention_mask[1])
print("\n")

for input_id in encoded_input["input_ids"]:
	decode_value = tokenizer.decode(input_id)
	print(decode_value)

# トークナイザーのスペシャルトークンの確認。
print(tokenizer.special_tokens_map)
print("bos_token :", tokenizer.bos_token, ",", tokenizer.bos_token_id)
print("eos_token :", tokenizer.eos_token, ",", tokenizer.eos_token_id)
print("unk_token :", tokenizer.unk_token, ",", tokenizer.unk_token_id)
print("pad_token :", tokenizer.pad_token, ",", tokenizer.pad_token_id)

# トークナイズ
def tokenize(prompt, tokenizer):
	# コンテキスト長
	CUTOFF_LEN = 256
	return tokenizer(
		prompt,
		truncation=True,
		max_length=CUTOFF_LEN,
		padding=False,
		# pad_token=tokenizer.eos_token,
	)

# トークナイズの動作確認
tokenize_temp = tokenize("hi there", tokenizer)
print(tokenize_temp)

print("--------------chat_template--------------")
messages = [
	{"role": "system", "content": "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。"},
	{"role": "user", "content": "GENIACプロジェクトの一つである東京大学松尾研究室のLLM開発プロジェクトについて説明してください。"},
	{"role": "assistant", "content": "GENIACとは経済産業省による日本の生成AI開発力向上のための取り組みです。"},
]

tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
print(tokenizer(
	tokenizer.apply_chat_template(messages, tokenize=False),
	return_tensors="pt",
	padding="max_length",
	truncation=True,
	max_length=256
))
print(tokenizer.apply_chat_template(messages, tokenize=False))
print(tokenizer.eos_token)