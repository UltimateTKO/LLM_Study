def zero3_config(batch_size_per_gpu, gpus):
	return {
		"bf16": {
			"enabled": True,
		},
		"zero_optimization": {
			"stage": 3,
			# "offload_param": {
			# 	"device": "cpu",
			# 	"pin_memory": True,
			# },
			# "offload_optimizer": {
			# 	"device": "cpu",
			# 	"pin_memory": True,
			# },
		},
		"optimizer": {
			"type": "Adam",
			"params": {
				"lr": 0.001
			}
		},
		# "gradient_clipping": "auto",
		"train_batch_size": batch_size_per_gpu * gpus,
		"train_micro_batch_size_per_gpu": batch_size_per_gpu,
		# "gradient_accumulation_steps": 4,  # 4ステップ蓄積
		"steps_per_print": 10,
		"flops_profiler": {
			"enabled": False,  # 修正
			"profile_step": 1
		},
		"wall_clock_breakdown": False,
	}

def zero2_config(batch_size_per_gpu, gpus):
	return {
		"bf16": {
			"enabled": True,
		},
		"zero_optimization": {
			"stage": 2,
			"offload_param": {
				"device": "cpu",
				"pin_memory": True,
			},
			# "offload_optimizer": {
			# 	"device": "cpu",
			# 	"pin_memory": True,
			# },
		},
		"optimizer": {
			"type": "Adam",
			"params": {
				"lr": 0.001
			}
		},
		# "gradient_clipping": "auto",
		"train_batch_size": batch_size_per_gpu * gpus,
		"train_micro_batch_size_per_gpu": batch_size_per_gpu,
		# "gradient_accumulation_steps": 4,  # 4ステップ蓄積
		"steps_per_print": 10,
		"flops_profiler": {
			"enabled": False,  # 修正
			"profile_step": 1
		},
		"wall_clock_breakdown": False,
	}