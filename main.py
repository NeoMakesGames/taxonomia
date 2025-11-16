import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

MODEL_ID = "zhihan1996/DNABERT-2-117M"


def disable_flash_attention(config):
	"""Force regular attention to avoid NVIDIA-only FlashAttention kernels."""
	flash_flags = (
		"use_flash_attn",
		"use_flash_attention",
		"flash_attn",
		"flash_attention",
		"use_flash_attn_triton",
	)
	for flag in flash_flags:
		if hasattr(config, flag):
			setattr(config, flag, False)

	preferred_impl = {
		"attn_implementation": "torch",
		"attention_implementation": "torch",
		"attention_impl": "torch",
		"attention_mode": "eager",
		"attn_type": "standard",
	}
	for attr, value in preferred_impl.items():
		if hasattr(config, attr):
			setattr(config, attr, value)

	# DNABERT-2 toggles FlashAttention whenever attention dropout is zero.
	dropout_attr = "attention_probs_dropout_prob"
	if getattr(config, dropout_attr, 0.0) == 0.0:
		setattr(config, dropout_attr, 1e-3)

	return config


def resolve_device():
	if torch.cuda.is_available():
		return torch.device("cuda")
	if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def main():
	device = resolve_device()

	config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
	config = disable_flash_attention(config)

	tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
	model = AutoModel.from_pretrained(
		MODEL_ID,
		trust_remote_code=True,
		config=config,
	).to(device)
	model.eval()

	dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
	inputs = tokenizer(dna, return_tensors="pt")["input_ids"].to(device)

	hidden_states = model(inputs)[0]
	embedding_mean = torch.mean(hidden_states[0], dim=0).detach().cpu()
	print(embedding_mean.shape)


if __name__ == "__main__":
	main()