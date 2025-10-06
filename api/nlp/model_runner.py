# runners/llama_generic.py
from typing import Callable, Dict, Any, Optional
import pandas as pd
from transformers import PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig
from sentence_transformers import SentenceTransformer, util
import torch


class PromptTask:
    def __init__(
        self,
        name: str,
        # (fewshots_df, note_text) -> full prompt
        build_prompt: Callable[[pd.DataFrame, str], str],
        k_fewshots: int,
    ):
        self.name = name
        self.build_prompt = build_prompt
        self.k = k_fewshots


class LlamaPromptRunner:
    """Generic, prompt-driven runner: the only thing that varies per task is the prompt."""

    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        # pool for few-shot (already filtered)
        train_df: pd.DataFrame,
        embedder: SentenceTransformer,            # "all-MiniLM-L6-v2" as in your scripts
        task: PromptTask,
        device: str = "cuda",
        max_ctx: int = 100000,
        # temp, top_p, repetition_penalty...
        gen_kwargs: Dict[str, Any] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.embedder = embedder
        self.task = task
        self.device = device
        self.max_ctx = max_ctx
        self.gen_kwargs = gen_kwargs or dict(
            max_new_tokens=100, use_cache=True, temperature=1.5, top_p=0.1, repetition_penalty=1.1)

    def _fewshots(self, note_text: str) -> pd.DataFrame:
        # same retrieval you already use everywhere
        input_emb = self.embedder.encode([note_text])[0]
        sims = []
        for idx, row in self.train_df.iterrows():
            tr = row["note_original_text"]
            if tr.strip() == note_text.strip():
                continue
            tr_emb = self.embedder.encode([tr])[0]
            sims.append((util.cos_sim(input_emb, tr_emb).item(), idx))
        sims.sort(reverse=True, key=lambda x: x[0])
        idxs = [i for _, i in sims[: self.task.k]]
        return self.train_df.loc[idxs]

    def predict_one(self, note_text: str) -> Dict[str, Any]:
        few_df = self._fewshots(note_text)
        prompt = self.task.build_prompt(few_df, note_text)

        inputs = self.tokenizer(
            [prompt], truncation=True, max_length=self.max_ctx - 100, return_tensors="pt").to(self.device)
        out_ids = self.model.generate(**inputs, **self.gen_kwargs)
        full = self.tokenizer.batch_decode(out_ids)[0]

        # All your scripts extract the first line after "### Response:" and strip the leading "Annotation: "
        try:
            line = full.split("### Response:\n", 1)[1].split("\n", 1)[0]
        except Exception:
            # fallback if newline is missing in some generations
            line = full.split("### Response:", 1)[-1].strip().split("\n")[0]
        normalized = line.replace("Annotation:", "").strip()

        return {"task": self.task.name, "raw": full, "normalized": normalized}


# main
if __name__ == "__main__":
    # Example usage (you need to define model, tokenizer, train_df, embedder, and task)
    # model = FastLanguageModel("path_to_model")
    # tokenizer = PreTrainedTokenizer.from_pretrained("path_to_tokenizer")
    # train_df = pd.read_csv("path_to_fewshot_data.csv")
    # embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # def build_prompt(fewshots_df: pd.DataFrame, note_text: str) -> str:
    #     # Implement your prompt building logic here
    #     prompt = "### Few-shot examples:\n"
    #     for _, row in fewshots_df.iterrows():
    #         prompt += f"Input: {row['note_original_text']}\nOutput: {row['annotation']}\n\n"
    #     prompt += f"### New input:\nInput: {note_text}\nOutput:"
    #     return prompt

    # task = PromptTask(name="ExampleTask",
    #                   build_prompt=build_prompt, k_fewshots=3)

    # runner = LlamaPromptRunner(model, tokenizer, train_df, embedder, task)

    # note = "Your input text here."
    # result = runner.predict_one(note)
    # print(result)
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"  # non-AWQ repo
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.bfloat16, quantization_config=bnb
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that responds as a pirate."},
        {"role": "user", "content": "What's Deep Learning?"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, do_sample=True, max_new_tokens=256)

    print(tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0])
