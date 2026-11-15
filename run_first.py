"""第一次运行：让 nano-vllm 回答一个问题。

在 Linux GPU 服务器安装依赖、下载模型后，从仓库目录执行：
    python run_first.py

下面的模型目录是服务器上的路径。脚本不负责安装依赖或下载模型。
"""

import os


# 第一次运行只需要确认这四项；其余配置保持下面的初始值。
MODEL_DIR = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
QUESTION = "请用一句话介绍你自己。"
MAX_NEW_TOKENS = 64  # 最多生成多少个新 token，不是字数。
TEMPERATURE = 0.6   # 采样随机程度；当前上游不支持设为 0。


def main():
    if not os.path.isfile(os.path.join(MODEL_DIR, "config.json")):
        raise FileNotFoundError(
            f"模型目录还没有准备好：{MODEL_DIR}\n"
            "请先下载完整的 Qwen3-0.6B 模型，或把 MODEL_DIR 改成实际下载目录。"
        )

    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("[1/3] 正在加载模型，首次运行可能需要编译，请稍等。", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": QUESTION}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if MAX_NEW_TOKENS <= 0 or TEMPERATURE <= 1e-10:
        raise ValueError("MAX_NEW_TOKENS 必须为正整数，TEMPERATURE 必须大于 1e-10。")
    if len(tokenizer.encode(prompt)) + MAX_NEW_TOKENS > 2048:
        raise ValueError("这份入门脚本的上下文上限为 2048 token，请缩短问题或输出长度。")

    # 小规模单卡配置：先跑通，后面再逐项学习这些引擎参数。
    llm = LLM(
        MODEL_DIR,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        max_num_batched_tokens=512,
        max_num_seqs=16,
        gpu_memory_utilization=0.60,
    )

    print("[2/3] 正在生成回答。", flush=True)
    print(f"问题：{QUESTION}", flush=True)
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS),
        use_tqdm=False,
    )
    if len(outputs) != 1 or not outputs[0]["token_ids"]:
        raise RuntimeError("没有获得预期的生成结果，请保留完整输出以便排查。")

    print("[3/3] 已生成结果。", flush=True)
    print(f"回答：{outputs[0]['text']}", flush=True)
    print(f"生成 token 数：{len(outputs[0]['token_ids'])}", flush=True)


if __name__ == "__main__":
    main()
