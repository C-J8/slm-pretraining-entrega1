import sys
from pathlib import Path

import torch
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import LlamaConfig, LlamaModel
from tokenizer import GPT2Tokenizer
from utils import get_device


def format_chat(messages: list[dict[str, str]]) -> str:
    text = ""
    for message in messages:
        role = "User" if message["role"] == "user" else "Assistant"
        text += f"{role}:\n{message['content']}\n"
    text += "Assistant:\n"
    return text


@st.cache_resource
def load_chat_model(checkpoint_path: str):
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_cfg = LlamaConfig.from_dict(checkpoint["config"]["model"])
    model = LlamaModel(model_cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    tokenizer = GPT2Tokenizer()
    return model, tokenizer, device


def generate_reply(
    checkpoint_path: str,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    model, tokenizer, device = load_chat_model(checkpoint_path)
    prompt = format_chat(messages)
    input_tokens = tokenizer.encode(prompt)
    idx = torch.tensor([input_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    decoded = tokenizer.decode(out[0].tolist())
    reply = decoded[len(prompt) :]
    stop_markers = [
        "<|endoftext|>",
        "\nUser:",
        "\nAssistant:",
        "User:",
        "Assistant:",
    ]
    for stop in stop_markers:
        if stop in reply:
            reply = reply.split(stop, 1)[0]
    return reply.strip()


st.set_page_config(page_title="SLM Chat")
st.title("SLM Chat")

with st.sidebar:
    checkpoint_path = st.text_input("Checkpoint", value="checkpoints/sft_best.pt")
    max_new_tokens = st.slider("Max new tokens", 16, 512, 128, step=16)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.8, step=0.05)
    top_k = st.slider("Top-k", 0, 200, 50, step=5)
    if st.button("Reset chat"):
        st.session_state.messages = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if not Path(checkpoint_path).exists():
    st.warning(f"Checkpoint not found: {checkpoint_path}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Digite uma mensagem"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if Path(checkpoint_path).exists():
        with st.chat_message("assistant"):
            with st.spinner("Gerando resposta..."):
                reply = generate_reply(
                    checkpoint_path,
                    st.session_state.messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                st.write(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})
