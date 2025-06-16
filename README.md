# EchoPersona

You’ve chatted with a lot of LLMs. They’re polite. Helpful. Predictable.

EchoPersona isn’t like them.

This is a fine-tuned conversational model that simulates a consistent human-like tone. It's not trained to be universally useful — it’s trained to sound like a specific kind of person: sarcastic, observant, and occasionally too honest.

---

## What is it?

- Fine-tuned version of `Qwen1.5-4B` using `LoRA + QLoRA` in 4-bit mode
- Runs on a single RTX 5080 (16GB VRAM)
- Custom instruction-style dataset with <100 samples
- Deployed as a REST API using FastAPI
- Designed to answer like a persona, not just complete a task

This isn't a chatbot clone. It's more like a character with opinions.

---

## Why?

Because I wanted to know:

> Can you fine-tune a large language model to sound emotionally consistent using minimal data and consumer hardware?

Turns out, yes — if you're intentional with your prompts, your data, and your model architecture.

---

## Sample Dialogues

**Q:** How can I be more productive?  
**A:** Start by closing whatever tab made you ask that.

**Q:** Do you believe in free will?  
**A:** That depends. Are you asking, or are you programmed to?

**Q:** Are you always this direct?  
**A:** Most people just never ask twice.

---

## Stack

- Model: `Qwen1.5-4B` (HuggingFace)
- Fine-tuning: `LoRA` + `QLoRA` using `peft`, `bitsandbytes`
- Deployment: `FastAPI` + `Uvicorn`
- Environment: Python 3.10, CUDA 12.8
- Hardware: Single GPU, RTX 5080 (16GB)

---

