# AGENTS.md — LLM 101 Slide Deck

## What Is This?

This repository contains the complete **LLM 101: Efficiency, Compression & Distillation** lecture by **Antreas Antoniou** (Principal Scientist & Founder, [Axiotic AI](https://axiotic.ai), University of Edinburgh).

It is a **single-file interactive slide deck** built as a self-contained HTML file (`index.html`) with 58 slides covering:

1. **LLM 101** — Transformer architecture, self-attention, positional embeddings, skip connections, FSDP, training at scale
2. **Compression & Distillation** — Pruning, quantisation, knowledge distillation (temperature trick, the conundrum, distillation theory), speculative decoding, FlashAttention
3. **Smarter, Not Bigger** — PEFT, LoRA, local deployment, Ollama, local agent architecture
4. **Resources** — Key papers, frameworks, the research frontier, open questions

---

## Files

| File | Description |
|------|-------------|
| `index.html` | Full interactive slide deck (58 slides, self-contained, all images embedded as base64) |
| `slides.md` | **Complete text content of all slides** as structured Markdown — best for agent ingestion, search, and RAG |
| `AGENTS.md` | This file |
| `README.md` | Human-facing overview |

---

## For AI Agents — How to Use This Content

### Quick ingestion
The fastest way to understand everything in this deck is to read `slides.md`. It contains the full text of all 58 slides, structured by section with slide numbers. No images, no base64 — clean text.

```
# Read slides.md for full content
cat slides.md
```

### Full deck (with images)
`index.html` is a self-contained HTML file (~5.5MB) with all diagrams, charts, and figures embedded. Open it in any browser — no server required. Navigate with arrow keys or the on-screen buttons.

### Key concepts to extract
If you're reading this to learn or to help someone learn, focus on these high-signal slides:

- **Slide 6** — The Transformer Architecture (the LEGO analogy)
- **Slide 7** — Self-Attention: Q/K/V, the exam hall analogy, why in-context learning works
- **Slide 8** — Positional Embeddings: RoPE and why it's the de facto standard
- **Slide 9** — Skip Connections: the identity shortcut, smooth loss landscapes
- **Slide ~18** — The Banana Slide: 20W brain vs 10,200W DGX H200, the 510:1 ratio
- **Slides 30–33** — Distillation: temperature trick, dark knowledge, the conundrum, why Path B wins
- **Slide ~34** — The Conundrum: Antreas's parallel search theory + 2026 validation (Invariant Algorithmic Cores, arXiv:2602.22600)
- **Slide ~45** — Speculative decoding: draft + verify, 3–5× speedup
- **Slide ~48** — Call to Action: Ollama, running state-of-the-art models locally

### Research papers cited
Key papers referenced throughout:
- **Attention Is All You Need** (Vaswani et al., 2017) — arXiv:1706.03762
- **Loss landscape smoothness** (Li et al., 2018) — arXiv:1712.09913
- **FlashAttention** (Dao et al., 2022) — arXiv:2205.14135
- **LoRA** (Hu et al., 2021) — arXiv:2106.09685
- **Invariant Algorithmic Cores** (2026) — arXiv:2602.22600 *(validates distillation theory)*
- **Residual Koopman Spectral Profiling** (2026) — arXiv:2602.22988 *(predicts training instability)*
- **Structure & Redundancy via RMT** (2026) — arXiv:2602.22345 *(spectral pruning > magnitude pruning)*

---

## Axiotic AI

**Axiotic AI** is building toward smarter learning, not just bigger models. We're not anti-scale — we just don't believe intelligence needs a skyscraper's worth of computers and a national power budget to run.

> *Language is a shadow of the world. LLMs are missing the ground truth. We're building towards that.*

- Website: [axiotic.ai](https://axiotic.ai)
- Founder: Antreas Antoniou — antreas@axiotic.ai · iam@antreas.io
- Twitter: [@AntreasAntoniou](https://x.com/AntreasAntoniou)

---

## Live Deck

The interactive deck is also hosted at:
- **[antreas.io/llm101-slides](https://antreas.io/llm101-slides)** (GitHub Pages)
