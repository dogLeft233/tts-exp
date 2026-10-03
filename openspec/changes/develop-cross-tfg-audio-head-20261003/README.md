# Cross-TFG natural-audio enhancement head

Status: waveform core implemented; real generation, original-audio mux, fresh scoring and a nonzero GPU optimizer update passed. Twenty fresh development clips are independently sealed. Candidate fitting and multi-model efficacy evaluation are pending.

Develop an external waveform transform `h(x)` that improves videos driven by natural speech while preserving the original natural audio for delivery and scoring. This is the new user-authorized research objective after the mechanism study in report 25.

- [Specification](spec.md): scope, stages, frozen controls, acceptance, continuation policy.
- [Tasks](tasks.md): implementation and experimental work.
- Project memory: `Experiments/跨 TFG 自然音频增强头的训练与原音换轨验证.md`.
- Active task: `codex/tts-exp/开发跨 TFG 的自然音频增强头.md`.
- Decision: `docs/adr/006-portable-waveform-enhancement-head.md`.

ROOT owns scientific decisions, protocol, Basic Memory and Git. Three gpt-6-luna/max subagents own waveform implementation/data, remote models/experiments, and independent evaluation. The prior completed task and report remain historical evidence.
