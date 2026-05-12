# Consultation: decoupling framing review (2026-05-11)

External reviewers asked: "is our framing right, is the MVP approach valid, what's the best path to demonstrate (or rule out) decoupling, and what alternative hypotheses should we consider?"

Reviewers:
- **Gemini 3.1 Pro** (`gemini-3.1-pro-preview`, headless YOLO)
- **GPT-5.5** via the codex CLI (`codex exec`, dangerously-bypass-approvals-and-sandbox)

Files:
- `prompt.md` — the consultation prompt sent to both agents (identical except for header)
- `gemini_response.md` — Gemini's independent analysis (~1k words)
- `codex_response.md` — Codex's independent analysis (~2k words, pushes harder on alternative-hypotheses framing)
- `synthesis.md` — orchestrator's synthesis: agreement, disagreement, recommended path forward, cross-refs to follow-up issues
- `codex_second_opinion.md` — Codex's critique of the synthesis (added if run)

Headline outcome: both reviewers diagnose the same MVP flaw (trivial x_nom → affine regulator-to-target masquerading as a tracker). They diverge on framing: Gemini accepts the regulator-vs-tracker framing as fundamentally correct; Codex argues objective mismatch and adversary mismatch are at least as plausible explanations for the GRU's missing Δv signature. The recommended path (Tier 1 → 2 → 3) is in `synthesis.md`.
