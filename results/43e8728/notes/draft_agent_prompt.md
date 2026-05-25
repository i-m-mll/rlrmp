You are drafting an updated central plan for the rlrmp cs2019-to-RNN robustness
programme.

Primary rule: base the plan first on issue `0b1f109` and especially the GPT-5.5
Pro review at `results/0b1f109/notes/gpt-5.5-pro_review.md`. All older issues,
including `35f64be` and `6f783fa`, are subordinate source material. Do not let an
older plan override the game-equivalence framing from the GPT-5.5 Pro review.

Repository and context:

- Repo: `/Users/mll/Main/10 Projects/10 PhD/rlrmp`
- New plan umbrella issue: `43e8728`
- Primary synthesis issue extract: `/tmp/rlrmp_plan_issue_wrangling/issues/0b1f109.txt`
- GPT-5.5 Pro review: `results/0b1f109/notes/gpt-5.5-pro_review.md`
- Review packet: `results/0b1f109/notes/external_review_packet.md`
- Prior v1 plan: `/tmp/rlrmp_plan_issue_wrangling/issues/35f64be.txt`
- Prior v2 plan: `/tmp/rlrmp_plan_issue_wrangling/issues/6f783fa.txt`
- Other extracted issues: `/tmp/rlrmp_plan_issue_wrangling/issues/*.txt`
- Full issue list snapshot: `/tmp/rlrmp_plan_issue_wrangling/issue_list_all.txt`

Issue-system instructions:

- Prefer the extracted issue files above for reading.
- If live issue inspection is needed, use `mandible issue show <id>` from the
  rlrmp repo. Do not add comments, labels, close, reopen, or create issues.
- Do not commit or edit files unless explicitly instructed by the caller.

What the plan must do:

1. Treat the first major experiment as a game-equivalence programme, not a simple
   behavioral reproduction experiment.
2. Require an analytical "game card" before training: state vector, delay
   augmentation, discretization, costs, control scaling, disturbance channel
   `B_w`, epsilon norm and time integration, gamma/gamma-star convention, LQR and
   H-infinity gains, nominal trajectories, analytical worst-case disturbance or
   equivalent closed-loop induced-gain characterization, induced gain, and Delta-v.
3. Treat open-loop trajectory epsilon versus state-dependent H-infinity adversary
   as a load-bearing formal uncertainty. The plan should either split these checks
   or explicitly pause for the user to decide after smart-model review.
4. Make future decision points explicit. When a significant unresolved choice
   becomes relevant, the implementing agent must stop, remind the user, ask for a
   decision/consent before proceeding, and suggest getting fresh feedback from a
   strong model when appropriate.
5. Wrap subordinate issues as subordinate components only. For example, cost
   schedule, perturbation amplitude, analysis pipeline, and decoupling/decomposition
   are more specific investigations into parts of the plan.
6. Do not reintroduce discarded decisions from `35f64be` or `6f783fa`. Carry forward
   only the pieces that remain valid under `0b1f109` plus the GPT-5.5 Pro review.
7. Include a section for supplementary analyses, one issue per line, each with a
   one- or two-sentence description of relevance and where it might fit.
8. Include a separate section for ops/hygiene/dependencies, one issue per line,
   each with a one- or two-sentence description of relevance.
9. Include recommended issue-disposition notes: which issues are superseded or
   historical, which should remain open as children, which should be commented on
   after the final plan is adopted.

Known issue-classification inputs from preliminary wrangling:

- Core carry-forward or close to core: `020a65b`, `6ec6b19`, `daa48c8`,
  `1ad3c16`, `63cec06`, `8fcb6c7`, `cf56e1e`, `f7b1b17`, `b41c940`.
- Later/supplementary: `ac06736`, `a5e1450`, `31043a5`, `a3edc0c`,
  `6d62018`, `65156e8`, `297260c`, `0af472c`.
- Artifact/publication hygiene: `fdad09d`, `e75ddd7`, `2ef67ca`, `2092cb5`,
  `6d5c906`.
- Ops support: `76d3a8e`, `216b368`, `a8ed10f`, `3bd407b`, `f7d40f1`,
  `f350f58`.
- Likely superseded/historical as central plan drivers: `35f64be`, `6f783fa`,
  `753508c`, `84ee4ff`, `ce34c2c`, `83fc5b5`.

Write a markdown plan draft suitable for synthesis into the final issue content.
Be concrete, but do not overfit to older issue details when the GPT-5.5 Pro review
points elsewhere.
