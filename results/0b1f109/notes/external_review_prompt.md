# Prompt Statement for External Review

You will be given three main materials:

1. `external_review_packet.md`, a methodological brief summarizing the current
   rlrmp plan for connecting cs2019's induced-robustness finding to trained
   recurrent neural controllers.
2. `/Users/mll/Documents/Claude/Projects/clench/synthesis-5.md`, a broader
   theoretical and motivational backdrop for the project.
3. `/Users/mll/Main/10 Projects/10 PhD/cs2019.pdf`, the motor-control paper most
   central to the framing.

You may also inspect the rlrmp and feedbax source code if you think it is necessary,
but the main requested review is conceptual, formal, and methodological rather
than an implementation audit.

Please treat the review packet's questions as tentative guideposts, not as a
complete or privileged checklist. If you think the packet asks the wrong questions,
please say so and reframe the problem in the way you think is most scientifically
or mathematically useful.

Primary request:

- Review the plan summarized in `external_review_packet.md`.
- Identify the strongest version of the plan.
- Identify the most important formal, methodological, or interpretive weaknesses.
- Suggest changes that would make the experimental logic cleaner or more decisive.
- Be explicit about what would falsify the current framing, rather than merely
  require an implementation fix.

Treatment of `synthesis-5.md`:

- Treat it as background orientation, not as part of the concrete experimental
  plan.
- Please separately flag any mathematical errors, conceptual confusions, or
  overclaims in `synthesis-5.md`.
- If anything in `synthesis-5.md` is cross-cutting with the methodological plan,
  please surface that connection in the main review.

Useful response structure, if it fits your analysis:

1. Overall verdict on the plan.
2. Best-case interpretation of what the plan is trying to test.
3. Major formal or methodological risks.
4. Whether the cs2019-to-RNN mapping is well-posed.
5. Whether the restricted-field versus broad-epsilon contrast is the right
   comparison.
6. Whether the linear same-game gate is sufficient and properly specified.
7. Suggested modifications or alternative decisive experiments.
8. Separate notes on `synthesis-5.md`.
9. Concrete questions the authors should answer before implementation.

Do not restrict yourself to this structure if another organization would be more
useful.
