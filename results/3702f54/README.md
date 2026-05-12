# Pre-go motor mask follow-up matrix (3702f54)

This is the pre-go motor mask follow-up to the f47abb1 lit-replication 6-cell matrix.
The f47abb1 results identified `lit__post_nojerk` and `lit__full_nojerk` as the leading
no-jerk powerlaw configurations, but both retained a residual ~2–3 mm of pre-go
anticipation (forward drift in the [−200 ms, 0 ms] hold window). This matrix tests two
complementary suppression strategies: (i) scaling the position-error weight 10× on the
hold/running terms (`__pos10` cells) to make whatever hold-period penalty exists more
biting, and (ii) re-introducing the `--nn-output-pre-go` motor mask lever — strategy 1
from `5acdaae` — at three weights spanning ~3 orders of magnitude. The 6 pre-go-output
cells cross `{1e-3, 5e-2, 1.0}` with `{pos×1, pos×10}` on the `full_trial_pl`
configuration, which has effectively-zero hold-period position penalty after the
powerlaw schedule and is therefore the cleanest test bed for an independent pre-go
output term. `lit__post_nojerk` and `lit__full_nojerk` from f47abb1 are NOT retrained
here; they are included in all comparison plots as the baseline anchors this matrix is
trying to improve upon.
