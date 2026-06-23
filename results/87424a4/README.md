# Steady-State Feedback Perturbation Bank

Issue `87424a4` materializes local steady-state feedback-response probes for
C&S GRU rows without rerunning the reach-context perturbation bank. The bank
holds the plant at the target, uses a deterministic zero-noise wash-in prefix,
then applies five-step feedback offset pulses for position, velocity, and
force/filter channels where the model feedback contract exposes them.
