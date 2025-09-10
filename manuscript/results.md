

- If we focus on a single type of perturbation outside the supplementary analysis, it should probably be SIRF rather than CF. 

## Part 1

### Response to mechanical perturbations

2x2 grid of center-out sets. 

#### Example center-out sets +/- SU

As on the NCM poster, plus PA zero. 

Small task variant.

#### Aligned center-out sets, by SU

Full task variant.

As on the NCM poster, plus PA zero.

#### Velocity profiles, by SU

As on the NCM poster, plus PA zero.

#### Quantification

Full task variant. 

Include the ones on the NCM poster, but maybe also:

- Tangling

### Supplementary

- (Maybe) CF versions of all of the above, if we leave them out of the main results. 
- Comparison of CF and SIRF robustness
	- i.e. what happens if we evaluate CF-trained on SIRF, or vice versa?
- Effect of noise on performance of trained models
- Effect of training on noise
- Frequency response 

## Part 2

### Response to mechanical perturbations

Same as in part 1, except comparisons are by SISU instead of by SU.

As on the NCM poster, plus PA zero, and velocity profiles. 

### Response to feedback perturbations

> At steady state (i.e. already at goal) apply a step-change to the velocity feedback, in a random direction, for 5 time steps. Align the response to impulse direction.

#### Velocity profiles

As on the NCM poster. 

#### Quantification

Max net control force, as on NCM poster. 

Also:

- Tangling, around the time of perturbation. 

#### Supplementary

- Input sensitivity: largest singular value of the Jacobian wrt inputs

### Fixed point analysis

#### PCs vs. SISU

As on NCM poster. **Was this with or without a mechanical perturbation?** We should probably show both. 

**Also**: along with the 3D plots as on the poster, could show 2D plot(s) along the SISU axis (i.e. showing 4 cardinal reaches), which could help to give an intuition for tangling?

#### Stability/contraction analysis

- Jacobian eigendecomposition at steady-state fixed points 
- Input sensitivity along state trajectory (not unsteady fixed points, I think)
- Hessian norm stuff (see TODO & ask o3 about it) - indicates the size of the neighbourhood in which the linearization is valid. Is this relevant?

### Unit stimulation

See [[Neural tuning#Context-dependence of unit stimulation effects]].

#### Visualization

##### Kinematic profiles

##### Hidden trajectories

Plot in PC space. I’m not sure what the ideal PCs are; we might ultimately want to project them in the same space as reaches, but for now the PCs can probably just be computed for the +/- unit stim trials themselves.

#### Modeling approaches


> [!NOTE]
> Make sure to:
> - center and scale the regressors and response variables! (mean subtract and divide by population std)
> - divide any euclidean norms of Jacobians etc. by the square root of their dimension (e.g. 2, for position channel) to make them comparable across analyses

##### Two-regression comparison

In each case, one regression per unit. 

**Regressors**: SISU and pert amp conditions
**First regression response vars**: unit stim responses; e.g. max deviation, max speed, etc. *following stimulation of the unit*
**Second regression response vars**: feedback gains; i.e. euclidean norm of respective unit’s component of the network’s steady state Jacobian wrt a feedback channel (position and velocity separately)

Thus we will be able to do multiple comparisons (be careful). For example, we could plot the coefficient predicting max deviation from SISU, against the coefficient predicting position feedback gain from SISU

##### Mixed model

See https://chatgpt.com/share/68b22c05-1178-8006-946f-efa44fd81e8e

This is more correct in general than the [[#Two-regression comparison]]. 

Essentially we do 

$$
Y_{i,s,p,r} \;=\; \underbrace{\alpha_0 + u_i}_{\text{baseline}} \;+\; \underbrace{\beta_G\,G_{i,s,p} + \beta_S\,S + \beta_P\,P + \beta_{SP}\,S P}_{\text{fixed effects}}\;+\; \underbrace{v_{iG}\,G_{i,s,p} + v_{iS}\,S + v_{iP}\,P}_{\text{group=-specific slopes (optional)}} \;+\; \varepsilon
$$
where 

- $Y_{i,s,p,r}$ is the response (e.g. max deviation) after stimulation of unit $i$, with SISU $S=s$, pert.amp. $P=p$, for experimental replicate $r$.
- $G_{i,s,p}$ is the feedback gain computed from the feedback channel Jacobian for the same unit, at steady state, in the same condition; note that we could include a second such term to include both feedback channels in the regression.
- $v_{iG}$ and other such terms are unit-specific coefficients; these are optional but if we include them we can later extract best/posterior values for the mean unitwise coefficients, which we can use for similar pairwise analyses/clustering as described in [[#Two-regression comparison]].
- *in any case* we should include the unit-specific offset $u_i$.

Note that we could include other interactions than $SP$; in particular the interactions between the gains $G$ and $S,P$ might be meaningful. 

A mixed model like this can be fit by either frequentist or Bayesian methods. 

## Supplementary: Adaptation versus robustness

There is already a bit of evidence for robustness (versus adaptiveness) in that the part 2 networks benefit from being given the SISU. If they were adaptive they’d just estimate the field parameter in the first 5 timesteps or so (no delay!) and then cancel it out more effectively, regardless of the SISU.

### Direction-reversal probe

Start evaluating on a field in one direction and then flip it midway through the trial. The resulting error should be larger if we’re adaptive than if we’re robust. 

Be careful. How can we set this up so that we know that the perturbation will result in a certain magnitude of deviation, depending on robust vs. adaptive?

### Regress late control forces against early perturbation parameter values 

If we’re simply robust, then control forces will be correlated with the SISU but not with the actual value of the field parameter. 

If we’re adaptive, we’ll change our control forces based on our estimate of the actual field parameter.