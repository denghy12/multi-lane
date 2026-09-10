# Hierarchical class-aware OOF stacking

## Goal

Test whether the three-view OOF evidence becomes useful when view weights may differ by emotion
class, without retraining Full, Person, or Face experts and without accessing test.

## Fixed inputs

- Existing seed0 three-fold image-group OOF Full/Person/Face probabilities.
- Existing complete-train and validation R2/R3 descriptors.
- Existing 100%-train Full/Person/Face validation endpoints.
- Reliable Face anchor `[0.64, 0.16, 0.20]`; invalid Face anchor `[0.80, 0.20, 0]`.
- Every class uses the stacking snapshot saved when its incremental task is introduced.

## Methods

- `C0`: fixed R1.
- `C1`: one centered three-view logit bias per class. The bias is trained only on OOF predictions
  for the class's introduction task and is regularized to zero, which is exactly C0.
- `C2`: freezes the selected C1 bias and adds a rank-2 interaction between shared sample reliability
  features and class-specific view factors. It has no class-specific MLP. Old class factors remain
  frozen; a class is evaluated with the shared projection snapshot saved at its introduction task.

C1 bias prior strengths and C2 interaction prior strengths are each limited to `{1, 3, 10}`.
Both methods use FP32, Adam with LR `1e-2`, batch size 512, and 120 epochs/task. C2 uses the same
quality, confidence, entropy, prediction-disagreement, and frozen CLIP descriptor features as OOF-R3.
Invalid Face weight is exactly zero.

## Selection rule

Run seed0 complete eight-task validation only. The reduced continuation margin is `0.05` final mAP:

1. best C2 must exceed both C0 and best C1 by at least `0.05` final validation mAP;
2. at least two classes must improve by more than `0.01` AP versus C0;
3. no single class may account for more than 80% of all positive class AP gain.

Only if all conditions pass may seed1/2 validation be prepared. This stage never launches test.
