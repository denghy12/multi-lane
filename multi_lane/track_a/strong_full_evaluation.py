"""Export fused and view predictions from one identical evaluation forward."""
from pathlib import Path
import numpy as np
from .runner import TaskMetrics, evaluate_view_diagnostics


def evaluate_fused_and_views(model, loader, device, task_id, threshold, amp, scores, view_scores):
    diagnostics=evaluate_view_diagnostics(model,loader,device,task_id,threshold,amp,score_output_path=view_scores)
    with np.load(view_scores) as saved:
        arrays={key:saved[key].copy() for key in (
            'schema_version','task_id','sample_ids','class_indices','targets','batch_lengths',
            'probability_device','probability_dtype','probability_operation','torch_version')}
        arrays['logits']=saved['fused_logits'].copy()
        arrays['probabilities']=saved['fused_probabilities'].copy()
    scores=Path(scores);scores.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(scores,**arrays)
    return TaskMetrics(**diagnostics['metrics']['fused']),diagnostics
