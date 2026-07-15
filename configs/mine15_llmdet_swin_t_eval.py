_base_ = './mine15_llmdet_swin_t_finetune.py'

# Detection-only eval config. This avoids loading the LMM branch for bbox inference.
model = dict(
    lmm=None,
    test_cfg=dict(max_per_img=300, chunked_size=-1))

# Keep training validation on COCO bbox AP; add cgF1 for this standalone eval.
val_evaluator = dict(
    metric=['bbox', 'cgf1'],
    cgf1_score_threshold=0.5,
    cgf1_use_cats=True)
test_evaluator = val_evaluator
