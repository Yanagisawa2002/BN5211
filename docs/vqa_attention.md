# VQA Attention Heatmaps

This document explains the utility command used for quick smoke tests and how to visualize the regions of an image that drive the model's visual question answering (VQA) responses.

## Bytecode compilation check

The project README recommends running:

```bash
python -m compileall \
    sam2/modeling/cross_modal_fusion.py \
    sam2/modeling/sam2_base.py \
    sam2/sam2_video_predictor.py \
    training/model/sam2.py \
    tools/rvos_inference.py
```

[`compileall`](https://docs.python.org/3/library/compileall.html) walks over the listed Python modules, parsing them and emitting bytecode. If the command succeeds silently, it confirms that the edited files are free of syntax errors. This makes it a lightweight sanity check before kicking off heavier integration tests.

## Producing attention maps during VQA

During inference the cross-modal fusion module projects the fused video tokens and the text query into a shared embedding space. The classifier token that summarizes the query is compared against the current frame features to compute a normalized attention map:

1. `CrossModalFusionModule.forward` extracts the per-frame tokens and the classifier vector representing the query (`cls_tokens`).
2. It multiplies (`einsum`) the classifier vector with the spatial features of the current frame to obtain a coarse attention map over the low-resolution feature grid.
3. The heatmap is min-max normalized so that every frame’s scores lie in `[0, 1]`.

The resulting tensor is returned to the higher-level SAM2 wrapper:

```python
fusion_image_embeddings, text_cls_tokens, attention_map = self.cross_modal_fusion(...)
current_out["vqa_attention_map"] = attention_map
```

These maps are propagated by the video predictor. When `propagate_in_video` yields frame-level results, it also collects the corresponding attention maps (interpolated back to the image resolution) so downstream consumers or tools can consume them alongside the segmentation masks.

## Visualizing the focus regions

To visualize the focus area for each question–answer pair:

1. Run VQA inference with `tools/rvos_inference.py`. Pass `--save-attention-overlays` and provide an output directory.
2. For every frame/object combination, the script reads the attention map, upsamples it to the image size, and saves an overlay that blends the heatmap with the RGB frame.

The helper `save_attention_overlay` performs the actual rendering: it normalizes the map, converts it to a grayscale image, applies a colormap, and composites it with the frame using an alpha blend. The resulting PNG highlights the pixels that most influenced the answer, making it easy to inspect the model’s focus region for each prediction.
