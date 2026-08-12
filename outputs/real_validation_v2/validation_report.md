# Real-scene ROI Validation Cleanup

No completion benchmark was rerun and no completion algorithm was changed.

## ROI validity

- **Planar replacement: NO VALID CANDIDATE.** After a deterministic scan of 24,000
  PCA-prioritized anchors, none passed every hard check. The closest candidate has 160
  Gaussians and acceptable density ratio (0.985), but fails normal spread (P95 39.2 deg
  > 15 deg), semantic purity (0.719 < 0.8), and connectivity (largest component 0.125).
  It is retained only as the top rejected candidate for visualization, not as a selected
  planar experiment.
- **Curved replacement: NUMERICALLY VALID, VISUAL REVIEW ADVISED.** `roi_D_curved_v2`
  has 160 Gaussians, spacing ratio 0.826, curvature 0.0408, two semantic IDs with 0.90
  dominant fraction, and largest component fraction 0.85. Its PCA-normal plot remains
  visibly noisy, so it should not yet be described as a clean smooth object without
  manual confirmation.
- **Junction: coordinates frozen exactly from v1.** Its normal P95 is 35.4 deg, but the
  local graph is fragmented (8 components; largest fraction 0.396). The normal plot
  supports orientation variation, not a strong claim of exactly two clean coherent
  surfaces.
- **Layered: coordinates frozen exactly from v1.** It contains IDs 5/7 (5 and 64
  Gaussians), but is also fragmented (18 components; largest fraction 0.246) and has a
  density ratio of 0.239. Keep it as the fixed reference requested, with these caveats.

## Overspawn audit

The v1 junction spawned 1,024 versus 53 removed Gaussians (19.3x); the old curved ROI
spawned 9,025 versus 53 (170.3x). Planar and layered were near 1.1x. The likely cause is
that spawning uses an axis-aligned grid over the configured hole extent, so a large or
sparse ROI produces a dense grid unrelated to observed boundary density. A diagnostic,
disabled estimate based on boundary density x projected missing area predicts about 29
for junction and 11 for old curved. GT counts were not used in those predictions.

## Confidence audit

Confidence is not calibrated. For C3-adaptive, junction has geometry confidence 0.940
but support 0.086 and semantic 0.152, collapsing the product to 0.0119. Old curved has
geometry 0.923, support 0.033 and semantic 0.173, producing 0.00534. Thus the
multiplicative formula can be dominated by one weak term. Historical per-term values
were only persisted for C3-adaptive; missing method terms are explicitly N/A.
