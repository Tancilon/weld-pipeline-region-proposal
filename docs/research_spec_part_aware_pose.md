# Part-Aware Pose Diffusion with Category-Specification Prompt for Industrial Welding

Research spec — **v0.7** — 2026-05-11 (Codex third-review round; v0.6 micro-patch — same day)

Codename: **FORGE 2.0** (Factorized Object-to-pRimitive Geometric Estimation, open-category edition).
Paper title candidate (primary): *CatSpec-Pose: Parametric Primitive Specification as Test-Time Category Prompt for Open-Category Part-Level 6D Pose Estimation*.
Working title (legacy): *FORGE: Part-Aware Pose Diffusion with Render-and-Compare on Online-Assembled Primitive Proxies*.

Target venue: CVPR 2027 / ICCV 2027 (primary), CoRL / RSS 2027 (backup).

Author: tancilon (GenPose2 / EoMT-dev branch)

---

## 0. What Changed

### v0.6 → v0.7 (2026-05-11, Codex third-review round — micro-patch)

Codex's third review: "**Go** — the architecture is sound and v0.6 is implementation-ready; do one round of small fixes first, then start P00." No architectural change. v0.7 closes the six text/protocol-consistency items Codex flagged plus two engineering suggestions:

1. **sub-mm requirement vs 1-3 mm world-accuracy claim conflict.** §1.1 said nuclear welding "requires sub-millimeter localization"; §1.5 admits the precision tier folds out to ~1-3 mm in robot/world coordinates. v0.7: §1.1 reworded to "mm-class localization (with sub-mm being the long-run target where the precision tier and calibration chain allow it)"; §1.5 keeps "local sensor coords sub-mm, robot/world 1-3 mm" and is now the single source of truth for the numbers.
2. **§2.2 still carried v0.5's single `τ≈2-3 mm`.** v0.7 splits it: `τ_coarse` (cm-scale — the triage threshold deciding *whether a seam needs a Tier-2 laser scan*) vs `τ_precision` (≈2-3 mm — the threshold deciding *whether a precision-tier result is executable*). The coarse gate is explicitly *not* an mm certification.
3. **R7 still described the old K=4-diffusion mitigation.** v0.7 rewrites R7 around K=1 flow matching + uncertainty-gated rerun {K=1,2,4} + the optional explicit refinement fallback — consistent with §3.6 / §3.8.
4. **P00 oracle was "non-learned least-squares" only, yet called "best possible".** A single least-squares fit can sit in a local optimum and *under*state the ceiling. v0.7 P00 runs the oracle in **three variants** — (a) GT-initialized local fit, (b) multi-start global fit, (c) a depth-noise Monte-Carlo / Cramér-Rao-style information-theoretic lower bound on centerline error — and the gate uses the *best* of the three. A bad result is only conclusive if even (b)/(c) say cm-level.
5. **measured-GT subset size was claimed as a benchmark-grade contribution but was only ≥2 categories × ~5 instances.** v0.7 marks the final size **TBD pending P0c**: if collecting ≥4 categories × ≥10 instances (covering the main weld types) turns out feasible after the P0c rig is built, contribution ⑤ keeps "weld feature benchmark" wording; if not, it is downgraded to "large-scale auto-GT benchmark + measured-GT validation subset". The spec carries both wordings with the decision flagged.
6. **§3.9 10a still said "contact faces were forced consistent".** Inconsistent with §3.8's soft-penalty story. v0.7: "regularized toward consistency", and adds that the analytic locus's *position* accuracy still inherits the pose/size/contact-face residual (it is topologically robust, not positionally exact).

Engineering suggestions absorbed:
- **Latency microbenchmark before P2** — a new P1.5 milestone runs synthetic-data timing of backbone / Stage-2 assembly / NvDiffRast at N_unroll∈{1..4} / RefineTransformer on the deployment-class GPU, so the 250-500 ms budget is sanity-checked early, not at P4.
- **P0c rig modes split** — hand-held laser wand: produces local measured-GT only (cannot validate hand-eye/base/TCP); arm-mounted laser: validates the full robot/world coordinate chain. §7 P0c now describes the two modes separately and says which deliverable each produces.

### v0.5 → v0.6 (2026-05-11, Codex second-review round)

v0.5 was handed back to Codex. Verdict: the *direction* is sound, but (a) the v0.5 edits introduced four internal contradictions, (b) several of Codex's first-round fixes were acknowledged in wording but not closed in engineering, and (c) the headline claim is still over-stated. v0.6 absorbs all of this. Net effect: the architecture is unchanged; the claims are downgraded to what is actually defensible; the pilot plan is reordered to test physical lower bounds *before* training large models.

**Round 6-0 — Headline claim downgrade (the big one).**
- Old framing: a part-aware pose method that produces mm-level weld seams end-to-end.
- New framing: **"CatSpec enables open-category coarse part pose and weld-seam *proposal*; precision-tier sensing (runtime local laser scan + full calibration chain) is what turns a proposal into industrial mm-grade execution."** The coarse RGB-D pipeline is a *proposal generator* (cm / low-cm level), not an mm-grade metrology system. This re-states §1.1, the research questions, §2.1 contribution ①, and §2.3. It is harder to attack and matches the engineering facts.

**Round 6-A — Fix the four v0.5 self-contradictions.**
1. *HARD vs penalty schedule* (§3.8 said "HARD contact constraints"; R25 said "penalty schedule, not a t=0 hard constraint"). v0.6: there are **no true hard constraints**; all adjacency/contact terms are soft penalties whose weight follows a hardening schedule over training and over the refiner's internal unrolled steps. "HARD when the spec declares it" is replaced by "high-weight when the spec declares it".
2. *single-pass vs internal sub-iterations* (§3.8 said "single forward pass"; R25 referenced "internal sub-iterations"). v0.6: the refiner internally **unrolls N_unroll ≤ 4 steps** (a fixed small number, not iterate-to-convergence) — this is what makes the R,t,s coupling tractable and what R25's hardening schedule lives on. At inference it is still *one network call*, but that call does N_unroll internal steps. For high-uncertainty cases an **optional explicit K-step refinement fallback** is available (off by default).
3. *Precision-tier semantics* (§1.5 implied runtime seam re-scan; §2.1/R23 said "one-time per-category calibration"). v0.6: precision tier = **per-instance / per-fixture runtime local laser scan** of the seam neighbourhood — because fixture variance, manufacturing tolerance, thermal distortion and assembly error are per-instance and a once-per-category calibration cannot cancel them. The open-category "zero training cost" only buys the **coarse-tier proposal**; the precision tier is a per-deployment-instance operation that *any* mm-grade welding system needs regardless of how the category was onboarded.
4. *Latency numbers disagree* (§1.5 ~500 ms coarse; §3.4 <250 ms; P4 gate on H100; original target RTX 3090). v0.6: state one consistent budget — **coarse-tier inference target 250-500 ms on a single modern workstation GPU (RTX 4090 / A6000 class); RTX 3090 will be slower; this is a target to be *verified* in P4, not a guarantee**; per-step breakdown is reported. Training is on 8×H200. The precision-tier laser scan adds ~1-3 s per local scan and is out of the per-frame budget by design (it is an escalation, not the default path).

**Round 6-B — Close the "acknowledged but not engineered" gaps.**
- *Calibration chain* (Codex: hand-eye / laser-camera / robot-base / TCP error budget was missing). v0.6 §1.5 adds an explicit **calibration-chain sub-component** with its own error budget; the precision-tier accuracy claim (1-3 mm in robot/world coordinates, 0.3-1.0 mm in local sensor coordinates) is stated *with* this chain, not without it.
- *Self-circular GT* (Codex: Stage-0 GT and 10a prediction both come from the same spec/contact-face declaration → the benchmark may only prove "the model reproduced the spec"). v0.6: AIWS-Weld gains a mandatory **independent-measurement subset** — for ≥ 2 categories, several real or high-fidelity-scanned instances with laser/CMM-measured centerline + endpoint GT, captured with the full sensor + calibration stack. All headline weld-feature metrics are reported on *both* the auto-GT split and the measured-GT subset; a large gap between the two is itself a reported finding. (See §4.1, §4.2, P1.)
- *Open-category evidence* (Codex: 4-train/2-test is a pilot, not proof). v0.6 strengthens the open-category protocol: synthetic DSL category augmentation (random valid compositions), leave-family-out splits (not just leave-category-out), and cross-dataset transfer (BOP-Industrial) are all required for the open-category headline number — 6 real categories alone are explicitly insufficient and labelled as such.
- *Uncertainty-gate threshold realism* (Codex: τ=2-3 mm conflicts with coarse-tier 5-10 mm depth noise). v0.6: on the coarse tier the gate is **not** a mm-certification — it is a *triage* rule whose job is to decide *which seams need a precision-tier scan*. Coarse-tier τ is set relative to coarse-tier achievable precision (cm-scale), and "every seam that will actually be welded gets a precision-tier scan" is the default industrial posture; the gate just lets low-stakes / tack-weld seams skip it. The 2-3 mm threshold applies to the *precision-tier* output.

**Round 6-C — Reorder the pilots: physical lower bounds before big-model training (rewrites §6 / §7).**
- New P0 sequence (all before any heavy training):
  - **P0-oracle** — feed GT part mask + correct *metric* CAD + correct contact faces + (simulated) D455 noise into the global fit; measure seam centerline error. If the oracle is already > 3-5 mm, no downstream network can recover it — the bottleneck is sensing/geometry, not learning, and the project must pivot to the precision-tier story immediately.
  - **P0-10a-noise** — inject {1, 2, 5, 10} mm and {0.5, 1, 2}° noise into part pose/size; measure the analytic contact-locus error of §3.9 10a, stratified by weld type. Quantifies the nonlinear amplification (R15) directly.
  - **P0-laser-loop** — 2 categories × ~5 instances: D455 proposes the scan window, a laser scanner (hand-held rig or arm-mounted) scans the local seam, an independent high-precision scan / CMM provides GT; report the *full coordinate-chain* error.
  - **P0a (metric upgrade)** — the existing no-CAD-expert pilot keeps its < 0.5 mm whole-mesh chamfer gate but adds **weld-adjacent face/edge metrics**: contact-face position error, normal error, dihedral error, contact-locus error. Whole-mesh chamfer hides errors exactly where the seam lives.
  - **P0b (gate upgrade)** — "spec-correct beats spec-shuffle by ≥ 30%" stays, but absolute thresholds are added: held-out coarse-tier centerline RMSE < 10 mm, failure rate < 10%, correct-contact-edge recall > 90%. A 30% relative gap with 40 mm absolute error is not a pass.
- P1 (dataset) must produce the independent-measurement subset above, or the weld-feature benchmark is not credible.

### v0.4 → v0.5 (2026-05-11, Codex independent-review round)

Context: a "fully zero-shot + consumer single-frame RGB-D + mm-precision" engineering alternative to v0.4 was drafted, then handed to Codex for independent third-party assessment. Codex's verdict: that combination is **not physically achievable** (optimistic 5-10 mm, realistic 15-40 mm with discrete catastrophic mis-matches), and its concrete recommendations — global assembly-constrained optimization, closed-set CAD-derived training, constraint-derived seams instead of naked mesh booleans, sensor layering, uncertainty gating — converge back onto v0.4's design. **Decision: keep the v0.4 architecture, absorb Codex's five fixes.** No contribution is dropped; several are sharpened or scope-qualified.

**Round 5-A — Sensing reality (v0.4 was silent on this).**
- v0.4 implicitly assumed RGB-D is enough for mm-level seam metrology. It is not: consumer RGB-D (RealSense D455) has ~5-10 mm depth noise at 0.5-0.6 m and an official working range that starts at ~0.6 m.
- v0.5 adds **§1.5 — two-tier sensing**:
  - **Coarse tier** — RGB-D (D455 / L515): whole-object localization + category + per-part coarse pose + coarse size, ~500 ms. This is what all main-paper experiments use.
  - **Precision tier (optional module)** — industrial blue-laser line scan / structured light: re-scans the coarse seam neighbourhood, pulling centerline accuracy to sub-mm (laser seam-tracking literature reports 0.1-0.3 mm). Reported as a supplementary experiment showing the precision ceiling, not as a hard dependency of the main claims.
- Honest framing: main results report centerline mm-error on RGB-D + dataset GT; the laser add-on is the path to industrial sub-mm.

**Round 5-B — Seam extraction: constraint-derived, not naked mesh boolean (rewrites §3.9 Step 10a).**
- v0.4's 10a was "robust mesh-mesh boolean with tolerance ε + point-cloud snap". Codex: with two independently-fitted parts at 5-15 mm relative error, an ε≈1 mm boolean yields empty / fragmented / multi-branch curves; the snap can't recover a true centerline because real seams are edges/grooves without clean depth points.
- v0.5 changes 10a to **derive the seam from the analytic contact constraints of the fitted parametric primitives** (plate ⊥ cylinder → contact circle; two plates at an angle → contact line), with "contact faces coplanar / perpendicular / parallel" promoted to **hard constraints inside the Stage-2/3 joint optimization**. The seam is a by-product of the constrained fit, not a post-hoc CSG operation. Point-cloud snap survives only as a *precision-tier* (post-laser-scan) refinement.

**Round 5-C — Make Stage-3 refinement an explicit global assembly optimization (sharpens contribution ③ / §3.8).**
- v0.4 already does joint SE(3)^K and has "inter-part guidance terms". v0.5 makes the energy explicit: variables = object pose + per-part (R, t, s) + contact-face parameters; residuals = depth point-to-plane + silhouette IoU + CAD contact/perpendicular/parallel constraints + size-range priors (from the spec) + weld-graph rules. The single-pass amortized refiner is trained to *amortize this energy*, which makes the whole stage interpretable and gives it the structure Codex asked for.

**Round 5-D — Uncertainty gating loop (v0.4 had uncertainty heads but no decision rule).**
- v0.5 adds an explicit gate: each seam outputs centerline covariance + endpoint covariance + torch-angle uncertainty; if predicted uncertainty exceeds τ (≈2-3 mm) **or** the per-part pose residual is inconsistent with the adjacency constraints, the system triggers (a) multi-view re-scan, (b) precision-tier laser re-scan, or (c) human confirmation — it does **not** emit a "mm-grade" result. Goes into §2.2 (robustness) and the deployment story.

**Round 5-E — Add the ablations that isolate where error comes from (expands §4.4).**
- Three new mandatory A/Bs Codex called for: (i) GT part mask vs predicted part mask; (ii) correct metric CAD vs unit-bbox CAD; (iii) RGB-D single-frame vs multi-view fusion vs +precision-tier laser. Metrics expand from "centerline mm-error" to centerline RMSE + Hausdorff + endpoint error + torch-angle error + **failure rate** (fraction of discrete catastrophic mis-matches).

**Open-category claim (contribution ①) — tiered statement.**
- Codex: "no truly zero-cost new-category extension exists". v0.5 keeps the open-category framing but states it in two tiers: writing a YAML spec for a new category immediately yields **coarse-tier** (cm-level) part pose + seam at zero training cost; reaching **precision tier** (mm-level) for that category requires a one-time laser calibration scan on first deployment — still far cheaper than per-instance CAD modeling, and no retraining. §2.1 carries this wording; the new R23 records the risk; R10's "categories expressible in the DSL" scoping still applies on top of it.

### v0.3 → v0.4 (2026-05-10, four audit rounds, same day)

Four follow-up audits the same day as v0.3 surfaced four classes of "too idealized" issues. v0.4 absorbs all four with concrete, citation-backed alternatives:

**Round 1 — Pipeline complexity and inference latency.**
- v0.3 estimated worst-case inference > 30 sec/frame (GenPose-style 50-candidate × 1000-step diffusion + OCCT in inner loop + N_iter=4 R&C).
- v0.4 replaces:
  - SE(3) diffusion → **SE(3)^K Flow Matching** (RFM-Pose Feb'26, SE(3)-PoseFlow Nov'25): 5 steps not 1000.
  - K=4 hypothesis → **K=1 + uncertainty head** (rely on flow matching's lower variance).
  - N_iter=4 iterative R&C → **single-pass amortized refinement** (RePOSE CVPR'22, NeFeS CVPR'24).
  - End-to-end joint training of all stages (DON6D 2024, IFPN, CRT-6D) to prevent error accumulation.
- New latency budget: **~250 ms / frame** (vs > 30 sec in v0.3).

**Round 2 — OCCT in inner loop is not GPU-friendly; Tier-3 boolean ops are non-trivial.**
- Literature confirms: SDF boolean → "Pseudo-SDFs" (CSG on Neural SDF SIGGRAPH-Asia'23); OCCT B-rep ops are CPU-bound 50-200 ms each.
- v0.4 splits Stage 2 into two paths:
  - **Path A — Parametric Mesh Templates** (SMPL-style; PHRIT ICCV'23, SOMA NVIDIA'26, DreamCAD 2026): 30 composite primitives each get a canonical mesh + analytic / MLP-learned vertex deformation function. ~10 ms inference. **Used for ~80% of workpieces.**
  - **Path C — FlexiCubes + analytic SDF + fuzzy-logic boolean** (FlexiCubes SIGGRAPH'23, Unified Differentiable Boolean 2024) for Tier 1-3 free DSL. ~30 ms. **Fallback for the remaining ~20%.**
- DiffCSG (SIGGRAPH-Asia'24) considered but rejected — engineering cost of replacing NvDiffRast pipeline too high.
- **OCCT/Build123d only used offline** (in library construction and auto-annotation), never in inference.

**Round 3 — "纯几何反推焊缝" was 3 problems collapsed into 1.**
- Literature decomposes seam derivation into 7 sub-problems: surface intersection, mesh artifact handling, weld type classification (WeldNet 99.6%, YOLOv5 18ms), groove geometry, endpoint localization, torch posture (dihedral angle), multi-pass planning.
- v0.4 splits step 10 into:
  - **10a — Robust geometric intersection** with tolerance ε + observation point cloud snap refinement (Sciencedirect 2025 medium-thickness plate).
  - **10b — SeamHead** (small MLP + analytic rules): outputs centerline + weld_type (6-way) + torch_attitude (3 angles) + endpoints + uncertainty.
  - **10c — Multi-pass planning** (deferred to future work, v1 paper does only single-pass).
- Spec YAML extended with `weld_meta` field per adjacency edge (weld_type prior, torch constraints, is_load_bearing).
- New evaluation metrics: centerline mm-error, weld type accuracy, torch attitude error, endpoint localization error.

**Round 4 — Team has no 3D modeling expertise; v0.3 implicitly assumed CAD modeling capability.**
- v0.3 implied 30-50 engineering days to author 30 composite primitives.
- v0.4 replaces hand authoring with a hybrid no-CAD-expert pipeline:
  - **Standard structural shapes (4 of 6 nuclear workpieces)** → [steelpy](https://github.com/evanfaler/steelpy) + AISC v16.0 database. Trivial integration.
  - **Custom non-standard shapes** → Claude Code + Build123d via Text-to-CadQuery paradigm (May'25, GPT-4o 90% / Claude 85% one-shot executable rate).
  - **Unknown geometry with only sample CAD available** → PrimitiveAnything (SIGGRAPH'25) auto-decomposition.
  - **Vertex deformation function** → per-primitive small MLP trained on samples generated from the parametric spec (NPM ICCV'21 paradigm). No analytic formulas hand-written.
- New cost estimate: **~12-15 engineering days, no 3D modeling expertise required.**

### v0.2 → v0.3 (2026-05-10, first audit round of the day)

A 2026-05-10 literature audit (RAG-6DPose IROS'25, Any6D CVPR'25, ZeroPose TCSVT'24, XYZ-IBD BOP'25, OMNI-PoseX 2026-04, PrimitiveAnything SIGGRAPH'25, CRISP CVPR'25, Vision-Guided Virtual Assembly for Welding Sciencedirect 2026) showed that three of v0.2's five contributions had been partially threatened by 2025 work. v0.3 reframes the project around a single new top contribution — **CatSpec-Pose** — that subsumes v0.2's Shape-CAD Retrieval head and recasts the project as **open-category** rather than closed-set:

- v0.2: trained on K categories, worked only on those K.
- v0.3: trained once, accepts any new category whose primitive composition is registered as a YAML spec — no retraining, no CAD modeling, no reference imagery.
- The "primitive library" is upgraded from a flat dict of 6 analytic primitives to a **4-tier CSG-style DSL** (analytic solid / profile-sweep / CSG modifier / neural residual). This handles thickness, swept profiles, fillets, and shells that the v0.2 dict could not express.
- All other v0.2 contributions (SE(3)^K diffusion, online-assembled render-and-compare, AIWS-Weld, seam mm-error) survive but get sharpened claims to avoid 2025 collisions.

---

## 1. Motivation & Problem Setup

### 1.1 Application context
Nuclear-industry robotic welding requires **mm-class** localization of weld seams on industrial workpieces (bellmouth, channel_steel, cover_plate, h_beam, square_tube, ...) — sub-mm is the long-run target on critical seams, achievable only where the precision tier and a well-calibrated robot/sensor chain allow it (§1.5: local laser measurement can be sub-mm; robot/world execution is currently ~1-3 mm). Current category-level 6D pose pipelines treat each workpiece as a monolithic rigid body; this discards the fact that **weld seams lie on the interface between primitive components** (e.g. a plate bonded to a cylinder).

A second observation: industrial welding lines see hundreds of workpiece variants per year. Any closed-set category-level method requires retraining for each new variant — an O(weeks) cost per variant that kills deployment economics. Industry workarounds (offline programming, vision-guided virtual assembly with full per-instance CAD registration) demand instance-level CAD modeling that itself takes hours-to-days per part. **Neither closed-set academic pose methods nor full-CAD industrial pipelines are economical for high-mix welding.**

**Where this work sits (v0.6 positioning).** This project does **not** claim to deliver sub-mm seam metrology from a single consumer RGB-D frame — two independent reviews established that combination is physically impossible. It delivers (i) **open-category part-level 6D pose + size + a weld-seam proposal** (centerline locus + endpoints + weld_type + torch attitude, with calibrated uncertainty) at the **coarse tier** (cm / low-cm), onboarding a new category by writing a YAML spec with no retraining and no per-instance CAD; and (ii) a clearly-scoped **precision tier** (per-instance runtime laser scan + calibration chain) that turns the proposal into mm-grade execution. The scientific contributions live in tier (i); tier (ii) is the engineering bridge to deployment and is demonstrated, not assumed.

### 1.2 Two converging gaps in the 2026 literature

**Gap 1 — Open-category 6D pose with structured prompts.**
Open-vocabulary pose work in 2024-2026 (OV9D'24, OMNI-PoseX'26, Horyon'25, VFM-6D NeurIPS'24) uses *text* as the test-time category descriptor. Text lacks geometric specificity — it cannot drive mm-precision welding. CAD-prompted work (ZeroPose TCSVT'24, SAM-6D CVPR'24) requires a *full instance CAD* — defeats the deployment-economics goal. **No published method uses a structured parametric primitive composition spec as the test-time category descriptor.**

**Gap 2 — Part-level 6D pose under category-level inputs.**
Existing part-level pose work is either articulated-object (ARTICULATE-ANYTHING ICLR'25; kinematic, not compositional) or assumes per-instance CAD. **No published method does part-level pose under the category-level (no test-time CAD) regime, and none links it to a downstream physically-meaningful metric like weld seam mm-error.**

### 1.3 Privileged information setup (preserved from v0.2, re-grounded)
The project has access to a prior configuration not matched by any existing academic benchmark:

| Prior | Train-time | Test-time |
|---|---|---|
| Instance CAD model | Available (privileged) | Not available |
| Category primitive composition spec (DSL) | Available | Available (manually registered, ~5 min/category for common shapes) |
| Instance size (W, H, D) | Available (from instance CAD) | Not available |
| Part-level 6D pose GT | Auto-generated from instance CAD | — |

This is a textbook **Learning Using Privileged Information (LUPI)** setup (Vapnik & Vashist 2009). v0.3's open-category framing reinforces it: at deployment time only the spec is available, not the instance CAD.

### 1.4 Research questions (updated v0.5)
1. Can a parametric primitive composition spec, used as a test-time prompt, drive a frozen pose model to estimate part-level 6D pose on workpiece categories never seen during training?
2. Can SE(3)^K **flow matching** (RFM-Pose / SE(3)-PoseFlow) exploit train-time privileged instance CAD to learn part-aware pose, generalizing at test time using only category-level priors and the spec prompt — within 5 sampling steps?
3. Does part-level pose estimation produce more accurate weld-seam localization than whole-object pose + post-hoc slicing, on both seen and unseen categories?
4. **(v0.4 new; v0.6 re-scoped)** Given a coarse-tier part pose with cm-level error, can a learned **SeamHead** stably output a *welding proposal* (centerline locus + weld_type + torch_attitude + endpoints + calibrated uncertainty) that is good enough to (a) drive a precision-tier laser scan to the right place and (b) be promoted to mm-grade execution after that scan — i.e. is the coarse proposal a sufficient prior, even though it is not itself mm-grade?
5. **(v0.5 new)** What fraction of the end-to-end seam-localization error is attributable to (a) part segmentation, (b) size/scale estimation, (c) the depth sensor — and which of these must be addressed by a precision-tier sensor rather than by a better algorithm? (Answered by the P0-oracle / GT-mask / metric-CAD / sensing-tier ablations.)

### 1.5 Sensing & calibration configuration (NEW in v0.5; corrected in v0.6)

Two independent reviews (Codex, 2026-05-11) established that v0.4 silently assumed RGB-D suffices for mm-level seam metrology — it does not — and that v0.5's first attempt at a fix had inconsistent latency numbers and conflated "runtime local scan" with "one-time category calibration". v0.6 states the stack as **two sensing tiers + one calibration chain**, with one consistent budget.

**Tier 1 — Coarse (the default, drives all main-paper experiments):**
- Sensor: consumer RGB-D — RealSense D455 / L515 class, or equivalent.
- Role: whole-object localization, category confirmation, per-part coarse 6D pose, coarse anisotropic size, and a **weld-seam proposal** (centerline locus + endpoints + weld_type + torch attitude) with calibrated uncertainty.
- Accuracy: ~5-10 mm depth noise at the working distance; the produced seam is **cm / low-cm level**, not mm.
- Latency target: **250-500 ms / frame on a single modern workstation GPU (RTX 4090 / A6000 class)**; an RTX 3090 will be slower; this is a target to be *verified in P4*, not a guarantee. Per-step breakdown is reported (backbone / flow matching / mesh / refine / SeamHead).
- Working distance: D455's stable range starts ~0.6 m; the experimental rig must respect this (0.5 m is too close). Sensor self-calibration and texture/depth QC are part of the data-collection protocol.

**Tier 2 — Precision (per-instance / per-fixture runtime escalation, not a per-category one-off):**
- Sensor: industrial blue-laser line scan / structured-light scanner.
- Role: re-scan **the seam neighbourhood of the specific instance currently in the fixture**, refine centerline + endpoints (feeds the precision-tier refinement of §3.9 10a Step 2). It is triggered per workpiece instance, because fixture variance, manufacturing tolerance, thermal distortion and assembly error are all per-instance — a once-per-category calibration cannot cancel them.
- Accuracy: ~0.3-1.0 mm in local sensor coordinates (laser seam-tracking literature: 0.1-0.3 mm under favourable conditions); **~1-3 mm in robot/world coordinates once the calibration chain is folded in** (see below).
- Latency: ~1-3 s per local scan. **Out of the per-frame budget by design** — it is an escalation path (§2.2), not the default loop.
- Default industrial posture: *every seam that will actually be welded gets a Tier-2 scan*; the uncertainty gate (§2.2) only lets low-stakes / tack-weld seams skip it.

**Calibration chain (its own error budget — added in v0.6 per Codex):**
- Components: camera intrinsics + RGB-D extrinsics; laser-scanner ↔ camera transform; camera/scanner ↔ robot flange (hand-eye); robot base ↔ world; TCP (torch tip) calibration.
- Each link contributes; the Tier-2 "1-3 mm in robot/world coordinates" figure is stated *with* this chain included. P0-laser-loop (§7) measures the realized end-to-end coordinate-chain error rather than assuming the per-sensor spec.
- The paper reports the calibration procedure and its residuals; "0.3 mm laser accuracy" is never quoted as the system accuracy.

**What this means for the claims:**
- All headline metrics (Part-ADD, centerline error, weld-feature metrics) are reported on the **coarse tier vs dataset GT**, and additionally on an **independent-measurement subset** (laser/CMM GT, §4.1) so the auto-GT-vs-measured-GT gap is itself visible. "mm-error" in a table is *measured against GT*, never an industrial mm guarantee.
- The precision tier is a clearly-scoped module, demonstrated by one ablation row (coarse-only vs coarse + Tier-2 laser, §4.4 #20) and the P0-laser-loop pilot. It is the path to industrial mm execution; it is not a dependency of the core scientific contributions.
- Open-category honesty: writing a YAML spec for a new category yields a **coarse-tier proposal at zero training cost**; mm-grade execution on that category needs a Tier-2 scan of each instance — exactly as it would for an already-onboarded category. The zero-cost claim is about *not retraining and not modeling per-instance CAD*, not about skipping the precision sensor.

---

## 2. Scope & Contributions

### 2.1 Planned contributions (5, sharpened in v0.4)

1. **CatSpec-Pose — parametric primitive specification as test-time category prompt.** Replace implicit category priors (GenPose) and unstructured visual category descriptors (text in OV9D, full CAD in ZeroPose) with a structured **primitive composition DSL** as the test-time category prompt. New categories are onboarded by writing a ~30-line YAML spec (no retraining, no CAD modeling, no reference image collection); 80% of common shapes only need to fill numeric params for one of 30 pre-defined composite primitives. The spec is encoded by a lightweight transformer and conditions all downstream heads via cross-attention. **First method to use a structured CSG-style DSL as a 6D pose prompt.** **(v0.6 scope statement)** the zero-cost onboarding delivers a *coarse-tier proposal* (cm / low-cm part pose + size + seam locus); turning that into mm-grade execution requires a *precision-tier per-instance laser scan* (§1.5) — which any mm-grade welding system needs for any workpiece, onboarded or not. The "zero cost" is about **no retraining and no per-instance CAD modeling**, not about skipping the precision sensor; this is still well clear of ZeroPose's full-CAD requirement and of OV9D/OMNI-PoseX's text-only specificity gap.

2. **Part-aware SE(3)^K joint pose flow matching.** Extend the generative SE(3) pose paradigm (GenPose, Diff9D) to a joint SE(3)^K over K primitive parts using **rectified flow matching on the SE(3) manifold** (RFM-Pose Feb'26, SE(3)-PoseFlow Nov'25), with inter-part physical constraints (no-overlap, adjacency from spec) as guidance terms. **First generative pose work that operates on joint pose of multiple parts rather than a single rigid body, using flow matching to achieve 5-step sampling versus diffusion's 50–1000 steps.**

3. **Bounded-compute amortized global-assembly render-and-compare on online-assembled primitive proxy with zero test-time reference.** Extend FoundationPose-style refinement to the category-level setting. Stage-1 outputs (primitive type from spec + size + coarse pose) are assembled on-the-fly into a differentiable proxy mesh via parametric mesh templates (Path A) or FlexiCubes (Path C). A transformer-based refinement head runs as **one network call with a fixed N_unroll ≤ 4 internal render/compare/Δ steps** (not a data-dependent iterate-to-convergence loop), trained as an amortized refiner (RePOSE CVPR'22, NeFeS CVPR'24) of an explicit global assembly-constrained energy (§3.8). **First render-and-compare method that simultaneously achieves: zero test-time reference (no anchor image vs Any6D, no reference video vs FoundationPose model-free, no full CAD vs ZeroPose) and bounded-compute amortized refinement (one network call, fixed N_unroll — vs a data-dependent N_iter=4-8 outer loop in classical iterative R&C).**

4. **AIWS-Weld dataset with open-category protocol and weld feature annotations.** Public release of a 6-category nuclear-workpiece dataset with part-level annotations (auto-generated from instance CAD), plus a held-out **open-category split** where N categories train and the remaining M are tested only via spec prompts, plus per-edge **weld feature annotations** (centerline + weld_type + torch_attitude + endpoints) auto-generated from the spec's `weld_meta` field. **First industrial pose benchmark with part-level annotations, an open-category protocol, AND welding-derived seam + feature GT.**

5. **Weld feature benchmark — beyond mm-error.** Introduce downstream, physically-meaningful evaluation metrics computed by the SeamHead from predicted part poses: centerline RMSE + Hausdorff, weld type 6-way classification accuracy, torch attitude per-axis error, endpoint localization error, and **failure rate** (discrete catastrophic mis-matches). Reported on **both** the auto-generated-GT split *and* an independent measured-GT subset (laser/CMM GT, §4.1), and at **both** sensing tiers (coarse-only and coarse + precision laser). **(v0.7 — wording pending the P0c size decision, §4.1):** *if* the measured subset reaches ≥ 4 categories × ≥ 10 instances → **"first weld-localization benchmark grounded in 6D part pose, across sensing tiers, against measured (not just CAD-derived) ground truth"**; *if it stays at the ~2×5 seed* → **"first large-scale auto-GT weld-feature benchmark grounded in 6D part pose, plus a measured-GT validation subset that breaks the spec-self-circularity"**. Either way it is the first benchmark to evaluate the full feature set a welding robot needs (centerline + type + attitude + endpoints + failure rate), not centerline alone.

### 2.2 Secondary contributions (robustness section)
- Depth-degraded inference fallback using Depth-Anything-V2 on samples with zero / corrupted depth (≈52% of current dataset).
- Category-balanced geometric augmentation (already implemented on EoMT-dev branch).
- **(v0.5; thresholds split in v0.7)** Uncertainty-gated escalation loop: every seam carries centerline/endpoint covariance + torch-angle uncertainty. **Two distinct thresholds**:
  - `τ_coarse` (cm-scale, e.g. ~5-15 mm — calibrated by ablation 22 / P0-10a-noise): the *triage* threshold. On the coarse tier the gate's only job is to decide **which seams need a Tier-2 laser scan**; it does **not** certify mm accuracy. If coarse uncertainty exceeds `τ_coarse` **or** the per-part pose residual contradicts the adjacency constraints, escalate (multi-view re-scan → precision-tier laser scan → human confirmation). The default industrial posture is "every seam that will actually be welded gets a Tier-2 scan anyway"; `τ_coarse` just lets low-stakes / tack-weld seams skip it.
  - `τ_precision` (≈2-3 mm — or stricter per the AWS class of the joint): the *execution* threshold, applied to the **precision-tier** result. A precision-tier seam whose post-scan uncertainty still exceeds `τ_precision` is flagged for human confirmation rather than welded.
  - Both calibrated via L_uncertainty (§3.10) and reported with reliability diagrams (predicted σ vs realized error) at each tier.
- Tier-4 neural residual deformation for parts that escape the analytic + sweep DSL — *future work*, scoped out of v0.3 main paper.

### 2.3 Out of scope
- Physical robot demo (future work) — but P0-laser-loop (§7) does run a small real laser-scan closed-loop measurement, which is *not* the same as a robot welding demo.
- **mm-grade seam metrology from coarse RGB-D alone** — explicitly out of scope; that is the precision tier's job (§1.5). The coarse pipeline outputs a *proposal*, not a certified mm result.
- Symmetric-object ambiguity *modeling* as a research contribution (handled at primitive level via per-primitive symmetry-group annotation, see §3.1).
- Online active perception / next-best-view (the uncertainty gate's escalation is a fixed triage rule, not learned NBV).
- Auto-induction of the spec from images / point clouds — future work; can leverage DeepCAD / PrimitiveAnything as initialization tools.
- Multi-pass / multi-layer thick-plate planning (§3.9 10c) — future work.

---

## 3. Technical Design

### 3.1 Primitive Library — 4-Tier CSG-Style DSL

The v0.2 flat dict of 6 analytic primitives could not express (a) wall thickness independently of outer geometry, (b) swept variable cross-sections, (c) fillet/chamfer transitions, (d) hollow shells. v0.3 replaces it with a CSG-style DSL inspired by ShapeAssembly (TOG'20) and OpenCASCADE B-rep semantics, restricted to the operations actually needed for industrial welding workpieces.

**Tier 1 — Analytic Solid Primitives** (~10 closed-form types, each with parameters, canonical local frame, and symmetry-group annotation):
```
box, cylinder, cone, sphere, torus, plate,
tube,           # cylinder with inner_radius — wall thickness first-class
prism_n,        # n-sided polygonal prism (hex nuts etc.)
ellipsoid, helix
```
Symmetry-group annotations: `{C_∞, D_∞, C_n, Z_2, None}` per primitive. Used by §3.7's symmetry-aware pose loss to project predictions to the SE(3)/G quotient.

**Tier 2 — Profile + 1D Operations** (handles variable cross-section, swept solids — covers the U-channel example and most plate-with-hole / channel-block parts):

2D profile vocabulary (closed planar curves):
```
rect, rounded_rect, circle, ellipse,
u_section, l_section, i_section, channel_section,
custom_polyline    # tail-coverage: arbitrary closed 2D polyline / B-spline
```
1D operations on profiles:
```
extrude(profile, length, [draft_angle])      # straight extrude (constant section)
loft(profile_start, profile_end, length)     # variable section
revolve(profile, axis, angle)                # axisymmetric solid
sweep(profile, path_curve)                   # extrude along curve (bent pipes)
helical(profile, axis, pitch, turns)         # helices, threads
```

**Tier 3 — CSG Modifiers** (post-process on solids):
```
boolean_union | subtract | intersect
fillet(target, edges, radius)
chamfer(target, edges, distance)
shell(target, faces_to_remove, thickness)    # solid → uniform-thickness shell
mirror(target, plane)
linear_pattern(target, dir, count)           # arrays (bolt holes)
```

**Tier 4 — Neural Residual Deformation** (long-tail fallback, *not in main paper*):
```
neural_residual(base_mesh, conditioning, representation=FFD|DeepSDF)
```
Activated only when DSL coverage fails. Phase-1 paper reports DSL coverage rate as a metric and scopes Tier 4 to future work.

**Pre-defined composite primitives** (~30 named patterns wrapping common Tier 2+3 sequences):
`channel_block`, `tube`, `flanged_pipe`, `L_bracket`, `bellmouth`, `H_beam`, `square_tube`, `box_with_holes`, `welded_T_section`, … (the full list is finalized during P0.5).

The composites preserve the "5 minutes to register" promise. Three-tier user interface:

| Workpiece complexity | Interface | Onboarding time |
|---|---|---|
| Common (~80%) | Pick from 30 composites; fill 5-10 numeric params | ~5 min |
| Specialized (~15%) | Hand-write Tier 1-3 DSL | ~30-60 min |
| Truly bespoke (~5%) | Tier 4 neural residual | varies, future work |

**Concrete example — image of the U-channel block (2026-05-10 discussion)**:
```yaml
channel_block_demo:    # outer block with rounded-bottom U trench
  operations:
    - {id: outer, op: extrude, profile: {type: rect, W: 100, H: 100}, length: 80}
    - {id: trench, op: extrude,
       profile: {type: u_section, outer_W: 60, outer_H: 80,
                 inner_W: 25, depth: 70, bottom_fillet: 15},
       length: 80}
    - {id: final, op: subtract, operands: [outer, trench]}
```
Three lines, parametric, deterministic — handles the case the v0.2 dict could not express.

**Implementation backend**: DSL parses to **Build123d** (Python wrapper of OpenCASCADE Technology) **at offline time only** (library construction, auto-annotation). Mesh extraction via OCCT's `BRepMesh_IncrementalMesh`; SDF via `mesh2sdf` for training supervision. **OCCT/Build123d is never invoked in the inference path** — at runtime, mesh assembly is handled by parametric mesh templates (Path A, §3.7) or FlexiCubes (Path C, §3.7).

**Library construction without 3D modeling expertise** — see §3.2 for the hybrid no-CAD-expert pipeline (steelpy + Claude Code + PrimitiveAnything + per-primitive MLP). The team does not need to write OCCT or Build123d code by hand.

### 3.2 Library Construction — no-CAD-expert hybrid pipeline (NEW in v0.4)

The 30 composite primitives are constructed once via a four-strategy hybrid that **requires no 3D modeling expertise**. The team needs only Python literacy, Claude Code access, and visual judgment of rendered previews.

**Strategy A1 — Standard structural shapes via [steelpy](https://github.com/evanfaler/steelpy)** (covers most of AIWS-Weld):
- `h_beam` → AISC W shape; `channel_steel` → AISC C shape; `square_tube` → AISC HSS; `cover_plate` → simple rectangular plate.
- One-line API call returns full parametric specs (depth, web thickness, flange width/thickness, etc.) backed by [AISC Shapes Database v16.0](https://www.aisc.org/publications/steel-construction-manual-resources/16th-ed-steel-construction-manual/aisc-shapes-database-v16.0/).
- For non-AISC standard parts: [TraceParts API](https://github.com/TraceParts) (100M+ industrial CAD models, free download in STEP/STL/IGES).
- Estimated effort: < 1 day to integrate steelpy + TraceParts client; covers 4 of the 6 nuclear workpiece categories.

**Strategy A2 — Custom shapes via Claude Code + Build123d (Text-to-CadQuery paradigm)**:
- For non-standard parts (`bellmouth` curved transitions, custom flanges, etc.).
- The team describes the geometry in natural language; Claude Code authors a Build123d Python script.
- Empirical first-attempt executable rate: GPT-4o 90%, Claude 85% (Text-to-CadQuery May'25, arXiv:2505.06507); feedback-loop boosts to ~95%.
- Open-source [text-to-cad harness (MIT)](https://www.r2clickthrough.com/text-to-cad-open-source-harness/) provides Build123d scaffolding + auto-export to STEP/STL/DXF/URDF.
- Estimated effort: ~1 hour per non-standard primitive (incl. visual feedback iteration). Total ≈ 1-2 days for the remaining 1-2 nuclear categories.

**Strategy A3 — Auto-induction from sample CAD via PrimitiveAnything (SIGGRAPH'25)**:
- For "we have a sample STL but no clean parametric description" workpieces.
- [PrimitiveAnything](https://primitiveanything.github.io/) decomposes any 3D mesh / point cloud into cuboids + elliptical cylinders + ellipsoids with SE(3)+scale parameters; trained on 120K HumanPrim samples; auto-regressive transformer; 95%+ storage reduction vs raw mesh.
- Used as fallback when standard libraries and LLM-authoring don't fit.
- Risk R19: PrimitiveAnything is trained on general 3D shapes, not industrial parts; pilot on 1-2 instances first to assess decomposition quality.

**Strategy B — Vertex deformation function via per-primitive MLP (NPM paradigm)**:
- Each composite primitive needs a function `(canonical_vertex, size_params) → vertex_displacement`.
- v0.3 had this as "hand-written analytic formula" (requires geometry expertise). v0.4 replaces with: sample N≈500 (size_params) tuples from the parametric spec, run Build123d to generate N STLs, train a small MLP on the (params, vertex_id) → displacement supervision.
- Precedents: [NPMs ICCV'21](https://github.com/pablopalafox/npms), NeuraLeaf ICCV'25, SOMA NVIDIA'26 — all train neural deformers on samples without handcrafted constraints.
- Estimated effort: ~1 hour training per composite × 30 = ~4 days.

**Total cost estimate (v0.4, no-CAD-expert)**:
| Task | Effort |
|---|---|
| steelpy + TraceParts API integration | 1 day |
| Claude Code + Build123d for 1-2 custom shapes | 1-2 days |
| PrimitiveAnything pipeline setup (fallback) | 2-3 days |
| Per-primitive MLP deformation training pipeline | 4 days |
| Validation (chamfer distance vs OCCT GT) | 2-3 days |
| **Total** | **~10-13 engineering days** |

vs v0.3's implicit ~30-50 days assuming hand-authored DSL + analytic deformers. **No team member needs to know OCCT, B-rep semantics, or CAD modeling conventions.**

**Future workpiece onboarding** (after the library is built once):
- Standard structural shape → look up in steelpy / AISC database (5 min).
- Non-standard industrial part → check TraceParts catalog first; if not present, Claude Code authoring (~30-60 min).
- Truly novel custom shape → PrimitiveAnything auto-induction from one sample STL (a few minutes inference, ~30 min validation).

### 3.3 Auto-annotation pipeline (Stage 0)

One-time offline preprocessing per **training** category:
1. Load instance CAD (`.obj` / `.ply`).
2. Parse the per-category DSL spec → expected primitive tree.
3. Use OCCT to instantiate the spec at parameterized sizes; RANSAC / least-squares fit each primitive's size + local pose to the instance mesh.
4. **Sanity check**: fit residual < threshold AND reconstructed volume IoU ≥ 0.95; otherwise discard sample.
5. Render each primitive through the camera model → per-frame 2D part masks.
6. Export `part_annotations.json` alongside existing COCO annotations.

Acceptance gate: overall pass rate ≥ 80%; median IoU ≥ 0.9.

For **open-category test categories**, only steps 1-4 are executed (to validate the spec); steps 5-6 are skipped because no training data is generated. At test time the spec is the only category-level input besides RGB-D.

**v0.4 addition — Weld feature GT auto-generation**: when `spec.adjacency_edges` declare `weld_meta` (weld_type prior, torch_constraints, is_load_bearing), Stage 0 also exports per-edge GT for SeamHead training: centerline (analytical from part interfaces), weld_type, torch_attitude (from dihedral angle of fitted primitives + AWS standard rules), endpoints. These become the supervision for §3.9 SeamHead.

### 3.4 11-step pipeline (v0.4; budgets corrected in v0.6)

The full method is decomposed into 11 steps spanning three time scales: **one-time library construction** (steps 1-2), **per-category onboarding** (steps 3-4), and **per-frame coarse-tier inference** (steps 5-11, **target 250-500 ms on an RTX 4090 / A6000-class GPU — to be verified in P4, not a guarantee; an RTX 3090 is slower**). The optional precision-tier laser re-scan (§1.5 Tier 2) is an escalation path, ~1-3 s per local scan, deliberately *outside* this per-frame budget.

```
                       ╔══════════════════════════════════════════╗
                       ║  ONE-TIME LIBRARY CONSTRUCTION (~12 days) ║
                       ║  Step 1. Parametric mesh template lib    ║
                       ║          (Path A — see §3.7)              ║
                       ║  Step 2. FlexiCubes + analytic SDF lib   ║
                       ║          (Path C fallback — see §3.7)    ║
                       ╚══════════════════════════════════════════╝

                       ╔══════════════════════════════════════════╗
                       ║  PER NEW CATEGORY (5–30 min, mostly human)║
                       ║  Step 3. Author YAML spec (§3.1)          ║
                       ║          incl. adjacency_edges.weld_meta  ║
                       ║  Step 4. Auto GT generation (§3.3)        ║
                       ║          training cats only; open-cat skip║
                       ╚══════════════════════════════════════════╝

                       ╔══════════════════════════════════════════╗
                       ║  PER-FRAME COARSE-TIER INFERENCE          ║
                       ║  (target ~250-500 ms; RTX 4090/A6000;     ║
                       ║   verify in P4 — not a guarantee)         ║
                       ╠══════════════════════════════════════════╣
                       ║                                            ║
                       ║   YAML Spec  →  Step 5: SpecEncoder (5ms) ║
                       ║                          │                 ║
                       ║                   spec_tokens (K × d)      ║
                       ║                          │                 ║
                       ║   RGB-D     →  Step 6: DINOv3+PointNet++   ║
                       ║                         (30-50 ms)         ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Step 7: Stage 1 — spec-conditioned       ║
                       ║   EoMT++ + SE(3)^K Flow Matching (5 steps),║
                       ║   K=1 hypothesis + uncertainty (~150 ms)   ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Step 8: Stage 2 — GPU mesh assembly      ║
                       ║   (Path A template OR Path C FlexiCubes)   ║
                       ║   (10–30 ms; OCCT NEVER invoked)           ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Step 9: Stage 3 — amortized global-     ║
                       ║   assembly refiner: 1 network call,        ║
                       ║   N_unroll≤4 internal render/compare/Δ     ║
                       ║   steps via NvDiffRast (~50 ms)            ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Step 10: SeamHead-10a constraint-derived ║
                       ║   seam locus from fitted primitives'       ║
                       ║   contact faces (~20 ms; not naked CSG;    ║
                       ║   point-cloud snap = precision-tier only)  ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Step 11: SeamHead-10b weld feature head  ║
                       ║   → centerline + weld_type + torch_att.    ║
                       ║     + endpoints + uncertainty (~10 ms)     ║
                       ║                          │                 ║
                       ║                          ▼                 ║
                       ║   Uncertainty gate (§2.2): σ>τ or          ║
                       ║   adjacency-inconsistent → escalate to     ║
                       ║   multi-view / precision laser / human     ║
                       ║                                            ║
                       ║   (Optional: explicit K-step refine        ║
                       ║    fallback, off by default)               ║
                       ║   (Optional 12: SeamHead-10c multi-pass    ║
                       ║    planning — deferred to future work)     ║
                       ╚══════════════════════════════════════════╝
        ┌─────────────────────────────────────────────────────────┐
        │  PRECISION-TIER ESCALATION (per workpiece instance,      │
        │  ~1-3 s, OUTSIDE the per-frame budget):                  │
        │  laser line-scan of the proposed seam neighbourhood →    │
        │  snap centerline/endpoints to groove ridge → mm-grade    │
        │  result in robot/world coords via calibration chain      │
        └─────────────────────────────────────────────────────────┘
```

**End-to-end joint training** (DON6D 2024 / IFPN / CRT-6D 2023 evidence): all steps 5-11 (and the differentiable parts of step 8, including the N_unroll-step refiner) are joint-trained with a single unified loss (§3.10) to prevent error accumulation across the cascade. No staged pretraining + freezing.

### 3.5 SpecEncoder

Inputs: parsed DSL spec (operation graph + parameter ranges + symmetry-group annotations + **v0.4 addition** `weld_meta` per adjacency edge).

Architecture:
- Each operation node → token via a small lookup (op-type embedding) + MLP for numeric params (size ranges encoded as Gaussian-RBF features).
- Adjacency from the DSL graph → relative positional encoding on tokens.
- Optional global category embedding (training-time auxiliary; replaced by spec at test time).
- Output: K spec_tokens, one per "part of interest" (typically each Tier 1-2 primitive that survives Tier-3 ops).

Properties:
- ~1M parameters, ~6 transformer layers; lightweight by design.
- **Shared across all categories** — no per-category retraining ever.
- Adding a new category at test time = parsing its YAML into tokens. No gradient computation, no fine-tuning.
- Robust to spec dropout (§3.7, R9): trained with random spec-token masking so the model doesn't collapse onto memorization.

### 3.6 Stage 1 Heads — Updated for v0.4 (Flow Matching)

Four heads, all operating on spec-conditioned queries. **v0.4 critical change**: the SE(3) pose head no longer uses diffusion; it uses **rectified flow matching on the SE(3) manifold** (RFM-Pose Feb'26, SE(3)-PoseFlow Nov'25), reducing sampling steps from ~50-1000 to **5**, and using **K=1 hypothesis + uncertainty head** rather than K=4 multi-hypothesis.

| Head | Input | Output | Notes |
|---|---|---|---|
| Part-EoMT++ | RGB-D feat + spec_tokens | K part masks | Extends current EoMT |
| **Part-Flow-Matching Pose** | masked part feat + spec_tokens | SE(3) per part + per-part scalar uncertainty | **Replaces v0.3's diffusion head**; rectified flow on SE(3) (SE(3)-PoseFlow); 5 integration steps; K=1 sample |
| Size Refinement | part feat + spec_tokens (carry size_range priors) | size_params per part within spec range | Replaces `networks/scalenet.py`; output clipped to spec range |
| Inter-part Consistency | all parts + spec adjacency | object-level pose/size + adjacency-violation loss | New |

**Why flow matching, not diffusion** — empirical evidence from RFM-Pose (Feb'26): "reduces required sampling steps by an order of magnitude compared with diffusion-based generative methods" on category-level 6D pose. SE(3)-PoseFlow (Nov'25): "the rectified flow matching objective enables accurate results within very few steps, owing to its continuous and constant velocity field formulation". This single substitution moves Stage 1 latency from ~25 s (GenPose-style 50 candidates × 1000 steps) to ~150 ms.

**Why K=1 + uncertainty, not K=4 hypothesis** — flow matching has substantially lower sample-to-sample variance than diffusion; multi-hypothesis selection has diminishing returns. The uncertainty scalar feeds into Stage 3's gating: high-uncertainty samples can optionally trigger a second pass (K=2 fallback), but K=1 is the default.

**Note on the v0.2 → v0.3 collapse (preserved)**: v0.2's "Shape-CAD Retrieval Head" is subsumed by CatSpec. The spec declares each part's primitive_type, so the model only *confirms* it (low-dim classifier sanity check). RAG-6DPose collision avoided.

### 3.7 Stage 2 — GPU Mesh Assembly (Path A + Path C, NEW in v0.4)

v0.3's Stage 2 invoked OCCT/Build123d in the inner loop (200-500 ms × K hypotheses; CPU-bound; non-batchable). v0.4 splits Stage 2 into two GPU-native, fully-differentiable paths, deciding per-spec at inference time which to use.

**Path A — Parametric Mesh Templates** (SMPL-style; covers the 30 named composite primitives, ~80% of workpieces):
- Each composite has a pre-tessellated canonical mesh + a learned `(canonical_vertex, size_params) → vertex_displacement` MLP (built per §3.2 Strategy B).
- Inference: GPU broadcast of vertex deformation, no OCCT call. ~10 ms per composite.
- Topology fixed; only vertex positions vary with size_params. Tier-3 modifiers (fillet/chamfer/shell) are baked into the canonical mesh + parameterized via the deformer.
- Precedents: SMPL (TOG'15), PHRIT (ICCV'23), DreamCAD (2026), SOMA (NVIDIA'26 — GPU-Warp accelerated parametric body model).

**Path C — FlexiCubes + analytic SDF + fuzzy-logic boolean** (fallback for Tier 1-3 free DSL not covered by composites):
- Each Tier-1 primitive has an analytic SDF (Inigo Quilez SDF library style: box, cylinder, torus, sphere, ...).
- Tier-3 boolean via fuzzy-logic operators (Unified Differentiable Boolean Operator with Fuzzy Logic, 2024) to avoid Pseudo-SDF problem (CSG on Neural SDF, SIGGRAPH-Asia'23).
- Voxelize on a GPU grid (typ. 64³ or 128³); extract mesh via FlexiCubes (SIGGRAPH'23, NVIDIA Kaolin integrated). ~30 ms total.
- Mesh consumed by NvDiffRast in Stage 3 the same way as Path A output.

**Path selection** (per inference, automatic): if `spec` only references the 30 composite primitives → Path A; otherwise → Path C. Both produce trimesh-format output that is plug-compatible with NvDiffRast (Stage 3).

**Why DiffCSG (SIGGRAPH-Asia'24) was rejected**: although DiffCSG renders CSG models without explicit mesh extraction (concept-elegant), it requires replacing NvDiffRast's pipeline and the team would have to maintain a separate rasterization stack. Engineering cost not justified given Path A covers most cases.

### 3.8 Stage 3 — Amortized Global-Assembly Refinement (REWRITTEN in v0.4; framed as a global energy in v0.5; contradictions fixed in v0.6)

FoundationPose (CVPR 2024) achieves sub-mm precision through **iterative** render-and-compare (N_iter = 4-8) on an instance CAD at inference. v0.3 inherited this iterative structure with N_iter = 4. v0.4 replaced it with an **amortized refiner** (RePOSE CVPR'22, NeFeS CVPR'24): the network does the refinement work without an outer iterate-to-convergence loop, eliminating per-iter mesh re-assembly cost.

**v0.6 precision on "single-pass" — one network call, N_unroll internal steps.** The refiner does **not** run a single MLP step, and it does **not** iterate to convergence. It internally **unrolls a fixed small number of steps `N_unroll ≤ 4`** (a hyperparameter; default 3) — alternating "render proxy / compare to observation / predict Δ(pose, size, contact-face params)". This bounded unrolling is what makes the R,t,s coupling tractable (one step cannot disentangle rotation from anisotropic scale on a near-unit-bbox proxy; a few can) and is the structure on which the constraint-hardening schedule below lives. At inference it is still **one network call** with a **fixed** compute cost (no data-dependent loop length). An **optional explicit K-step refinement fallback** (a real outer loop, K up to 8) exists for cases the uncertainty gate flags as high-risk; it is **off by default** and reported separately.

**The energy the refiner amortizes** (all terms are soft penalties — there are no true hard constraints, see R25):

```
E( object_pose, {(R_k, t_k, s_k)}_{k=1..K}, contact_face_params ) =
      w_depth   · Σ point-to-plane / point-to-mesh residual (observation vs assembled proxy)
    + w_sil     · Σ silhouette-IoU residual (rendered mask vs EoMT++ part mask)
    + w_contact(τ) · Σ adjacency contact residuals (coplanar / perpendicular / parallel) — HIGH-WEIGHT when spec.weld_meta declares the contact type; weight w_contact ramps up over the N_unroll internal steps and over training (hardening schedule), never to ∞
    + w_overlap · Σ no-interpenetration penalty between parts
    + w_size    · Σ size-prior residual (s_k softly penalized toward the spec size_range; not hard-clamped)
    + w_seam    · Σ weld-graph rule residuals (seam locus must lie on a contact face declared in spec.adjacency_edges)
```

The RefineTransformer is trained (with privileged GT renders) so that its N_unroll-step output minimizes E. This makes the refinement interpretable, gives the seam extraction (§3.9) a constrained foundation rather than a brittle post-hoc boolean, and is the structure the independent reviews recommended over "independent part fit → mesh intersection".

CatSpec-Pose's combined claim (v0.6 wording): **first render-and-compare method achieving (a) zero test-time reference (no anchor image vs Any6D, no reference video vs FoundationPose model-free, no full CAD vs ZeroPose) AND (b) bounded-compute amortized refinement (one network call, N_unroll ≤ 4 internal steps — vs a data-dependent N_iter outer loop in classical iterative R&C)**. The spec + Stage-1 predictions assemble the proxy; the refiner's bounded unrolling produces the final part poses.

Key implementation details:
- **Refine transformer**: cross-attention between render tokens, observation tokens, **and spec_tokens**; per-part MLP heads for Δpose (se(3) tangent vector), Δsize (parametric deltas), and Δcontact-face params; shared across internal steps.
- **Differentiable renderer**: NvDiffRast (rasterization). Inputs proxy mesh from §3.7 Path A or Path C; re-rendered between internal steps (N_unroll renders per call).
- **Training**: the same N_unroll-step unrolling at train time (with full backprop through the unrolled steps); curriculum hardens w_contact / w_size over training; trained on the **full Stage-1 noise distribution including large-error samples** so the bounded unrolling is robust to bad initialization (R7, R25).
- **No multi-hypothesis selection at inference** (K=1 from Stage 1 + uncertainty); training may still sample K=4 from flow matching for diversity, but only the lowest-uncertainty one is selected during inference.
- **Privileged train-time supervision**: when instance CAD is available, Stage 3 also compares against a *ground-truth* render from the true instance CAD. At inference only the proxy render is used. Gap quantified by ablation 6.

Latency: one network call = N_unroll × (NvDiffRast render ~10 ms + RefineTransformer step ~7 ms) ≈ **~50 ms** at N_unroll=3, vs v0.3's data-dependent 4-iter outer loop × (10+20+500 OCCT) ≈ ~2.1 s per hypothesis × 4 hypotheses ≈ ~8.5 s. The optional K-step fallback, when triggered, adds K × ~17 ms.

### 3.9 Stage 4 — SeamHead (NEW in v0.4)

v0.3's "step 10" assumed `Refined {part_i} → Weld seam (analytical from spec.adjacency_edges)` — a "pure geometric" claim that the 2024-2025 welding literature exposes as 5-7 independent sub-problems collapsed into one line. v0.4 replaces this with an explicit two-substage (10a + 10b) head; the third substage (10c, multi-pass) is deferred to future work.

**v0.5 scope clarification (from Codex review):** the SeamHead is a *refiner of weld attributes*, not a *fixer of bad geometry*. It refines weld_type, torch attitude, endpoints and reports uncertainty given a topologically-correct centerline locus from 10a; it does **not** repair a centerline whose *position* is wrong because the upstream part pose was wrong. A wrong centerline is handled by the §2.2 uncertainty gate (escalate to re-scan / laser / human), never silently "corrected" by the MLP. This keeps the claim defensible: SeamHead's contribution is the attribute set required to drive a torch, conditioned on a sound locus — not magical error recovery.

**Substep 10a — Constraint-derived seam from fitted-primitive contact geometry** (~20 ms; *rewritten in v0.5 — no longer a naked mesh boolean*):
- Inputs: fitted parametric primitives from §3.7 + global-optimization output from §3.8 (which already enforces contact-face coplanarity / perpendicularity / parallelism as constraints) + `spec.adjacency_edges` (which declares, per edge, *which* primitive faces touch and *what kind* of locus the seam is).
- Step 1 — **analytic contact-locus computation**: for each adjacency edge, the seam is the analytic intersection of the two *declared contact faces* of the fitted primitives, not the mesh-mesh boolean of the whole parts. Examples: plate ⊥ cylinder → contact circle of known radius; two plates at a declared dihedral angle → contact line segment; plate-on-plate lap → boundary edge of the upper plate. Because the contact faces are **regularized toward consistency** by the §3.8 contact penalties (high-weight when the spec declares them), this locus is **topologically well-defined** — a single, correctly-typed curve — even when the parts have residual mm-level pose error elsewhere. Its *position* accuracy, however, still inherits the residual pose/size/contact-face error: a circle whose plane is tilted by 1° of part-pose error is still a circle, just slightly mis-placed. That residual is exactly what the uncertainty estimate (10b) reports and what the precision tier (Step 2 below) corrects.
- Step 2 — **(precision-tier only) observation point-cloud snap refinement**: when a precision-tier laser scan of the seam neighbourhood is available (§1.5), snap the analytic centerline to the concavity ridge / groove bottom in that high-quality point cloud (flatness descriptor + K-means clustering, following the medium-thickness plate seam extraction work, Sciencedirect 2025). On the coarse tier this step is skipped — RGB-D depth is too noisy to be a reliable ridge detector, and a wrong snap is worse than the analytic locus.
- Output: per-edge 3D polyline + neighbourhood geometric descriptors (dihedral angle, surface normals, gap, parallelism) + a flag for whether precision-tier refinement was applied.
- Why this is more robust: the failure mode of the v0.4 boolean (empty / fragmented / multi-branch curve when ε < upstream error) is gone — the locus is derived from constrained, parameterized faces, so it degrades gracefully (the *position* of the circle/line may be off by the part-pose error, but it is always a single, topologically-correct curve, which the SeamHead and the uncertainty gate can then reason about).

**Substep 10b — SeamHead network** (~10 ms):
- Inputs: per-edge geometric descriptors + `spec.weld_meta` prior (weld_type prior, torch_constraints, is_load_bearing).
- Outputs (per adjacency edge):
  - `centerline`: refined 3D polyline + per-vertex confidence
  - `weld_type`: 6-way classification ∈ {fillet, butt, lap, corner, edge, groove}
  - `groove_geometry`: parametric groove specs (only when type=groove): {V_angle, root_opening, root_face}
  - `torch_attitude`: (deflection, elevation, rotation) — work + travel + roll angles
  - `endpoints`: (start_3d, end_3d) — handles termination at edges/holes/transitions
  - `is_multi_pass`: bool flag (consumed by future 10c)
- Architecture: per-edge MLP (~200K params) + analytic dihedral-angle computation for torch_attitude initialization (overridden by MLP delta).
- Training data: GT auto-generated by §3.3 Stage-0 from `spec.weld_meta` annotations (no human labeling).
- Training losses: L_centerline (Chamfer to GT polyline), L_type (CE), L_attitude (per-axis MSE), L_endpoint (L2). All combined in §3.10.
- Precedents: WeldNet (Expert Systems w. Apps 2024, 99.6% type accuracy), modified YOLOv5 weld classifier (Sciencedirect 2024, 18 ms 100% precision), torch posture via neighborhood centroid (Sciencedirect 2024).

**Substep 10c — Multi-pass / multi-layer planning** (deferred to future work):
- Triggered when `plate_thickness > threshold` (typ. 8 mm for nuclear workpieces).
- Outputs N parallel passes + layer heights per AWS standards.
- v1 paper scope = single-pass only (thin plates < 8 mm). Document as future work; cite Handheld 3D Scanning Multi-Layer Multi-Pass (MDPI Symmetry 2025) for the planned approach.

### 3.10 Training loss — Updated for v0.4

```
L = λ1  · L_part_mask          (EoMT CE + Dice)                                              [Stage 1]
  + λ2  · L_part_pose_FM       (rectified flow matching loss on SE(3); symmetry-quotient)   [Stage 1, REPLACES v0.3 diffusion loss]
  + λ3  · L_part_size          (MSE on size GT from instance CAD; clamped to spec range)    [Stage 1, privileged]
  + λ4  · L_primitive_type     (CE; very low weight, sanity)                                [Stage 1, privileged]
  + λ5  · L_interpart          (no-overlap + adjacency from spec)
  + λ6  · L_object_consist     (parts composed → object pose/size)
  + λ7  · L_refine_pose        (Δpose/Δsize/Δcontact regression, backprop through N_unroll≤4 unrolled steps; hardening-weight curriculum) [Stage 3]
  + λ8  · L_refine_photometric (render vs observation RGB-D)                                 [Stage 3]
  + λ9  · L_refine_gt_render   (vs render from instance CAD)                                 [Stage 3, privileged]
  + λ10 · L_spec_dropout       (random spec-token masking augmentation, see R9)             [training aug]
  + λ11 · L_centerline         (Chamfer distance to GT polyline)                            [Stage 4 / SeamHead]
  + λ12 · L_weld_type          (CE on 6-way weld type classification)                       [Stage 4 / SeamHead]
  + λ13 · L_torch_attitude     (per-axis MSE on deflection/elevation/rotation)              [Stage 4 / SeamHead]
  + λ14 · L_endpoint           (L2 on start/end points)                                     [Stage 4 / SeamHead]
  + λ15 · L_uncertainty        (NLL on Stage 1 uncertainty calibration; Kendall'17 style)   [Stage 1 head]
```

Privileged (instance-CAD-derived) supervision is used only during training. At inference the model consumes RGB-D + DSL spec only. Joint end-to-end training across all stages (DON6D 2024 / IFPN / CRT-6D 2023) — no staged pretraining with frozen sub-modules.

---

## 4. Evaluation Plan

### 4.1 Datasets
- **Omni6DPose** — backbone parity; confirm no regression vs GenPose++.
- **AIWS-Weld** (to be released) — primary benchmark. Three splits / subsets:
  - **seen split** — auto-generated part + weld GT (from instance CAD + spec, §3.3); fair comparison vs baselines.
  - **open-category split** — held-out categories tested spec-only. **(v0.6)** strengthened: in addition to leave-category-out, include **leave-family-out** splits (hold out a whole shape family, not just one member) and **synthetic-DSL-augmented categories** (random valid primitive compositions) so the open-category number is not a 2-category fluke. The paper states explicitly that 6 real categories alone do not prove open-category generality — the synthetic + leave-family-out + cross-dataset evidence carries that claim.
  - **(v0.6; size TBD in v0.7) independent-measurement subset** — real (or high-fidelity-scanned) instances with **laser/CMM-measured centerline + endpoint GT**, captured with the full sensor + calibration stack (§1.5). This is the *non-self-circular* ground truth: auto-GT comes from the same spec the model is prompted with, so a model that merely "reproduces the spec" would score well on it; the measured subset breaks that loop. Headline weld-feature metrics are reported on **both** auto-GT and measured-GT, and the **gap between them is itself a reported result**. **Size decision deferred to P0c** (which builds the rig): a ≥ 2-category × ~5-instance seed is the floor (enough to *validate*); if scaling to **≥ 4 categories × ≥ 10 instances covering the main weld types** proves feasible, the measured subset is large enough that contribution ⑤ stays "weld feature benchmark"; if not, ⑤ is worded as "large-scale auto-GT benchmark + measured-GT validation subset". Both wordings appear in §2.1 with the decision flagged.
- **BOP-Industrial 2025** (XYZ-IBD + IPD) — secondary benchmark via auto-fit part-level re-annotation; demonstrates method works beyond own dataset (and contributes cross-dataset evidence for the open-category claim).

### 4.2 Metrics (expanded in v0.4)

**Pose accuracy**:
| Metric | Dataset | Role |
|---|---|---|
| Object ADD / IoU | Omni6DPose | Backbone sanity |
| Object ADD | AIWS-Weld (seen split) | Fair comparison vs baselines |
| Part ADD (per primitive) | AIWS-Weld (seen) | Primary contribution validation |
| **Part ADD (open-cat split)** | AIWS-Weld (unseen via spec) | **CatSpec headline metric** |
| Part ADD | BOP-Industrial (XYZ-IBD/IPD) | Cross-benchmark generalization |
| Zero-depth subset ADD | AIWS-Weld | Robustness section |

**Weld feature accuracy** (NEW in v0.4; v0.5 adds Hausdorff + failure rate; v0.6: every row reported at **two sensing tiers** [coarse-only / coarse + precision laser] AND against **two GTs** [auto-GT split / independent measured-GT subset], §4.1):
| Metric | Dataset / split | Role |
|---|---|---|
| **Centerline RMSE** | AIWS-Weld seen + open-cat + measured-GT subset | Per-vertex L2 to GT polyline, RMS; the coarse-tier number is expected to be cm-level — it is a *proposal-quality* metric, not an mm guarantee |
| **Centerline Hausdorff** | AIWS-Weld (all) | Worst-case deviation — catches partial mis-alignment that RMSE averages away |
| **Weld type accuracy** | AIWS-Weld (all) | 6-way classification (fillet/butt/lap/corner/edge/groove) |
| **Torch attitude error** | AIWS-Weld (all) | Per-axis (deflection / elevation / rotation) MAE |
| **Endpoint localization error** | AIWS-Weld (all) | L2 on start + end 3D positions |
| **End-to-end seam-to-pose error** | AIWS-Weld (all) | Composite torch-placement error vs GT (industrial usability proxy); reported separately for coarse-only and coarse+laser |
| **Failure rate** | AIWS-Weld (all) | Fraction of seams with discrete catastrophic error (wrong part-ID match, wrong adjacency edge, centerline off by > 1 weld-bead width) — reported *before* the uncertainty gate and *after* (gate catches → escalates) |
| **Auto-GT vs measured-GT gap** | AIWS-Weld measured-GT subset | **(v0.6)** the difference between scores on auto-generated GT and laser/CMM GT — a large gap means the auto-GT is partly self-circular and the real number is the measured one |
| **Calibration-chain residual** | P0-laser-loop rig | **(v0.6)** end-to-end coordinate-chain error (hand-eye + laser-camera + robot-base + TCP), reported with the precision-tier accuracy claim |

**Deployment cost** (NEW in v0.4):
| Metric | Dataset | Role |
|---|---|---|
| Onboarding cost (min) | AIWS-Weld + BOP-Industrial | Wall-clock cost vs FoundationPose / Any6D / ZeroPose |
| DSL coverage rate | AIWS-Weld + BOP-Industrial | What fraction of workpieces are expressible without Tier-4 neural residual |
| **Per-frame latency (ms)** | AIWS-Weld | Total + per-step breakdown (B2 backbone / B3 flow matching / B4 mesh / B5 R&C / B6 SeamHead) |
| **GPU memory peak (GB)** | AIWS-Weld | For deployment planning |
| **Library construction time (engineer-days)** | One-shot | Reported once for the 30-composite library; supports the no-CAD-expert claim |

### 4.3 Baselines
- **GenPose++** — direct predecessor (closed-set category-level)
- **FoundationPose** (CVPR'24) — instance-CAD upper bound + reference-video model-free
- **Any6D** (CVPR'25) — single-anchor model-free
- **ZeroPose** (TCSVT'24) — full-CAD prompted zero-shot
- **OMNI-PoseX** (Apr'26) — text-prompted open-vocabulary
- MegaPose / CosyPose — render-and-compare alternatives
- SAM6D / OnePose++ — cited but not directly compared (different problem regimes)

### 4.4 Ablations (expanded in v0.4)

**CatSpec mechanism**:
1. w/o part decomposition (collapse to whole-object pose).
2. **w/o spec prompt** (replace CatSpec with implicit category embedding) — CatSpec headline ablation.
3. **Spec-shuffle** — wrong (permuted across categories) spec at test time. Measures whether model truly uses spec structure rather than memorizing. **Pilot success gate.**
4. **Spec-dropout schedule** — sweep training-time spec-token masking rate {0, 0.1, 0.3, 0.5}.
5. w/o inter-part consistency loss.
6. w/o privileged CAD supervision (no instance CAD at train time) — LUPI ablation.

**Stage 1 generative model (NEW v0.4)**:
7. **Diffusion vs Flow Matching** — replace SE(3)^K flow matching with classical SE(3) diffusion (GenPose-style). Measures the speed-accuracy trade.
8. **Flow matching steps sweep** {1, 2, 5, 10, 25} — find minimum step count without accuracy loss.
9. **K hypothesis count sweep** {1, 2, 4} — validate that K=1 + uncertainty is sufficient given flow matching's lower variance.

**Stage 3 R&C (NEW v0.4; updated v0.6)**:
10. **N_unroll sweep** {1, 2, 3, 4} for the refiner's internal steps, **plus the optional explicit K-step fallback** {0, 4, 8} — validate that a small fixed N_unroll disentangles R,t,s and matches a data-dependent iterative R&C, and that the fallback is rarely needed. (This subsumes what v0.4 called the "single-pass vs N_iter sweep".)
11. w/o render-and-compare refinement (Stage 3 removed; Stage 1 only).

**Stage 2 mesh assembly (NEW v0.4)**:
12. **Path A (parametric template) vs Path C (FlexiCubes)** on the same workpiece — measures the trade between template-fit accuracy and FlexiCubes voxel-resolution accuracy.
13. **Per-primitive MLP deformer fidelity** — chamfer distance between MLP-deformed mesh and OCCT ground truth across the parameter range. Validates §3.2 Strategy B works.

**SeamHead (NEW v0.4)**:
14. **w/o SeamHead** (use raw geometric intersection from §3.9 Step 10a only, no MLP refinement) — measures the value of the learned head.
15. **w/o weld_meta prior** (drop spec.weld_meta input to SeamHead) — measures whether prior helps.
16. **w/o point cloud snap refinement** in Step 10a — measures the value of observation-driven snapping.

**Library construction (NEW v0.4)**:
17. **Hand-authored composite vs steelpy/Claude Code/PrimitiveAnything-derived** — for at least one composite, compare downstream pose accuracy across construction strategies. Validates the no-CAD-expert claim.

**Error attribution (NEW v0.5 — the three A/Bs the first Codex review asked for)**:
18. **GT part mask vs predicted part mask** — feed Stage-2 the ground-truth part segmentation instead of EoMT++ output. Isolates how much end-to-end seam error is caused by part-segmentation error. If GT-mask still misses the target tolerance, segmentation is not the bottleneck and the open-category zero-shot-segmentation worry is moot.
19. **Correct metric CAD vs unit-bbox CAD** — replace the unit-bbox normalized primitives with the true metric instance CAD (privileged, for diagnosis only). Isolates the size/scale-estimation error contribution.
20. **Sensing tier sweep** — coarse RGB-D single-frame vs coarse RGB-D multi-view fusion (3-5 views, TSDF) vs coarse + precision-tier laser re-scan of the seam neighbourhood. Isolates the depth-sensor error contribution and quantifies the precision ceiling of each tier.
    - Read together, 18-20 decompose the end-to-end centerline error into {segmentation, scale, sensor} so reviewers (and the deployment plan) know which lever actually moves the number.

**Physical lower bound (NEW v0.6 — runs FIRST, in P00, before any heavy training)**:
21. **Oracle-everything (3 variants — see P00 in §7)** — feed GT part mask + correct *metric* CAD + correct contact-face declarations + simulated D455 noise into (a) a GT-initialized local fit, (b) a multi-start global fit, (c) a Monte-Carlo / Cramér-Rao information-theoretic lower bound; read out the seam centerline error for each. The oracle ceiling = min{a, b}, sanity-checked vs the (c) floor. This is the *best the coarse tier can possibly do*; the learned pipeline can only be worse. If even (b)/(c) say > 3-5 mm, no amount of network engineering recovers mm-grade from coarse RGB-D — pivot to the precision-tier story immediately. (The experiment that decides whether the rest of the spec is worth executing as written.)
22. **10a noise sensitivity** — inject {1, 2, 5, 10} mm + {0.5, 1, 2}° noise into part pose/size; measure the analytic contact-locus error of §3.9 10a, stratified by weld type. Quantifies the nonlinear amplification (R15) and tells the uncertainty gate what its thresholds should actually be.

**Robustness**:
23. w/o depth-degraded fallback.
24. **Uncertainty-gate calibration** — predicted σ vs realized error scatter + reliability diagram, *per tier*. Sweep `τ_coarse` ∈ {5, 10, 15, 20} mm (report Tier-2-scan trigger rate vs missed-bad-seam rate — the triage trade) and `τ_precision` ∈ {1, 2, 3, 5} mm (report human-confirm trigger rate vs residual error after escalation — the execution trade). Makes explicit that the coarse gate is triage, not mm certification.
25. *(Stage-3 N_unroll / fallback sweep — see ablation 10 above; listed here for cross-reference because it is also an error-attribution lever: how much of the R,t,s coupling error is removed per internal step.)*

---

## 5. Risk Register

R1 — **Primitive fitting quality.** Some instances may not decompose cleanly (welding deformation, burrs, manufacturing variance). *Mitigation*: Stage-0 pilot measures fit IoU on ≥ 20 instances; if > 20% fall below 0.9 IoU, introduce Tier-4 neural residual or relax acceptance gate.

R2 — **Variable part count.** Different instances of the same category may contain different numbers of primitives. *Mitigation*: EoMT is query-based and natively supports variable part counts; the pose head uses a set-transformer with dynamic queries. Spec declares max-K per category.

R3 — **Seam ground truth availability.** *Mitigation*: compute seam 3D curves analytically from instance-CAD part interfaces (auto-fit boundaries via Stage 0), then project to 2D — fully automated.

R4 — **Privileged-info generalization gap.** Model may overfit to instance-CAD supervision at train time and fail when only category priors + spec are available at test. *Mitigation*: ablation 6 specifically measures this; if gap is too large, introduce stochastic masking of privileged signals during training.

R5 — **Omni6DPose regression.** Heavy architectural changes may break backbone performance. *Mitigation*: P2 gates on "Omni6DPose ADD does not regress"; roll back if violated.

R6 — **Differentiable rendering throughput.** Stage 3 renders the proxy once per internal unrolled step (N_unroll renders per network call) plus the same at train time with backprop. *Mitigation v0.3*: NvDiffRast; cache per-primitive mesh templates; bound the loop. *v0.4 update*: amortized refiner removes the data-dependent outer loop. *v0.6 update*: the cost is fixed and small — N_unroll ≤ 4 (default 3) renders per call at inference, ~50 ms total (§3.8); the optional explicit K-step fallback (off by default) is the only thing that adds variable cost, and only when the uncertainty gate triggers it.

R7 — **Stage-1 initialization quality.** If the Stage-1 pose is > ~30° or > ~10 cm from GT, the bounded N_unroll-step Stage-3 refiner may not pull it in. *Mitigation* (v0.7 — updated for flow matching, no longer "K=4 diffusion"): (a) flow matching's lower sample variance plus the inter-part consistency loss already shrink the bad-init tail vs v0.3's diffusion head; (b) the uncertainty head flags high-σ Stage-1 outputs, which trigger an **uncertainty-gated rerun** with K∈{1,2,4} flow-matching samples and render-loss selection among them; (c) for the still-bad residue, the **optional explicit K-step refinement fallback** (§3.8, K up to 8, off by default) is engaged; (d) anything that survives all of this and is still inconsistent with the adjacency constraints is escalated by the §2.2 gate rather than trusted. Quantified by ablations 8 (flow-matching step sweep), 9 (K sweep), 10 (N_unroll / fallback sweep).

R8 — **Proxy-vs-real gap.** Online-assembled primitive proxy ≠ real workpiece (welding deformation, burrs). *Mitigation*: privileged L_refine_gt_render at train time; ablation 6 quantifies the gap.

R9 — **Spec memorization rather than spec structural use.** Model may learn category-name → memorized output without parsing spec structure. *Mitigation*: ablation 3 (spec-shuffle) is the dedicated check; if shuffled-spec ADD ≈ correct-spec ADD, the spec is decorative. *Counter-design*: train with random spec-token dropout (L_spec_dropout) and a curriculum that includes synthetic categories generated by random DSL composition.

R10 — **Closed primitive vocabulary limits the open-category claim.** "New category" is bounded by the Tier 1-3 primitive vocabulary. *Mitigation*: explicitly scope claim to "categories expressible in the DSL"; report DSL coverage rate (§4.2) on a held-out workpiece set; document Tier 4 neural residual as the future-work vehicle for truly bespoke shapes.

R11 — **DSL authoring cost defeats the 5-min onboarding promise.** *Mitigation*: ship 30 pre-defined composite primitives so 80% of workpieces require no Tier 1-3 DSL writing; ship a Build123d examples library; auto-induction (DeepCAD / PrimitiveAnything-style) is future work.

R12 — **Symmetry handling at the primitive level.** Cylinder C_∞ × cylinder ⇒ joint pose under-determined. *Mitigation*: per-primitive symmetry-group annotation in the DSL; symmetry-aware L_part_pose projects predictions to SE(3)/G before computing loss.

**Added in v0.4 (R13–R21):**

R13 — **Parametric mesh template fidelity at parameter-space edges.** Per-primitive MLP deformer (§3.2 Strategy B) trained on N≈500 samples may extrapolate poorly near the boundaries of size_param ranges. *Mitigation*: training samples must explicitly cover both ends of each parameter range, not only centers; report per-composite chamfer-distance vs OCCT GT distribution; ablation 13 in §4.4 measures this. Fallback: increase N to 1000 for problem composites.

R14 — **FlexiCubes voxel resolution trade-off.** Path C uses 64³ or 128³ voxel grids; mm-level seam precision may require 256³ → 8× memory growth. *Mitigation*: 80% of workpieces go through Path A (no voxelization); Path C only used for free-DSL outliers. Report voxel resolution as hyperparameter; sweep in supplementary if reviewers ask.

R15 — **Pose error nonlinear amplification at near-parallel surfaces.** mm-level part-pose error can yield cm-level seam error when two surfaces are nearly parallel (small dihedral angle → high derivative of intersection position w.r.t. pose). *Mitigation*: train SeamHead with synthetic Stage-1 pose noise; SeamHead uncertainty head reports per-edge confidence; downstream torch planner can accept high-uncertainty edges only after vision-based re-acquisition.

R16 — **Mesh tessellation artifacts produce spurious local intersections.** Path A parametric templates and Path C FlexiCubes both have approximation errors at edges/fillets that may create false-positive intersection curves. *Mitigation*: §3.9 Step 10a tolerance ε filters very-short segments; observation point cloud snap refinement removes off-surface candidates; report tessellation-artifact-IoU metric.

R17 — **Weld type and torch attitude not derivable from pure intersection geometry.** v0.3's "纯几何反推" claim was wrong; type / attitude need either rules or learning. *Mitigation*: §3.9 SeamHead explicitly handles this with WeldNet/YOLOv5-style classification head + dihedral-angle analytic init for torch attitude. Cite WeldNet (99.6% type acc) and Sciencedirect 2024 torch posture work as evidence the sub-problems are solvable.

R18 — **Multi-pass / multi-layer planning out of v1 scope.** Thick-plate welding (> 8 mm) needs N parallel passes + layer height planning that current spec doesn't capture. *Mitigation*: explicit scope limit in §2.3; cite MDPI Symmetry 2025 (handheld 3D scanning multi-layer multi-pass) as future-work template; v1 paper experimental scope = thin plates only.

R19 — **PrimitiveAnything decomposition quality on industrial parts unverified.** PrimitiveAnything was trained on general 3D shapes (HumanPrim 120K), not industrial workpieces. Decomposition IoU on nuclear workpieces unknown. *Mitigation*: P0 7-day pilot includes a 2-day no-CAD-expert sub-pilot that runs PrimitiveAnything on 1-2 nuclear workpiece samples and measures decomposition IoU; if < 0.85, fallback to Claude Code authoring (§3.2 Strategy A2) for those categories.

R20 — **Claude Code first-attempt failure rate on Build123d generation.** Text-to-CadQuery May'25 reports GPT-4o ≈ 90%, Claude ≈ 85% one-shot executable; ~10-15% of generated code fails compile. *Mitigation*: text-to-cad harness has a feedback loop (compile error → re-prompt) that lifts success rate to ~95%; budget 10-15 min/composite of human review time; for repeated failures, fall back to manual Build123d authoring with team support via Claude Code chat.

R21 — **Per-primitive MLP deformer over-fit to training samples.** If MLP capacity is too high relative to N=500 training samples, will memorize rather than learn smooth deformation. *Mitigation*: small architecture (≤ 3 layers, ≤ 256 hidden); add jitter augmentation on training params; monitor validation chamfer-distance during training; cross-check at evenly-spaced parameter combinations not in training.

**Added in v0.5 / corrected in v0.6 (R22–R28, from the two Codex reviews):**

R22 — **Coarse-tier RGB-D cannot reach industrial mm precision; the "mm-error" headline could be misread as a deployment guarantee.** Consumer RGB-D has ~5-10 mm depth noise at the working distance. *Mitigation*: §1.5 makes the two-tier sensing explicit; the paper states clearly that headline mm-error is *measured against dataset GT* (and additionally against the measured-GT subset, R26), not an industrial sub-mm guarantee; the coarse pipeline is positioned as a *proposal generator* (§1.1, §2.1); ablation 20 reports the precision ceiling of coarse-only vs coarse+laser; the precision tier is the documented path to industrial mm execution. P0-oracle (§7) measures the coarse tier's hard ceiling before any heavy training.

R23 — **No truly zero-cost new-category extension; the open-category claim must be precisely scoped.** Codex: a new category still needs parametric CAD, part graph, weld rules, size priors, symmetry definitions — and mm-grade execution always needs a precision-tier scan. *Mitigation* (v0.6): contribution ① claims only that a YAML spec yields a **coarse-tier proposal** at zero *training/CAD-modeling* cost; the **precision-tier scan is per-workpiece-instance** (fixture/manufacturing/thermal variance is per-instance) and is needed for *any* category, onboarded or not — so it is not an extra cost of "new" categories specifically. R10's "categories expressible in the DSL" scoping still applies on top of this. The open-category *headline number* additionally requires synthetic-DSL + leave-family-out + cross-dataset evidence (R28), not just the 2 held-out real categories.

R24 — **Constraint-derived seam (§3.9 10a) is only as good as the spec's contact-face declaration.** If `spec.adjacency_edges` declares the wrong faces or the wrong locus type, the analytic seam is confidently wrong. *Mitigation*: the per-category onboarding includes a visual-render check of every adjacency edge (the seam is overlaid on the rendered assembly and eyeballed); auto-annotation Stage-0 cross-checks the declared locus against the instance-CAD part interface for training categories; the uncertainty gate flags edges whose analytic locus is far from any observed surface concavity; the measured-GT subset (R26) catches systematic locus errors that auto-GT would hide.

R25 — **Stage-3 high-weight contact penalties can drag the assembly to a wrong configuration under bad Stage-1 initialization.** If `w_contact` is large from the first internal step, a far-off Stage-1 pose can be "snapped" onto a contact configuration that satisfies the perpendicular/coplanar penalty but is the wrong assembly. *Mitigation* (consistent with §3.8 — there are **no true hard constraints**): `w_contact` and `w_size` follow a **hardening schedule** — small over the refiner's first internal unrolled step(s), larger by the last step, and ramping up over training; they never reach ∞. The amortized refiner is trained on the full Stage-1 noise distribution including large-error samples so the bounded N_unroll-step path is robust to bad initialization. If the uncertainty gate (§2.2) detects an internally-inconsistent result, it escalates rather than trusting it. Related to R7 (Stage-1 init quality).

R26 — **Self-circular ground truth.** Stage-0 auto-generates GT (part masks, NOCS, sizes, weld centerlines, torch frames) from the *same* spec + contact-face declaration that the model is prompted with at inference; a model that merely "reproduces the spec" would score well, so auto-GT metrics alone do not prove the seam is *physically* right. *Mitigation*: the mandatory independent-measurement subset (§4.1) — ≥ 2 categories, real/high-fidelity-scanned instances with laser/CMM-measured centerline + endpoint GT captured through the full sensor + calibration stack. Headline weld-feature metrics are reported on both auto-GT and measured-GT; the gap is a reported result; P0-laser-loop (§7) builds the rig that produces this subset.

R27 — **Calibration chain is an uncosted error source.** The precision-tier "0.3-1.0 mm in local sensor coords" figure says nothing until hand-eye + laser-camera + robot-base + TCP calibration residuals are folded in; in robot/world coordinates the realistic figure is ~1-3 mm. *Mitigation*: §1.5 names the calibration chain as its own sub-component with its own error budget; P0-laser-loop measures the realized end-to-end coordinate-chain error rather than quoting per-sensor specs; the paper reports the calibration procedure and residuals and never quotes "0.3 mm laser accuracy" as system accuracy.

R28 — **Six real categories cannot support a strong open-category claim.** A 4-train / 2-test split is a pilot, not proof of generality; the SpecEncoder could learn shortcuts specific to a small category family. *Mitigation*: the open-category headline number requires (a) synthetic-DSL-augmented categories (random valid primitive compositions, generated at scale), (b) leave-*family*-out splits (hold out an entire shape family, not one member), and (c) cross-dataset transfer (BOP-Industrial). The paper states the 6-category limitation explicitly and labels the synthetic + leave-family-out + cross-dataset results as the load-bearing evidence for "open-category". Related to R9 (spec memorization), R10 (DSL-bounded vocabulary).

---

## 6. Milestones (reordered in v0.6 — physical lower bounds before heavy training)

| Phase | Window | Deliverable | Gate |
|---|---|---|---|
| **P00 Physical Lower Bound** *(NEW v0.6; oracle strengthened in v0.7 — runs FIRST)* | 2026-05-12 → 2026-05-16 (~4 days) | P0-oracle in **3 variants** (GT-init local fit / multi-start global fit / Monte-Carlo–CRLB lower bound) on GT mask + correct metric CAD + correct contact faces + simulated D455 noise → oracle ceiling = min{a,b} vs (c) floor. P0-10a-noise: inject {1,2,5,10} mm + {0.5,1,2}° pose/size noise, measure analytic contact-locus error by weld type → calibrate τ_coarse/τ_precision. Also: scan-window recall (can the coarse proposal place a Tier-2 scan over the true seam?). | **Hard pivot gate**: if even variant (b)/(c) say > 5 mm, *stop optimizing the coarse pipeline for an mm target* — re-scope around "coarse proposal + precision-tier execution" before P2-P4. If (a) is cm-level but (b)/(c) are mm-level, fix the optimizer first. (The experiment that decides whether the rest of the plan stands as written.) |
| **P0a No-CAD-expert Pilot** | 2026-05-16 → 2026-05-18 (2 days) | steelpy + Claude Code Build123d + PrimitiveAnything pilot on 3 sample workpieces; 1 per-primitive MLP deformer trained | Whole-mesh chamfer (MLP-deformed vs OCCT GT) < 0.5 mm **AND** weld-adjacent metrics OK: contact-face position error, normal error, dihedral error, contact-locus error all within the budget P0-10a-noise says the gate can tolerate (whole-mesh chamfer alone hides errors exactly where the seam lives) |
| **P0b Spec-Shuffle Pilot** | 2026-05-18 → 2026-05-25 (7 days) | DSL v0 (YAML schema + 6 category specs incl. weld_meta); SpecEncoder PoC wired into EoMT-dev; spec-shuffle signal on 1 unseen category | spec-correct part-ADD > spec-shuffle by ≥ 30% relative **AND** absolute thresholds: held-out coarse-tier centerline RMSE < 10 mm, failure rate < 10%, correct-contact-edge recall > 90%. A 30% relative gap at 40 mm absolute error is **not** a pass. |
| P0c Laser-Loop Pilot *(NEW v0.6; two modes split in v0.7 — overlaps P0.5)* | 2026-06 (~1-2 weeks) | **Mode A (hand-held):** ≥ 2 cats × ≥ 5 instances; D455 → hand-held laser → CMM GT *in local frame*; seeds the §4.1 measured-GT subset. **Mode B (arm-mounted):** full chain calibrated (hand-eye + laser-camera + base + TCP) with per-link residuals; D455 → arm-laser → CMM GT *in robot/world coords*. | Mode A: local centerline error < 1 mm. Mode B: world centerline error < 3 mm + calibration-chain residual breakdown reported. If a gate fails, fix the named cause (algorithm/laser-cal for A, calibration chain for B) before downstream work depends on it. |
| P0.5 Library Construction | 2026-05-25 → 2026-06 | 30 composite primitives via §3.2 hybrid strategy; per-primitive MLP deformers; FlexiCubes Path C for free DSL | All 6 AIWS-Weld categories representable; coverage rate measured |
| P1 Dataset | 2026-06 → 2026-07 | 6 AIWS-Weld categories annotated (4 seen + 2 open-cat); **+ leave-family-out splits + synthetic-DSL-augmented categories** (R28); weld_meta per adjacency edge; **+ independent-measurement subset** (≥ 2 categories, laser/CMM GT, R26, building on P0c); BOP-Industrial part-level auto-fit pilot on 1 XYZ-IBD object | Open-cat + leave-family-out + synthetic splits documented; measured-GT subset captured; BOP IoU report; auto + measured weld GT both available |
| **P1.5 Latency microbenchmark** *(NEW v0.7 — before P2)* | 2026-07 (~2 days) | Synthetic-data timing on the deployment-class GPU (RTX 4090 / A6000): backbone (DINOv3+PointNet++), SpecEncoder, Stage-1 flow matching {1,2,5} steps, Stage-2 mesh assembly (Path A / Path C), NvDiffRast render × N_unroll∈{1,2,3,4}, RefineTransformer step, SeamHead, plus data-transfer overhead | Per-component ms numbers in hand; if the naive sum already blows past ~600-800 ms, surface it now and decide what to cut (smaller backbone? fewer flow steps? N_unroll=2?) *before* committing the P2-P4 implementation effort. The 250-500 ms target is sanity-checked here, not first discovered at P4. |
| P2 Spec-conditioned heads + Flow Matching | 2026-07 → 2026-08 | SpecEncoder + spec-conditioned EoMT++ + SE(3)^K flow matching head replace ScaleNet; N_unroll Stage-3 refiner; ablation diffusion vs flow matching on Omni6DPose | Omni6DPose ADD does not regress; flow matching matches diffusion accuracy at < 1/10 latency |
| P3 Main method (joint training) | 2026-08 → 2026-10 | Joint training of all stages 5-11; first AIWS-Weld part-ADD; first open-cat / leave-family-out part-ADD; SeamHead trained on weld feature GT; coarse-tier + precision-tier weld-feature numbers on auto-GT and measured-GT | Open-cat ADD signal > zero; coarse-tier centerline RMSE reported honestly (cm-level is acceptable as a *proposal*); coarse+laser centerline RMSE < target on measured-GT subset; weld type acc > 90%; failure rate < 10% |
| P4 Ablations + Weld Feature Benchmark + BOP | 2026-10 → 2026-11 | Full ablation tables incl. **P0-oracle lower bound, GT-vs-predicted mask, metric-vs-unit CAD, sensing-tier sweep, N_unroll sweep, uncertainty-gate calibration**; weld-feature metrics on AIWS-Weld at both tiers vs both GTs; BOP-Industrial part-ADD on full XYZ-IBD/IPD subset | Main tables complete; error attribution {segmentation, scale, sensor} quantified; latency verified at **250-500 ms/frame on an RTX 4090 / A6000-class GPU** (not "<250 ms on H100"), with per-step breakdown |
| P5 Writing | 2026-11 → 2026-12 | CVPR 2027 submission — headline framed as "open-category coarse part pose + weld-seam proposal; precision tier for mm execution" | — |

Buffer: P4 may absorb depth-degraded fallback as Section 5.3 Robustness.
Compute budget: training 8× H200; deployment/latency claims on a single RTX 4090 / A6000-class GPU (an RTX 3090 is slower and is *not* the budget reference).

---

## 7. Immediate Next Steps — Three Sequential Pilots (v0.6 reorder)

v0.6 puts a **physical lower-bound pilot (P00) first** — before the no-CAD-expert pilot and before any training — because the second Codex review's sharpest point is that if the coarse tier's *oracle* error is already cm-level, no amount of network engineering rescues mm-grade from consumer RGB-D, and the project must re-scope immediately. P0a (no-CAD-expert library) and P0b (spec-shuffle) follow, with upgraded gates. P0c (laser-loop) overlaps P0.5.

### P00 — Physical Lower-Bound Pilot (~4 days, runs FIRST, decides whether the rest stands as written)

The point: measure the *best the coarse tier can possibly do*, with everything else made perfect, before investing in the learned pipeline.

1. **D0-D2 — P0-oracle (three variants — a single least-squares fit can sit in a local optimum and *under*state the ceiling, so the gate uses the *best* of the three):**
   - **(a) GT-initialized local fit** — classical least-squares assembly fit, initialized at the GT pose/size, on depth corrupted by a **calibrated D455 noise model** (axial + lateral noise vs distance, from the sensor datasheet / self-calibration data). Measures how far the fit *drifts* from a perfect start under realistic depth noise.
   - **(b) multi-start global fit** — the same fit from many random initializations, keep the best by data residual. Guards against (a)'s local-optimum risk and gives a tighter upper bound on the achievable centerline error.
   - **(c) information-theoretic lower bound** — a depth-noise Monte-Carlo / Cramér-Rao-style bound: given the D455 noise model, the part geometry, and the contact-locus formula, what is the *theoretical* minimum centerline-error variance? This is the floor no estimator can beat.
   - Take ≥ 20 AIWS-Weld instances with instance CAD; feed all three the **GT part masks + correct metric instance CAD + correct contact-face declarations**. Report analytic seam centerline error (RMSE + Hausdorff, stratified by weld type) for each variant; the **oracle ceiling = min over {a, b}**, sanity-checked against the (c) floor.
2. **D3 — P0-10a-noise.** With the oracle assembly (variant b), inject controlled noise into part pose ({1, 2, 5, 10} mm translation, {0.5, 1, 2}° rotation) and part size ({1, 3, 5}%), and measure how the §3.9 10a analytic contact-locus error responds — i.e. the *gain* from part-pose error to seam error, per weld type and per dihedral angle. This calibrates `τ_coarse` / `τ_precision` and quantifies R15.
3. **D4 — decision gate (on the *best* oracle, variant min{a,b}; sanity-checked vs the (c) floor).**
   - If oracle coarse-tier centerline RMSE ≤ ~3 mm: the coarse pipeline *can* in principle be mm-ish; proceed as written.
   - If ~3-5 mm: borderline; proceed but flag the headline as "low-cm proposal" and lean hard on the precision tier in the paper framing.
   - If > 5 mm (the realistic Codex prediction) **and the (c) floor is also cm-level** (so it is a physics limit, not a fitting artifact): **re-scope now** — the paper's main claim becomes "open-category coarse part pose + weld-seam *proposal*; precision-tier laser scan is required for industrial mm execution", P0c is promoted in priority, and P3's centerline gate is set to a *proposal-quality* threshold (e.g. RMSE < 10 mm, failure rate < 10%), not an mm threshold. (If variant (a) is cm-level but (b) or the (c) floor is mm-level, the problem is the fit, not the physics — fix the optimizer before concluding.)
   - In *all* outcomes, additionally report whether the coarse proposal can **reliably place a Tier-2 laser-scan window over the true seam** (scan-window recall) — even a cm-level proposal is useful if this recall is high; this is the metric that decides whether the proposal-tier story holds (Codex's go/no-go criterion).

### P0a — No-CAD-Expert Pilot (2 days, after P00)

The CatSpec-Pose claim collapses if the team can't actually produce composite primitives without 3D modeling expertise. Verify this empirically before committing to the spec pipeline.

1. **D-2** — Install `steelpy`, `build123d`, `mesh2sdf`, `nv-flexicubes`, `kaolin`, [PrimitiveAnything](https://github.com/PrimitiveAnything/PrimitiveAnything). Verify all import.
2. **D-2** — Generate 3 sample workpieces:
   - One via **steelpy** (e.g., `aisc.W_shapes.W12X26` → trimesh).
   - One via **Claude Code + Build123d** (prompt: "create parametric bellmouth `(r_in, r_out, length, curvature)`").
   - One via **PrimitiveAnything** (input: an existing AIWS-Weld instance STL → primitive decomposition).
3. **D-1** — For one of the three composites (suggest `tube` or `box_with_holes` for simplicity), train a **per-primitive MLP deformer**: sample N=500 (size_params) × Build123d → mesh → train MLP, validate on held-out param combinations.
4. **D-1 decision gate** (upgraded in v0.6): pass requires (a) whole-mesh chamfer (MLP-deformed vs OCCT GT) < 0.5 mm **and** (b) **weld-adjacent metrics within the budget P0-10a-noise established**: contact-face position error, surface-normal error, dihedral error, and contact-locus error at the weld interface. Whole-mesh chamfer alone is *not* sufficient — it averages over the whole part and hides errors exactly where the seam lives. If (a) passes but (b) fails, the deformer needs more samples *near the weld interface* or a higher-resolution canonical mesh there (R13 / R21).

### P0b — Spec-Shuffle Pilot (7 days, after P0a passes)

1. **D0** — write Tier 1-3 DSL YAML schema (~50 lines) including `weld_meta` field per adjacency edge.
2. **D0-D1** — author DSL specs for all 6 existing AIWS-Weld categories using composite primitives from P0a (or fall back to current v0.3 `primitive_library` shapes if P0a composites incomplete). Validate by visual rendering, **overlaying the analytic seam locus on every adjacency edge** (this is also the R24 check).
3. **D2** — implement SpecEncoder: spec → tokens (~200 lines PyTorch).
4. **D2-D3** — wire SpecEncoder into existing EoMT-dev branch as cross-attention KV for query tokens. Keep diffusion pose head from v0.3 (don't switch to flow matching yet — that's P2).
5. **D4-D5** — quick-train: 5 categories train, 1 category held out (suggest `bellmouth` or `square_tube`); evaluate part-ADD + coarse-tier centerline RMSE + failure rate on held-out with correct spec.
6. **D6** — run spec-shuffle ablation: re-evaluate held-out with permuted spec.
7. **D7 — decision gate (upgraded in v0.6)**: pass requires (a) spec-correct part-ADD > spec-shuffle by ≥ 30% relative **and** (b) absolute floors on held-out coarse tier: centerline RMSE < 10 mm, failure rate < 10%, correct-contact-edge recall > 90%. A 30% relative gap at 40 mm absolute error is **not** a pass — it means the spec carries *some* signal but the pipeline is not yet a usable proposal generator; diagnose before P0.5.

Pilot success criterion: **the spec demonstrably drives the model (relative gate) AND the model produces a usable coarse proposal (absolute floors)** — both, not just the first.

### P0c — Laser-Loop Pilot (~1-2 weeks of rig time, overlaps P0.5) — TWO MODES, kept separate

The precision tier and the measured-GT subset (R26, R27) need a real measurement loop. v0.7 splits the rig into two modes because they validate different things — do not conflate them.

**Mode A — hand-held laser wand (cheap, produces *local* measured-GT only):**
1. Pick ≥ 2 categories (suggest one flat — `cover_plate` — and one curved — `bellmouth`); ≥ 5 real instances each (this is the seed; scale-up to ≥ 4 categories × ≥ 10 instances is the P1 stretch goal that decides contribution ⑤'s wording, §4.1).
2. Rig: D455 (coarse) + hand-held laser line scanner + an independent high-precision scanner / CMM for GT.
3. Calibrate laser↔camera and the local frame; **this mode cannot validate hand-eye / robot-base / TCP** — it produces seam GT *in the local sensor frame only*.
4. Loop: D455 proposes the seam neighbourhood → hand-held laser scans it → snap centerline/endpoints to the groove ridge → compare to CMM GT *in the local frame*.
5. Deliverables: precision-tier centerline error in *local sensor* coords; the seed of the §4.1 measured-GT subset. Gate: local centerline error < 1 mm; if not, the laser-snap algorithm or laser↔camera calibration is the problem.

**Mode B — arm-mounted laser (validates the full robot/world chain):**
6. Mount the laser scanner on the robot flange; calibrate the **full chain** — camera intrinsics/extrinsics, laser↔camera, camera/scanner↔flange (hand-eye), base↔world, TCP — and **report each link's residual** (this is the R27 deliverable).
7. Loop: D455 proposes the window → arm moves the laser there → scan → snap → express in robot/world coords → compare to CMM GT *in robot/world coords*.
8. Deliverables: precision-tier centerline error in *robot/world* coords; the calibration-chain residual breakdown. Gate: world centerline error < 3 mm; if not, the calibration chain (not the algorithm) is the problem and gets fixed before anything downstream depends on it.

(Mode A can run first/cheaply to de-risk the laser-snap algorithm; Mode B is the one that backs the §1.5 "1-3 mm in robot/world coords" claim.)

---

## 8. Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-04-23 | Drop Symmetry-Aware Diffusion direction | Insufficient novelty; prior art exists |
| 2026-04-23 | Adopt LUPI framing | Matches actual data availability; no prior work in category-level pose |
| 2026-04-23 | Defer robot demo | Resource uncertainty; story works without it |
| 2026-04-23 | Keep depth-degraded fallback as secondary contribution | Real problem, but not the main story |
| 2026-04-23 | Part annotation via auto-fit (no human labeling) | Instance CAD + primitive library is sufficient; removes labeling bottleneck |
| 2026-04-23 | Adopt FoundationPose-style render-and-compare refinement | Primary precision lever |
| 2026-04-23 | Render against online-assembled primitive proxy (not instance CAD) | Core novelty in v0.2 |
| 2026-04-23 | Codename FORGE | Industrial welding fit |
| **2026-05-10** | **Upgrade to v0.3: CatSpec-Pose as #1 contribution; reframe project as open-category** | 2025 literature audit (RAG-6DPose IROS'25, Any6D CVPR'25, ZeroPose TCSVT'24, XYZ-IBD BOP'25, OMNI-PoseX'26) threatened v0.2 contributions; new open-category framing wins back novelty and connects to OV9D/OMNI-PoseX trend with structured-prompt advantage |
| **2026-05-10** | **Replace flat primitive dict with 4-tier CSG-style DSL (analytic + sweep + modifier + neural residual)** | Closed primitives can't express thickness, swept profiles, fillets, shells; Build123d/OpenCASCADE is the natural backend; tiered fallback preserves 5-min onboarding for common cases |
| **2026-05-10** | **Subsume v0.2 Shape-CAD Retrieval into CatSpec** | Spec already declares primitive type; retrieval becomes confirmation; collision with RAG-6DPose avoided |
| **2026-05-10** | **Add open-category split to AIWS-Weld; add BOP-Industrial part-level auto-fit as secondary benchmark** | Open-cat split needed to back the CatSpec headline claim; BOP needed to avoid "only-your-own-dataset" review reject |
| **2026-05-10** | **Replace `cylinder` primitive with `tube(outer_r, inner_r, length)`** | Wall thickness must be first-class for industrial workpieces (pipes, hollow shafts) |
| **2026-05-10** | **Defer Tier 4 neural residual + auto spec induction (DeepCAD-style) to future work** | Avoid one-paper-doing-everything anti-pattern; phase-1 scope = analytic + sweep + CSG only |
| **2026-05-10** | **30 pre-defined composite primitives as the primary user interface; full DSL as advanced fallback** | Preserves "5 min onboarding" promise without crippling expressiveness |
| **2026-05-10 (v0.4 R1)** | **Replace SE(3) diffusion with SE(3)^K rectified flow matching for Stage 1 pose head** | RFM-Pose Feb'26 + SE(3)-PoseFlow Nov'25 show ~10x sampling-step reduction at equivalent accuracy; resolves the > 30s/frame inference bottleneck of v0.3 |
| **2026-05-10 (v0.4 R1)** | **Reduce K=4 hypothesis to K=1 + uncertainty head** | Flow matching has lower variance than diffusion; multi-hypothesis selection has diminishing returns |
| **2026-05-10 (v0.4 R1)** | **Replace iterative N_iter=4 R&C with single-pass amortized refinement** | RePOSE CVPR'22 + NeFeS CVPR'24 demonstrate one-pass refinement matches iterative on accuracy; eliminates per-iter mesh re-assembly cost |
| **2026-05-10 (v0.4 R1)** | **Joint end-to-end training of all stages (no staged pretraining + freezing)** | DON6D 2024 + IFPN + CRT-6D 2023: error accumulation in cascade is mitigated by joint backprop, not by stage isolation |
| **2026-05-10 (v0.4 R2)** | **Stage 2 split into Path A (parametric mesh templates, SMPL-style) and Path C (FlexiCubes + analytic SDF + fuzzy boolean)** | OCCT/Build123d in inner loop is CPU-bound 50-500ms; SDF min/max gives Pseudo-SDF (CSG on Neural SDF SIGGRAPH-Asia'23); Path A handles 80% of workpieces at ~10ms, Path C is fallback at ~30ms |
| **2026-05-10 (v0.4 R2)** | **Reject DiffCSG (SIGGRAPH-Asia'24) as Path B option** | Conceptually elegant but requires replacing NvDiffRast pipeline; engineering cost not justified given Path A coverage |
| **2026-05-10 (v0.4 R3)** | **Split "step 10 seam derivation" into 10a robust geometric intersection + 10b SeamHead + 10c future multi-pass** | "纯几何反推" was 5-7 problems collapsed into 1; literature (WeldNet 2024, BIM+vision 2025, torch posture 2024, Sciencedirect seam extraction 2025) decomposes into intersection + classification + attitude + endpoints + multi-pass |
| **2026-05-10 (v0.4 R3)** | **Extend YAML spec with `weld_meta` field per adjacency_edge** (weld_type prior, torch_constraints, is_load_bearing) | Required to auto-generate SeamHead training GT and to provide priors at inference |
| **2026-05-10 (v0.4 R3)** | **C5 contribution renamed: "Weld seam mm-error benchmark" → "Weld feature benchmark"** (centerline + type + attitude + endpoints) | Single mm-error metric is insufficient for industrial usability; 4-metric benchmark grounds the claim better |
| **2026-05-10 (v0.4 R3)** | **Defer 10c multi-pass / multi-layer planning to future work; v1 paper scope = thin plates only** | Multi-pass has its own literature (MDPI Symmetry 2025); avoids one-paper-doing-everything anti-pattern |
| **2026-05-10 (v0.4 R4)** | **Replace hand-authored composite primitives with 4-strategy hybrid: steelpy + Claude Code + PrimitiveAnything + per-primitive MLP** | Team has no 3D modeling expertise; cost drops from ~30-50 days to ~12-15 days; no team member needs OCCT/B-rep knowledge |
| **2026-05-10 (v0.4 R4)** | **Per-primitive MLP deformer (NPM paradigm) replaces hand-written analytic deformation function** | NPMs ICCV'21 + NeuraLeaf ICCV'25 + SOMA NVIDIA'26 prove neural deformers work without handcrafted constraints |
| **2026-05-10 (v0.4 R4)** | **Add P0a 2-day no-CAD-expert pilot before P0b spec-shuffle pilot** | The CatSpec story collapses if no-CAD library construction doesn't actually work; verify empirically before committing |
| **2026-05-11 (v0.5)** | **Hand a fully-zero-shot / consumer-single-frame / mm-precision engineering alternative to Codex for independent review; on its verdict, keep the v0.4 architecture and absorb its five fixes** | Codex judged that combination physically unachievable (realistic 15-40 mm + discrete mis-matches) and its recommendations — global assembly optimization, closed-set CAD-derived training, constraint-derived seams, sensor layering, uncertainty gating — converge back onto v0.4; better to harden v0.4 than to pursue an impossible spec |
| **2026-05-11 (v0.5-A)** | **Add explicit two-tier sensing (§1.5): coarse RGB-D for main experiments, optional precision-tier laser for sub-mm** | v0.4 was silent on sensing; consumer RGB-D cannot reach mm-level seam metrology; honest framing protects the headline metric from over-interpretation |
| **2026-05-11 (v0.5-B)** | **Rewrite §3.9 Step 10a: seam = analytic contact locus of fitted parametric primitives, not a mesh-mesh boolean; point-cloud snap demoted to precision-tier-only** | An ε≈1 mm boolean on two parts at 5-15 mm relative error gives empty/fragmented/multi-branch curves; a constraint-derived locus degrades gracefully (right topology, position off by part-pose error) which the SeamHead + uncertainty gate can handle |
| **2026-05-11 (v0.5-C)** | **Frame Stage-3 refinement as amortizing an explicit global assembly-constrained energy (object pose + per-part R,t,s + contact-face params; depth + silhouette + contact + size + weld-graph residuals)** | Makes refinement interpretable and gives seam extraction a constrained foundation; this is the structure the independent review recommended over independent-part-fit-then-intersect |
| **2026-05-11 (v0.5-D)** | **Add an uncertainty-gated escalation loop: σ > τ≈2-3 mm or adjacency-inconsistent residual → multi-view re-scan / precision laser / human confirm, never emit a "mm-grade" result** | v0.4 had uncertainty heads but no decision rule; the gate is the deployment safety valve and makes SeamHead's "no magic error recovery" scope honest |
| **2026-05-11 (v0.5-E)** | **Add error-attribution ablations (GT vs predicted part mask; correct metric CAD vs unit-bbox; sensing-tier sweep) and Hausdorff + failure-rate metrics** | Decomposes end-to-end seam error into {segmentation, scale, sensor}; tells reviewers and the deployment plan which lever actually matters; failure rate surfaces discrete catastrophes that continuous metrics average away |
| **2026-05-11 (v0.5)** | **Tier-qualify the open-category claim (contribution ①): coarse-tier zero-cost, precision-tier needs one-time per-category laser calibration** | Codex (round 1): no truly zero-cost extension exists; the tiered wording keeps the claim true and still well clear of ZeroPose's full-CAD requirement — *superseded by the v0.6 entry below, which corrects the "one-time per-category" wording* |
| **2026-05-11 (v0.6)** | **Second Codex review of v0.5; absorb all of it: downgrade the headline, fix four self-contradictions, close the "acknowledged-not-engineered" gaps, reorder pilots so physical lower bounds run first** | v0.5's edits introduced contradictions and the headline was still over-stated; v0.6 keeps the architecture but makes the claims defensible — the project is now positioned as "open-category coarse part pose + weld-seam proposal; precision-tier sensing for mm execution" |
| **2026-05-11 (v0.6-0)** | **Headline downgrade: the coarse RGB-D pipeline is a *proposal generator* (cm / low-cm), not an mm-grade metrology system; mm execution is the precision tier's job** | Two independent reviews; matches engineering fact; harder to attack; re-states §1.1, RQ, §2.1①, §2.3, §2.1⑤ |
| **2026-05-11 (v0.6-A1)** | **No true hard constraints in Stage 3 — "HARD when spec declares it" → "high-weight when spec declares it", on a hardening penalty schedule** | v0.5 §3.8 said HARD while R25 said penalty schedule; a true hard constraint can be infeasible under bad init; resolves the contradiction and keeps R25 coherent |
| **2026-05-11 (v0.6-A2)** | **Stage-3 refiner = one network call with N_unroll ≤ 4 fixed internal render/compare/Δ steps; optional explicit K-step fallback off by default** | v0.5 said "single forward pass" while R25 referenced "internal sub-iterations"; one step cannot disentangle R,t,s on a near-unit-bbox proxy, a few can; "one network call, fixed compute" preserves the amortization claim without the contradiction |
| **2026-05-11 (v0.6-A3)** | **Precision tier = per-instance / per-fixture runtime laser scan, NOT a one-time per-category calibration** | Fixture variance, manufacturing tolerance, thermal distortion and assembly error are per-instance; a once-per-category calibration cannot cancel them; corrects the v0.5 wording in §2.1①/R23 |
| **2026-05-11 (v0.6-A4)** | **One consistent latency budget: coarse-tier 250-500 ms on an RTX 4090 / A6000-class GPU, to be *verified* in P4, not a guarantee; RTX 3090 slower; precision-tier laser ~1-3 s is deliberately outside the per-frame budget** | v0.5 had ~500 ms / <250 ms / H100 / RTX 3090 all in different places; this picks one story and a verification milestone |
| **2026-05-11 (v0.6-B1)** | **Add an explicit calibration chain (hand-eye / laser-camera / robot-base / TCP) with its own error budget to §1.5; precision-tier accuracy is stated *with* the chain (~1-3 mm in robot/world coords)** | Codex: the calibration chain was an uncosted error source; "0.3 mm laser" is meaningless as system accuracy without it (R27) |
| **2026-05-11 (v0.6-B2)** | **Add a mandatory independent-measurement subset to AIWS-Weld (≥ 2 categories, laser/CMM GT); report all weld-feature metrics on both auto-GT and measured-GT; the gap is itself a result** | Codex: Stage-0 auto-GT and 10a predictions both come from the same spec → auto-GT alone can only prove "the model reproduced the spec" (R26); P0c builds the rig |
| **2026-05-11 (v0.6-B3)** | **Strengthen the open-category protocol: synthetic-DSL-augmented categories + leave-*family*-out splits + cross-dataset transfer are the load-bearing evidence; 6 real categories are explicitly insufficient** | Codex: a 4-train/2-test split is a pilot, not proof; the SpecEncoder could learn small-family shortcuts (R28) |
| **2026-05-11 (v0.6-B4)** | **On the coarse tier, the uncertainty gate is a *triage* rule (which seams need a laser scan), not an mm certification; the 2-3 mm threshold applies to the precision-tier output** | Codex: τ=2-3 mm conflicts with coarse-tier 5-10 mm depth noise; clarifying the gate's job removes the conflict |
| **2026-05-11 (v0.6-C)** | **Reorder pilots: P00 physical lower bound (P0-oracle + P0-10a-noise) runs FIRST; if oracle coarse-tier centerline error > 5 mm, re-scope the paper before P2-P4. Upgrade P0a gate (weld-adjacent metrics, not just whole-mesh chamfer) and P0b gate (absolute floors, not just the 30% relative gap). Add P0c laser-loop pilot.** | Codex: test the physical ceiling before investing in big-model training; whole-mesh chamfer hides seam-edge error; a relative gate with cm-class absolute error is not a usable result |
| **2026-05-11 (v0.7)** | **Third Codex review → "Go, after a small round of fixes"; absorb all six text/protocol items + two engineering suggestions; no architectural change** | v0.6 was implementation-ready; the remaining items were claim/consistency issues a reviewer would catch, cheap to fix now |
| **2026-05-11 (v0.7-1)** | **§1.1: "requires sub-millimeter" → "requires mm-class (sub-mm is the long-run target where the precision tier + calibration chain allow it)"; §1.5 is the single source of truth for the numbers (local sub-mm, robot/world 1-3 mm)** | Codex: §1.1 "sub-mm requirement" contradicted §1.5's "1-3 mm robot/world" — can't imply both |
| **2026-05-11 (v0.7-2)** | **Split the uncertainty-gate threshold into `τ_coarse` (cm-scale triage — which seams need a Tier-2 scan) and `τ_precision` (≈2-3 mm — whether a precision result is executable)** | Codex: a single `τ≈2-3 mm` conflicts with coarse-tier 5-10 mm depth noise; the coarse gate is triage, not mm certification |
| **2026-05-11 (v0.7-3)** | **Rewrite R7 around K=1 flow matching + uncertainty-gated rerun {K=1,2,4} + optional explicit fallback (was "K=4 diffusion samples + render-loss selection")** | The old mitigation contradicted the v0.4 switch to flow matching + K=1 |
| **2026-05-11 (v0.7-4)** | **P00 oracle runs in 3 variants — GT-init local fit / multi-start global fit / Monte-Carlo–CRLB lower bound; gate uses min{a,b} sanity-checked vs the (c) floor; also report scan-window recall** | Codex: a single least-squares fit can sit in a local optimum and *under*state the ceiling — a bad result is only conclusive if (b)/(c) agree; scan-window recall is the real "is the proposal useful" metric |
| **2026-05-11 (v0.7-5)** | **measured-GT subset size deferred to P0c: ~2 cats × 5 instances is the validation floor; ≥ 4 cats × ≥ 10 if feasible keeps "weld feature benchmark" wording for contribution ⑤, else it becomes "large-scale auto-GT benchmark + measured-GT validation subset"** | Codex: 2×5 is too small to carry a benchmark-grade contribution claim; cost is unknown until the P0c rig exists |
| **2026-05-11 (v0.7-6)** | **§3.9 10a: "contact faces were forced consistent" → "regularized toward consistency"; add that the analytic locus is topologically robust but its position still inherits the pose/size/contact-face residual** | Consistency with §3.8's soft-penalty story; avoids overclaiming the locus is positionally exact |
| **2026-05-11 (v0.7-eng)** | **Add P1.5 latency-microbenchmark milestone before P2; split P0c into Mode A (hand-held, local measured-GT) and Mode B (arm-mounted, validates full robot/world chain)** | Codex: don't discover the latency budget is unrealistic at P4; hand-held and arm-mounted laser validate different things and must not be conflated |

---

## 9. Related Work to Survey

(Ordered by relevance to v0.3's claims; ★ = direct head-to-head comparison required.)

**Open-vocabulary / open-category 6D pose** — CatSpec direct comparison:
- ★ OMNI-PoseX (Apr'26) — text-prompted, SO(3)-aware flow matching
- ★ OV9D (CVPR'24) — open-vocab category-level NOCS via DINO+SD
- ★ Open-vocabulary 6D Pose (CVPR'24, Corsetti et al.)
- Horyon (2025) — text-prompt cross-scene matching
- VFM-6D (NeurIPS'24) — foundation-feature category prototype matching

**CAD-prompted / model-free pose** — Stage 3 zero-reference comparison:
- ★ ZeroPose (TCSVT'24) — full-CAD-prompted, DOR pipeline
- ★ Any6D (CVPR'25) — single-anchor model-free
- ★ FoundationPose (CVPR'24) — reference-video model-free
- UA-Pose (CVPR'25) — partial-reference + online completion
- Pos3R (CVPR'25), FoundPose (ECCV'24), CRISP (CVPR'25; test-time adaptation)

**CAD-as-knowledge-base retrieval pose** — collision-avoidance citation:
- ★ RAG-6DPose (IROS'25) — explicitly differentiate "structured parametric retrieval" vs "visual-feature retrieval"

**Render-and-compare pose refinement** — Stage 3 ancestry:
- FoundationPose, MegaPose (CoRL'22), CosyPose (ECCV'20), DeepIM (ECCV'18)

**Diffusion for pose** — Stage 1 ancestry:
- GenPose / GenPose++, 6D-Diff (CVPR'24), Particle-based 6D Diffusion (Dec'24), SE(3)-diffusion (Urain et al.), Diff9D (Feb'25), Lattice-Deformation+Diffusion (May'25), RGB Diffusion Category-Level Pose (Dec'24)

**Closed-set category-level 6D pose** — comparison & predecessors:
- NOCS, GenPose, GenPose++, SAM6D, ZS6D, Sca-pose (2025)

**Primitive abstraction & shape DSLs** — DSL ancestry:
- ShapeAssembly (TOG'20) — DSL for shape generation
- 3D Shape Programs (ICLR'19)
- Tulsiani et al. (CVPR'17), Neural Parts (Paschalidou)
- SuperDec (2025), Light-SQ (SIGGRAPH-Asia'25)
- ★ PrimitiveAnything (SIGGRAPH'25) — cite as future-work auto-induction tool, not direct competitor (does shape, not pose)

**B-rep / CAD program neural networks** — DSL implementation reference:
- DeepCAD (ICCV'21), SkexGen (ICML'22), BrepNet, UV-Net, GenCAD, Free2CAD

**Differentiable rendering**:
- NvDiffRast, PyTorch3D, SoftRas

**Privileged information**:
- Vapnik & Vashist (2009), Lopez-Paz (2015) distillation-as-LUPI

**Articulated-object pose** — boundary case (not the same problem):
- ARTICULATE-ANYTHING (ICLR'25)

**Industrial pose benchmarks** — positioning:
- T-LESS, ITODD, IPD, ★ XYZ-IBD (BOP-Industrial 2025) — cite with "we add part-level + open-cat protocol that BOP-Industrial lacks"

**Welding seam application** — downstream metric grounding:
- YOLOv5 / RT-DETR-based 2D seam detection (Nature SciRep '25, Springer JAMT '25)
- ★ Vision-guided virtual assembly for welding (Sciencedirect 2026) — cite with "we replace per-instance CAD registration with parametric spec prompt"

**Comprehensive survey**:
- Liu et al., *Deep Learning-Based Object Pose Estimation: A Comprehensive Survey*, IJCV 2026 (arXiv:2405.07801)

---

### v0.4 additions to Related Work (organized by audit round)

**Efficient generative pose (Round 1 — replaces v0.3 SE(3) diffusion)**:
- ★ RFM-Pose: Reinforcement-Guided Flow Matching for Fast Category-Level 6D Pose, Feb'26 (arXiv:2602.05257) — primary cite for §3.6 flow matching head
- ★ SE(3)-PoseFlow: 6D Pose Distributions for Uncertainty-Aware Manipulation, Nov'25 (arXiv:2511.01501) — SE(3) manifold flow matching template
- FMPose3D: Monocular 3D Pose via Flow Matching, 2026
- Joint Learning Pose Regression + Diffusion with Score Scaling Sampling (arXiv:2510.04125, Oct'25)
- SANA-Sprint: One-Step Diffusion via Continuous-Time Consistency Distillation (ICCV'25) — alternative if pure flow matching insufficient

**Single-pass amortized refinement (Round 1 — replaces v0.3 N_iter=4 R&C)**:
- ★ RePOSE: Fast 6D Object Pose Refinement via Deep Texture Rendering (CVPR'22) — single-pass refinement template
- ★ NeFeS: Neural Refinement for Absolute Pose Regression with Feature Synthesis (CVPR'24) — single-pass + 54.9% accuracy improvement vs iterative
- DON6D: Decoupled One-Stage Network for 6D Pose Estimation (Sci. Reports 2024) — anti-cascade evidence for joint training
- CRT-6D: Cascaded Refinement Transformers (WACV 2023) — joint-train cascade evidence

**Differentiable mesh / CSG / SDF (Round 2 — Stage 2 backend)**:
- ★ FlexiCubes: Flexible Isosurface Extraction for Gradient-Based Mesh Optimization (SIGGRAPH'23) — Path C mesh extraction; integrated in NVIDIA Kaolin
- ★ A Unified Differentiable Boolean Operator with Fuzzy Logic (2024, arXiv:2407.10954) — fuzzy boolean to avoid Pseudo-SDF
- DiffCSG: Differentiable CSG via Rasterization (SIGGRAPH-Asia'24, arXiv:2409.01421) — considered as Path B but rejected
- Constructive Solid Geometry on Neural Signed Distance Fields (SIGGRAPH-Asia'23) — Pseudo-SDF problem statement
- A Simple Approach to Differentiable Rendering of SDFs (SIGGRAPH-Asia'24)
- Skipping Spheres: Fast Sphere Tracing (CGVC 2024)
- Robust tessellation of CAD models without self-intersections (Oxford 2024) — mesh artifact handling
- Inigo Quilez's analytic SDF library (https://iquilezles.org/articles/distfunctions/) — analytic primitive SDFs

**Parametric mesh templates / Neural Parametric Models (Round 2 — Path A; Round 4 — Strategy B)**:
- ★ SMPL: Skinned Multi-Person Linear Model (TOG'15) — parametric template canonical example
- ★ PHRIT: Parametric Hand Representation with Implicit Template (ICCV'23) — non-human parametric template
- ★ NPMs: Neural Parametric Models for 3D Deformable Shapes (ICCV'21) — learned deformer paradigm
- NeuraLeaf: Neural Parametric Leaf Models with Shape and Deformation Disentanglement (ICCV'25)
- SOMA: Unifying Parametric Human Body Models (NVIDIA 2026) — GPU-Warp accelerated MLP correctives
- DreamCAD: Multi-modal CAD Generation via Differentiable Parametric Surfaces (2026, arXiv:2603.05607) — Bézier patches alternative
- Parameterize Structure with Differentiable Template (2024, arXiv:2410.10399)
- MeshODE: Robust Mesh Deformation framework

**Weld feature derivation (Round 3 — SeamHead literature)**:
- ★ WeldNet: Weld seam type identification (Expert Systems w. Apps 2024) — 99.6% type classification accuracy
- ★ Modified YOLOv5 weld classifier (Sciencedirect 2024) — 18ms 100% precision
- ★ BIM and vision-based welding trajectory planning (Sciencedirect 2025) — full pipeline 1mm accuracy
- ★ Novel weld seam extraction via semantic segmentation + point cloud (Sciencedirect 2025) — point cloud snap refinement template for §3.9 Step 10a
- ★ Welding torch posture planning based on neighborhood centroid (Sciencedirect 2024) — torch attitude via dihedral angle
- Coarse-to-Fine Detection of Multiple Seams for Robotic Welding (arXiv:2408.10710, 2024)
- Weld seam detection 2D + 3D point cloud coarse-to-fine (IJAMT 2025)
- 3DWS: 3D convolutions for weld seam segmentation (J. Intell. Manuf. 2024)
- EfficientWeld — endpoint localization
- Detection and control of welding torch position and posture (Tsinghua 2024)
- CAD integrated automatic recognition of weld paths (IJAMT 2021) — classical CAD intersection baseline
- Handheld 3D Scanning Multi-Layer Multi-Pass Welding (MDPI Symmetry 2025) — future-work template for 10c
- 3D curve weld seam path and posture planning with line laser (Sciencedirect 2025)

**LLM-assisted CAD generation (Round 4 — Strategy A2)**:
- ★ Text-to-CadQuery: A New Paradigm for CAD Generation (May'25, arXiv:2505.06507) — GPT-4o 90% / Claude 85% one-shot success
- CAD-Coder: Text-to-CAD with Chain-of-Thought (May'25, arXiv:2505.19713) — open-source vision-language model
- Don't Mesh with Me: CSG via Code-Gen LLM (Nov'24, arXiv:2411.15279) — fine-tune LLM for CSG
- claude-cad PyPI package — Claude-driven CAD generation
- Text-to-CAD harness (MIT-licensed, https://www.r2clickthrough.com/text-to-cad-open-source-harness/) — Build123d scaffolding for Claude Code

**Industrial standard parts libraries (Round 4 — Strategy A1)**:
- ★ steelpy (https://github.com/evanfaler/steelpy) — AISC W/M/S/HP/WT/MT/ST/Pipe/HSS/L/Double-L Python library
- AISC Shapes Database v16.0 (https://www.aisc.org/publications/steel-construction-manual-resources/16th-ed-steel-construction-manual/aisc-shapes-database-v16.0/) — official structural steel catalog
- TraceParts API (https://github.com/TraceParts) — 100M+ industrial CAD models, free download
- NopSCADlib (https://github.com/nophead/NopSCADlib) — OpenSCAD parametric mechanical parts
- BOSL2 (https://github.com/BelfrySCAD/BOSL2) — Belfry OpenSCAD Library v2

**Auto-induction CAD primitives (Round 4 — Strategy A3)**:
- ★ PrimitiveAnything (SIGGRAPH'25, arXiv:2505.04622) — auto-decomposition STL → primitive assembly
- CADDreamer: CAD Object Generation from Single-view Images (Feb'25, arXiv:2502.20732)

**FoundationPose deployment evidence (latency budget reference)**:
- FoundationPose Isaac ROS — > 120 FPS tracking on Jetson Thor; "compute-intensive" initial estimation
