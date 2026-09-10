# PTH station retrofit guide — adding AOI to the existing lean-tube bench

Companion to the annotated sketch [pth_station_modification.png](pth_station_modification.png)
(original photo: [pth_station_original.png](pth_station_original.png)).
Everything below uses **standard lean-tube (pipe-and-joint) accessories** —
clamps, joints, brackets — no welding, no drilling into the bench frame, fully
reversible.

## What to add, and where (references the sketch)

| # | Sketch callout | Lean-tube addition | Mounting position |
|---|----------------|--------------------|-------------------|
| 1 | Camera + crossbar + FOV (yellow) | One horizontal lean tube across the two rack uprights + camera clamp mount | Across the uprights just above the work-surface zone; camera centered over the jig, pointing straight down |
| 2 | Board jig + fiducials (magenta) | Corner-stop jig (printed or aluminum) + 4 fiducial stickers | Center of the blue ESD mat, clear of the red fixture on the right; fiducials on the jig corners, never under the board |
| 3 | LED bar light ×2, diffuse, ~45° (orange) | 2 LED bars + pivot clamps on the uprights | Left and right uprights, mid-height, angled ~30–45° down/in toward the jig — NOT top-down (glare on glossy parts) |
| 4 | Mini PC — no GPU (cyan) | VESA/shelf bracket under the table | Right side under the table, off the floor, ventilated |
| 5 | Operator monitor (lime) | Monitor arm clamped to the left upright | Eye level, upper-left — angled to the operator, must not block the component-bin shelves |
| 6 | Capture trigger (red) | USB button or foot pedal | Table front edge within easy reach (or floor pedal) |
| 7 | WD 40–60 cm dimension (white) | — | Camera-to-board distance set by the crossbar height; fix it, then never move it |

## Specs (ties back to BOM + data guide)

- **Camera**: manual focus + manual exposure, smallest component ≥ ~15 px in
  frame — for this bench that means a ≥ 12 MP sensor at the 40–60 cm working
  distance shown in the sketch. See the BOM in `docs/execution_plan.md`
  (Phase 1) and resolution guidance in `docs/data_collection_guide.md`.
- **Lighting**: two diffuse LED bars at ~45° cancel shadows cast by tall PTH
  components (connectors, electrolytics) onto small neighbors — the specific
  PTH shadowing risk called out in the execution plan's Risks section.
- **Working distance**: the 40–60 cm dimension is fixed once focus/exposure
  are calibrated; any later change invalidates the golden board and
  `expected_components.json` geometry.

## Wiring & cable routing

- **Power**: one fused power strip under the table feeds mini PC, monitor,
  and LED drivers. Route along the rack's vertical tubes with cable ties;
  never across the work surface.
- **Camera cable**: USB3 (or GigE) from the crossbar down the right upright
  to the mini PC. Keep the run < 3 m for passive USB3; use a locking/screw
  connector at the camera end — bench vibration loosens plain USB.
- **Trigger**: USB button/pedal to the mini PC along the table edge.
- **ESD**: the blue mat is an EPA (ESD-protected area). The camera, lights,
  crossbar, and all mounts stay **outside/above the EPA boundary** — nothing
  added touches the mat except the jig, which must be ESD-safe material
  (dissipative, bonded to the mat's ground point). No ungrounded metal parts
  within 30 cm of exposed boards.

## Installation order (one afternoon)

1. Clamp the crossbar across the uprights at the chosen height.
2. Mount the two LED bars on the uprights, aim ~45° at the mat center.
3. Mount the camera on the crossbar, centered over the future jig position.
4. Place the jig on the mat; stick the 4 fiducials on the jig corners.
5. Mount the mini PC (under table) and the monitor arm + monitor; route
   power and the camera/trigger cables.
6. Software: create the `aoi-app` env, `streamlit run src/app.py`
   (see `docs/execution_plan.md` Phase 2/5).
7. Calibrate: set manual focus/exposure, verify the 20-shot reseat test
   (`src/align_board.py` reports `via fiducials`).
8. Capture the golden board into `data/golden/` — data collection per
   `docs/data_collection_guide.md` starts here.

## Sketch callout → BOM mapping

| Sketch callout | BOM item (`docs/execution_plan.md`, Phase 1) |
|----------------|----------------------------------------------|
| Camera + crossbar + FOV | Camera (webcam or machine-vision) + lens; crossbar from jig-frame/extrusion budget |
| LED bar light ×2 | Lighting: 2× LED bar + diffuser |
| Board jig + fiducials | Jig frame + fiducial marker stickers |
| Mini PC — no GPU | Shop-floor PC (i5/Ryzen 5, 16 GB, no GPU) |
| Operator monitor | Not in BOM — any spare 19–24" display works (~$0–120) |
| Capture trigger | Not in BOM — USB button/pedal (~$10–30) |
| WD 40–60 cm | Determined by lens choice (BOM "lens" row) |
