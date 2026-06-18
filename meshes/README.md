# CAD meshes

Drop your exported SolidWorks STL files here, one per link, then point
`config/geometry.yaml` at them (e.g. `mesh: meshes/upper.stl`).

Suggested exports (from the assemblies in `../../`):
- `column.stl`  — base + cycloidal module (the rotating column)
- `upper.stl`   — upper-arm link (shoulder → elbow)
- `forearm.stl` — forearm link (elbow → wrist)
- `tool.stl`    — wrist + gripper

**Export tips (SolidWorks → File → Save As → STL):**
- Units: **millimetres** (the config is mm; the URDF exporter converts to m).
- Set the part **origin at the joint pivot** so meshes line up with the
  kinematic frames — saves a lot of fiddling.
- Use a medium resolution (fine enough to look right, coarse enough to load fast).
- Also grab each part's **mass** (Tools → Evaluate → Mass Properties) and put
  it in `geometry.yaml` for accurate dynamics.
