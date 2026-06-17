# ARobot — Mechanical & Control Design Notes

Engineering rationale behind the numbers in `config/arm_config.yaml`.
All values are **first-pass estimates** — refine once the body is built and
real masses are known. Then it's just editing the config.

## Drivetrain summary
| Joint | Motor | Reduction | Gross output torque | Usable* |
|-------|-------|-----------|--------------------|---------|
| Base | NEMA 23 (3 N·m) | 15:1 cycloidal | 45 N·m | ~38 N·m |
| Shoulder | NEMA 23 (3 N·m) | 15:1 cycloidal | 45 N·m | ~38 N·m |
| Elbow | NEMA 17 (~0.3 N·m) | 4.5² = 20.25:1 (double belt) | 6.1 N·m | **~4 N·m** |
| Wrist | Waveshare bus servo | internal | ~2.9 N·m | ~2.5 N·m |

\* Usable = gross × drivetrain efficiency (cycloidal ~0.85, dual belt ~0.81)
minus a derate for dynamic torque drop and a stall safety margin.

**The elbow is the binding constraint** — roughly 10× weaker than the shoulder.
Every length/mass decision below is driven by keeping the elbow happy.

## Link-length split (total reach 650 mm)
| Segment | Symbol | Length | Why |
|---------|--------|--------|-----|
| Base height | L1 | 150 mm | structural; tune to base assembly |
| Upper arm | L2 | 300 mm | longest segment — the strong shoulder carries it |
| Forearm | L3 | 230 mm | kept shorter to limit the weak elbow's moment arm |
| Wrist→tip | L4 | 120 mm | wrist servo + gripper |

Design rule: **bias length toward the upper arm.** Moving length from the
forearm to the upper arm trades shoulder torque (abundant) for elbow torque
(scarce).

## Static torque budget (worst case: arm horizontal, 0.3 kg payload)
Assumptions: aluminum links ≈ 0.39 kg/m; elbow module ≈ 0.6 kg; wrist
assembly ≈ 0.25 kg; payload 0.30 kg at the tip.

| Joint | Demand | Usable | Safety factor |
|-------|--------|--------|---------------|
| Shoulder | ≈ 5.5 N·m | ~38 N·m | **~7** (could lift ~2 kg) |
| Elbow | ≈ 1.7 N·m | ~4 N·m | **~2.5** ← the limiter |
| Wrist | ≈ 0.4–1 N·m | ~2.5 N·m | ~2.5+ |

**Payload budget:** 0.30 kg nominal, **0.50 kg ceiling** (elbow-limited).
If you need more payload, shorten L3 or step the elbow up to a NEMA 17 with
higher holding torque.

## Material: PETG joints + aluminum links
PETG yield ≈ 50 MPa, but **printed parts are anisotropic** — layer adhesion is
the weak plane and PETG **creeps** under sustained load. Design to ~12–15 MPa
(safety factor 3–4 on yield) and treat the Z (inter-layer) direction as weakest.

**The base & shoulder cycloidal housings are the critical PETG parts** — they
transmit 45 N·m. Recommendations there:
- **Walls:** ≥ 4–5 perimeters; **infill ≥ 50%** (or solid) around the output
  flange and bearing seats.
- **Metal where it counts:** press-fit steel bearings; metal output shaft;
  **heat-set brass inserts** for every fastener (never thread PETG directly).
- **Spread bolt loads** with large washers / metal backing plates — bolt holes
  are the #1 stress riser and crack initiator.
- **Fillets everywhere** — no sharp internal corners (stress concentration).
- **Print orientation:** lay parts so the main torque becomes in-plane shear
  (XY), not tension across layers (Z).
- The cycloidal **pins/disc** in PETG will wear under contact stress — keep the
  M8 dowel pins (metal) and consider a metal or metal-faced disc for longevity.
- **Heat:** PETG softens near ~80 °C glass transition; NEMA 23s run warm under
  load — keep a thermal gap / vents between motor body and load-bearing walls.

**Aluminum links** are the right call: far stiffer over a 300 mm span, no creep,
and they move mass off the PETG. Keep PETG confined to joint housings, not long
bending members.

## Open items
- Confirm the Waveshare wrist servo model → updates its torque + resolution.
- Real masses after the first print → re-run this budget (the numbers above are
  estimates).
- Decide encoder model for the closed-loop steppers → `encoder_counts_per_rev`.
