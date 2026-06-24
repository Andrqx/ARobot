"""Convert a STEP (CAD) file into an STL mesh for the visualizer.

    python examples/convert_step.py <input.step> <output.stl> [mesh_size_mm]

Uses gmsh (which bundles OpenCASCADE) to read the STEP solid and write a
triangulated surface mesh. gmsh is an optional tool — install with
``pip install gmsh`` — it is NOT required to run the arm in simulation.

STEP files are normally authored in millimetres, which matches the rest of the
stack, so the STL comes out in mm with no scaling.
"""

import sys


def convert(step_path: str, stl_path: str, size_max: float = 4.0) -> int:
    import gmsh

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.3)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 20)
        gmsh.open(step_path)
        gmsh.model.mesh.generate(2)          # surface (2D) mesh of the solid
        gmsh.write(stl_path)
        tags, _ = gmsh.model.mesh.getElementsByType(2)   # type 2 = triangles
        return len(tags)
    finally:
        gmsh.finalize()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    inp, out = sys.argv[1], sys.argv[2]
    size = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0
    n = convert(inp, out, size)
    print(f"{inp} -> {out}  ({n} triangles)")
