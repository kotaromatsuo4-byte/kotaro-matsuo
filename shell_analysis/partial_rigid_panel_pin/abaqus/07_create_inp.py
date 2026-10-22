from pathlib import Path


NODE_ID_OFFSET = 1000
ELEM_ID_OFFSET = 5000


def read_float_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([float(value) for value in line.split()])
    return rows


def read_int_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([int(value) for value in line.split()])
    return rows


def inp_node_id(node_id):
    return NODE_ID_OFFSET + node_id


def inp_elem_id(elem_id):
    return ELEM_ID_OFFSET + elem_id


def write_node_inp(path, nodes):
    lines = []
    for node_id, coords in enumerate(nodes):
        x, y, z = coords
        lines.append(
            f"{inp_node_id(node_id)},"
            f"{x:.16e},{y:.16e},{z:.16e}"
        )
    Path(path).write_text("\n".join(lines) + "\n")


def write_elem_inp(path, faces):
    lines = []
    for elem_id, face in enumerate(faces):
        node_ids = ",".join(str(inp_node_id(node_id)) for node_id in face)
        lines.append(f"{inp_elem_id(elem_id)},{node_ids}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_mpc_inp(path, origin_ids):
    lines = ["*MPC"]
    for node_id, origin_id in origin_ids:
        if node_id == origin_id:
            continue
        lines.append(f"PIN, {inp_node_id(node_id)}, {inp_node_id(origin_id)}")
    Path(path).write_text("\n".join(lines) + "\n")


def write_fix_inp(path, fix_rows):
    lines = []
    for node_id, dof_zero_based in fix_rows:
        dof = dof_zero_based + 1
        lines.append(f"{inp_node_id(node_id)},{dof},{dof}")
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    nodes = read_float_rows("06_node_new.dat")
    faces = read_int_rows("06_face_new_numeric.dat")
    origin_ids = read_int_rows("06_origin_vert_id.dat")
    fix_rows = read_int_rows("06_fix.dat")

    write_node_inp("101_node.inp", nodes)
    write_elem_inp("102_elem.inp", faces)
    write_mpc_inp("103_mpc.inp", origin_ids)
    write_fix_inp("104_fix.inp", fix_rows)
    print("finish")


if __name__ == "__main__":
    main()
