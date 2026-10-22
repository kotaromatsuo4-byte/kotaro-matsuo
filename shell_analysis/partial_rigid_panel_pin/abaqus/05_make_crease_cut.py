from collections import defaultdict, deque
from pathlib import Path


def read_int_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append([int(x) for x in line.split()])
    return rows


def read_int_set(path):
    return {row[0] for row in read_int_rows(path)}


def read_points(path):
    points = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            points.append(tuple(float(x) for x in line.split()))
    return points


def edge_key(a, b):
    return tuple(sorted((a, b)))


def build_edge_faces(faces):
    edge_faces = defaultdict(list)
    for fid, face in enumerate(faces):
        for i, a in enumerate(face):
            b = face[(i + 1) % len(face)]
            edge_faces[edge_key(a, b)].append(fid)
    return edge_faces


def is_closed_crease(crease):
    return len(crease) > 2 and crease[0] == crease[-1]


def unique_in_order(values):
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def crease_vertices(crease):
    if is_closed_crease(crease):
        return crease[:-1]
    return crease


def all_crease_edges(crease_rows):
    return {edge_key(a, b) for crease in crease_rows for a, b in zip(crease, crease[1:])}


def all_crease_vertices(crease_rows):
    return unique_in_order(v for crease in crease_rows for v in crease_vertices(crease))


def incident_face_components(vertex, faces, edge_faces, blocked_edges):
    incident = {fid for fid, face in enumerate(faces) if vertex in face}
    adjacency = {fid: set() for fid in incident}

    for edge, fids in edge_faces.items():
        if vertex not in edge or edge in blocked_edges:
            continue
        local = [fid for fid in fids if fid in incident]
        for fid in local:
            adjacency[fid].update(other for other in local if other != fid)

    components = []
    seen = set()
    for start in sorted(incident):
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        component = set()
        while queue:
            fid = queue.popleft()
            component.add(fid)
            for nxt in adjacency[fid]:
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(component)
    return components


def choose_base_component(vertex, components, ext_verts):
    if len(components) <= 1:
        return 0

    if vertex in ext_verts:
        return max(range(len(components)), key=lambda i: (len(components[i]), max(components[i])))

    return max(range(len(components)), key=lambda i: (len(components[i]), -min(components[i])))


def label_faces_by_origin_duplicates(faces, origin_ids):
    ids_by_origin = defaultdict(list)
    for vid, origin_id in enumerate(origin_ids):
        ids_by_origin[origin_id].append(vid)

    label_by_id = {}
    for origin_id, ids in ids_by_origin.items():
        for copy_index, vid in enumerate(ids):
            if copy_index == 0:
                label_by_id[vid] = str(origin_id)
            else:
                label_by_id[vid] = f"{origin_id}" + "'" * copy_index

    return [[label_by_id[vid] for vid in face] for face in faces]


def cut_by_final_sectors(points, faces, crease_rows, ext_verts):
    seam_edges = all_crease_edges(crease_rows)
    edge_faces = build_edge_faces(faces)
    vertices = all_crease_vertices(crease_rows)

    new_points = list(points)
    replacement_groups = []
    sector_log = []

    for vertex in vertices:
        components = incident_face_components(vertex, faces, edge_faces, seam_edges)
        if len(components) <= 1:
            sector_log.append(
                {
                    "vertex": vertex,
                    "components": [sorted(component) for component in components],
                    "base_component": 0,
                    "copies": [],
                }
            )
            continue

        base_index = choose_base_component(vertex, components, ext_verts)
        copies = []
        for index, component in enumerate(components):
            if index == base_index:
                continue
            new_id = len(new_points)
            new_points.append(points[vertex])
            replacement_groups.append({"old": vertex, "new": new_id, "faces": component})
            copies.append({"new": new_id, "faces": sorted(component)})

        sector_log.append(
            {
                "vertex": vertex,
                "components": [sorted(component) for component in components],
                "base_component": base_index,
                "copies": copies,
            }
        )

    numeric_faces = []
    for fid, face in enumerate(faces):
        numeric = []
        for vertex in face:
            replacement = None
            for group in replacement_groups:
                if vertex == group["old"] and fid in group["faces"]:
                    replacement = group["new"]
                    break
            numeric.append(vertex if replacement is None else replacement)
        numeric_faces.append(numeric)

    origin_ids = list(range(len(points)))
    for group in replacement_groups:
        origin_ids.append(group["old"])

    labelled_faces = label_faces_by_origin_duplicates(numeric_faces, origin_ids)
    return new_points, labelled_faces, numeric_faces, origin_ids, replacement_groups, sector_log


def write_points(path, points):
    lines = [" ".join(f"{x:.10f}" for x in point) for point in points]
    Path(path).write_text("\n".join(lines) + "\n")


def write_labelled_faces(path, faces):
    Path(path).write_text("\n".join(" ".join(face) for face in faces) + "\n")


def write_numeric_faces(path, faces):
    lines = [" ".join(str(vid) for vid in face) for face in faces]
    Path(path).write_text("\n".join(lines) + "\n")


def write_origin_ids(path, origin_ids):
    lines = [f"{vid} {origin_id}" for vid, origin_id in enumerate(origin_ids)]
    Path(path).write_text("\n".join(lines) + "\n")


def write_sector_log(path, sector_log, replacement_groups, crease_rows):
    lines = ["crease_rows"]
    for index, crease in enumerate(crease_rows):
        lines.append(f"line {index} " + " ".join(str(v) for v in crease))

    lines.append("sectors")
    for item in sector_log:
        lines.append(f"vertex {item['vertex']}")
        lines.append(f"base_component {item['base_component']}")
        for index, component in enumerate(item["components"]):
            face_ids = " ".join(str(fid) for fid in component)
            lines.append(f"component {index} faces {face_ids}")
        for copy in item["copies"]:
            face_ids = " ".join(str(fid) for fid in copy["faces"])
            lines.append(f"copy {copy['new']} faces {face_ids}")

    lines.append("duplicates")
    for group in replacement_groups:
        face_ids = " ".join(str(fid) for fid in sorted(group["faces"]))
        lines.append(f"duplicate {group['old']} {group['new']} faces {face_ids}")

    Path(path).write_text("\n".join(lines) + "\n")


def main():
    points = read_points("01_node.dat")
    faces = read_int_rows("02_face.dat")
    crease_rows = read_int_rows("03_crease.dat")
    ext_verts = read_int_set("04_ext_vert_id.dat")

    (
        new_points,
        labelled_faces,
        numeric_faces,
        origin_ids,
        replacement_groups,
        sector_log,
    ) = cut_by_final_sectors(points, faces, crease_rows, ext_verts)

    write_points("06_node_new.dat", new_points)
    write_labelled_faces("05_face_new.dat", labelled_faces)
    write_numeric_faces("06_face_new_numeric.dat", numeric_faces)
    write_origin_ids("06_origin_vert_id.dat", origin_ids)
    write_sector_log("05_cut_steps.dat", sector_log, replacement_groups, crease_rows)
    write_sector_log("05_dup_vert_id.dat", sector_log, replacement_groups, crease_rows)
    print("finish")


if __name__ == "__main__":
    main()
