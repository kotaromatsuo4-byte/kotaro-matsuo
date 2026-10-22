import os
import re
import numpy as np

base = os.path.dirname(os.path.abspath(__file__))

# 保存先フォルダ
result_dir = os.path.join(base, "result")
os.makedirs(result_dir, exist_ok=True)

# 元の節点数（重複節点を除く）
node = np.loadtxt(os.path.join(base, "01_node.dat"))
nnode = node.shape[0]

datfile = os.path.join(base, "202_result.dat")

with open(datfile, encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

mode = None
reading = False
count = 0
mode_data = []

for line in lines:

    # モード番号
    m = re.search(r"E I G E N V A L U E\s+N U M B E R\s+(\d+)", line)
    if m:
        if mode is not None:
            np.savetxt(
                os.path.join(result_dir, f"mode{mode}.dat"),
                np.asarray(mode_data),
                fmt="%.16e",
            )

        mode = int(m.group(1))
        mode_data = []
        reading = False
        count = 0
        continue

    # 節点表開始
    if "NODE FOOT-" in line:
        reading = True
        continue

    if not reading:
        continue

    cols = line.split()

    if len(cols) < 7:
        continue

    try:
        int(cols[0])
    except ValueError:
        continue

    # 重複節点は読み飛ばす
    if count >= nnode:
        reading = False
        continue

    U1 = float(cols[1])
    U2 = float(cols[2])
    U3 = float(cols[3])

    mode_data.append([U1, U2, U3])
    count += 1

# 最後のモード
if mode is not None:
    np.savetxt(
        os.path.join(result_dir, f"mode{mode}.dat"),
        np.asarray(mode_data),
        fmt="%.16e",
    )

print(f"Finished. Output folder: {result_dir}")