import numpy as np
import os
import glob

# ======== 設定區 ========
INPUT_ROOT  = r"C:\Study\AutoEncoder\layer_output"
OUTPUT_ROOT = r"C:\Xilinx\Vivado\AutoEncoder\Pattern\LIF\test1"
FILES = ["input.npy", "mem.npy", "spike.npy"]

os.makedirs(OUTPUT_ROOT, exist_ok=True)

# ======== 主程式 ========
lif_dirs = sorted(glob.glob(os.path.join(INPUT_ROOT, "lif*")))  # lif1~lif5, lif_out

for lif_path in lif_dirs:
    lif_name = os.path.basename(lif_path)
    print(f"\n=== 處理層級：{lif_name} ===")

    step_dirs = sorted(glob.glob(os.path.join(lif_path, "step_*")))

    for step_path in step_dirs:
        step_name = os.path.basename(step_path)
        print(f"  → Step: {step_name}")

        # 對應輸出資料夾
        out_dir = os.path.join(OUTPUT_ROOT, lif_name, step_name)
        os.makedirs(out_dir, exist_ok=True)

        # 處理每個檔案
        for fname in FILES:
            in_path = os.path.join(step_path, fname)
            if not os.path.exists(in_path):
                continue

            out_path = os.path.join(out_dir, fname.replace(".npy", "_hex.txt"))
            arr = np.load(in_path, allow_pickle=False)

            print(f"    處理 {fname}：shape={arr.shape}")

            # ---------- spike.npy: 輸出 1-bit ----------
            if "spike" in fname.lower():
                arr_bin = (arr > 0).astype(np.uint8)
                bit_list = [str(int(x)) for x in arr_bin.flatten()]

                print("      模式: 1-bit spike 輸出")
                print("      前 3 筆：", bit_list[:3])

                with open(out_path, "w", encoding="utf-8") as f:
                    for b in bit_list:
                        f.write(b + "\n")

            # ---------- input.npy / mem.npy: int8 HEX ----------
            else:
                arr_clipped = np.clip(arr, -1.0, 1.0)
                arr_int8 = np.round(arr_clipped * 127).astype(np.int8)
                hex_list = [format(x & 0xFF, '02X') for x in arr_int8.flatten()]

                print("      模式: int8 HEX 輸出")
                print("      前 3 筆：")
                for i in range(min(3, len(hex_list))):
                    val = arr.flatten()[i]
                    qv  = arr_int8.flatten()[i]
                    hx  = hex_list[i]
                    print(f"        原值={val: .6f}, int8={qv:4d}, HEX={hx}")

                with open(out_path, "w", encoding="utf-8") as f:
                    for h in hex_list:
                        f.write(h + "\n")

            print(f"      ✅ 已輸出：{out_path}")

print("\n✨ 全部層級與步驟轉換完成！")
