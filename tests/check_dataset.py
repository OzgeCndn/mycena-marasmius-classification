"""İndirilen Classes.7z'nin makaledeki Table 4 sayılarıyla eşleşip eşleşmediğini kontrol eder.

Kullanım: python3 -m tests.check_dataset --data-dir data/Classes
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data import load_config

EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/hyperparams.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    expected = cfg["dataset"]["expected_counts"]
    data_dir = Path(args.data_dir)

    total_found, total_expected = 0, 0
    ok = True
    for cname, exp_count in expected.items():
        class_dir = data_dir / cname
        if not class_dir.is_dir():
            print(f"[EKSİK] {cname}: klasör bulunamadı ({class_dir})")
            ok = False
            continue
        found = len([p for p in class_dir.iterdir() if p.suffix in EXTS])
        total_found += found
        total_expected += exp_count
        status = "OK" if found == exp_count else "UYUŞMUYOR"
        if found != exp_count:
            ok = False
        print(f"[{status}] {cname}: bulunan={found}, beklenen (Table 4)={exp_count}")

    print(f"\nToplam: bulunan={total_found}, beklenen=1582 ({total_expected} sınıf bazlı toplam)")
    print("SONUÇ: " + ("Veri seti Table 4 ile eşleşiyor ✔" if ok else "Veri setinde uyuşmazlık var, yukarıdaki listeyi kontrol edin."))


if __name__ == "__main__":
    main()
