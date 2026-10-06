#!/usr/bin/env python
"""
MotionMillion (HF: InternRobotics/MotionMillion) の展開物を、このリポジトリの
データローダが期待するレイアウトに配置する。

期待レイアウト (dataset/dataset_VQ.py, dataset_tokenize.py 等):
    <data_root>/motion_data/vector_272/<name>.npy
    <data_root>/texts/<name>.txt
    <data_root>/split/version1/{tokenizer_96,t2m_60_300}/{train,val,test,all}.txt
    <data_root>/mean_std/vector_272/{mean,std}.npy

tar.gz 内部のディレクトリ構成は配布側の都合で変わり得るため、split に書かれた
名前 (例: "MotionGV/folder0/xxx") を staging 以下のファイルパスの末尾一致で
解決し、hardlink で期待パスに置く。解決できなかった名前を除いた split を
<data_root>/split/version1_avail/ に書き出す (HumanML3D/BABEL/AIST など HF に
含まれないデータが split に載っている場合の救済)。

使い方:
    python prepare/arrange_motionmillion.py --data-root dataset/MotionMillion \
        --staging-dir dataset/MotionMillion/_extracted --raw-dir dataset/MotionMillion/raw_hf
    python prepare/arrange_motionmillion.py --data-root dataset/MotionMillion --verify-only
"""
import argparse
import collections
import errno
import os
import shutil
import sys
import time

MOTION_TYPE = "vector_272"
SPLIT_VERSION = "version1"
AVAIL_VERSION = "version1_avail"
SPLIT_FILES = ("train", "val", "test", "all")


def log(msg):
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


def norm_name(line):
    n = line.strip().replace("\\", "/")
    if n.startswith("./"):
        n = n[2:]
    for ext in (".npy", ".txt"):
        if n.endswith(ext):
            n = n[: -len(ext)]
    return n


# ----------------------------------------------------------------------------
# split / mean_std
# ----------------------------------------------------------------------------
def find_split_root(staging_dir):
    """staging 以下から `<...>/version1/tokenizer_96` を含む split ルートを探す。"""
    for root, dirs, _ in os.walk(staging_dir):
        if os.path.basename(root) == SPLIT_VERSION and any(
            d in dirs for d in ("tokenizer_96", "t2m_60_300")
        ):
            return os.path.dirname(root)
        # staging の深い所 (motion) まで降りない
        if root.count(os.sep) - staging_dir.count(os.sep) > 4:
            dirs[:] = []
    return None


def install_split(staging_dir, data_root):
    dst = os.path.join(data_root, "split", SPLIT_VERSION)
    src_root = find_split_root(staging_dir)
    if src_root is None:
        if os.path.isdir(dst):
            log(f"split: staging に見つからないが {dst} は存在するので続行")
            return dst
        sys.exit("ERROR: split (version1/...) が staging にも data_root にも見つかりません")
    src = os.path.join(src_root, SPLIT_VERSION)
    if os.path.isdir(dst):
        log(f"split: {dst} は既に存在 (上書きしない)")
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst)
        log(f"split: {src} -> {dst}")
    return dst


def install_mean_std(raw_dir, data_root):
    dst_dir = os.path.join(data_root, "mean_std", MOTION_TYPE)
    os.makedirs(dst_dir, exist_ok=True)
    pairs = {"mean.npy": "Mean.npy", "std.npy": "Std.npy"}
    for dst_name, src_name in pairs.items():
        dst = os.path.join(dst_dir, dst_name)
        src = os.path.join(raw_dir, "mean_std", src_name) if raw_dir else None
        if os.path.isfile(dst):
            if src and os.path.isfile(src):
                import numpy as np
                a, b = np.load(dst), np.load(src)
                if a.shape != b.shape or not np.allclose(a, b):
                    log(f"WARNING: {dst} と HF の {src_name} が一致しません (既存を優先)")
            continue
        if src and os.path.isfile(src):
            shutil.copy2(src, dst)
            log(f"mean_std: {src} -> {dst}")
        else:
            log(f"WARNING: {dst} が無く、HF の {src_name} も見つかりません")


# ----------------------------------------------------------------------------
# 名前解決
# ----------------------------------------------------------------------------
def collapse(rel):
    """tar の stem ディレクトリと tar 内部の先頭ディレクトリが同名で二重になる
    (例: MotionLLAMA/fit3d/fit3d/train/...) ので、連続する同名要素を 1 つに潰す。"""
    out = []
    for p in rel.split("/"):
        if not out or out[-1] != p:
            out.append(p)
    return "/".join(out)


class Index:
    """staging 以下のファイル索引。
    by_collapsed: 二重ディレクトリを潰した相対パス -> 実相対パス (拡張子なし)
    by_base     : basename -> [実相対パス]
    """

    def __init__(self):
        self.by_collapsed = {}
        self.by_base = collections.defaultdict(list)
        self.n = 0

    def add(self, rel):
        self.by_collapsed.setdefault(collapse(rel), rel)
        self.by_base[rel.rsplit("/", 1)[-1]].append(rel)
        self.n += 1


def build_index(staging_dir, ext, exclude_dirs=()):
    """staging 以下の *ext を索引化。"""
    index = Index()
    t0 = time.time()
    for root, dirs, files in os.walk(staging_dir):
        dirs[:] = [d for d in dirs if os.path.join(root, d) not in exclude_dirs]
        for f in files:
            if f.endswith(ext):
                rel = os.path.relpath(os.path.join(root, f), staging_dir)[: -len(ext)]
                index.add(rel.replace(os.sep, "/"))
                if index.n % 200000 == 0:
                    log(f"  indexed {index.n} {ext} files...")
    log(f"index {ext}: {index.n} files, {len(index.by_base)} unique basenames ({time.time()-t0:.0f}s)")
    return index


def resolve(name, index):
    """split 名 -> staging 相対パス。
    1) 二重ディレクトリを潰したパスの末尾一致 (例: motion_272rpr/MotionLLAMA/interx/G001/P1)
    2) basename 一致の候補を、先頭ディレクトリ (サブセット名) で絞る
    """
    parts = name.split("/")
    cands = index.by_base.get(parts[-1])
    if not cands:
        return None, "missing"
    # 1) 潰したパスの末尾が name と一致
    suffix = [c for c in cands if collapse(c) == name or collapse(c).endswith("/" + name)]
    if len(suffix) == 1:
        return suffix[0], "ok"
    pool = suffix if suffix else cands
    # 2) name の先頭ディレクトリ (サブセット名, 例 Mirror_MotionGV) を含む候補に絞る。
    #    basename だけの一致で別サブセット (Mirror/非Mirror) のファイルを掴まないよう、
    #    name にディレクトリがある場合は先頭ディレクトリ一致を必須にする。
    if len(parts) > 1:
        head = parts[0]
        pool = [c for c in pool if ("/" + head + "/") in ("/" + c + "/")]
        if not pool:
            return None, "missing"
    if len(pool) == 1:
        return pool[0], "ok"
    return None, "ambiguous"


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        os.link(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            return
        if e.errno != errno.EXDEV:
            raise
        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)


def read_split_names(split_dir):
    """{subset: {split: [names]}}"""
    out = {}
    for subset in sorted(os.listdir(split_dir)):
        sdir = os.path.join(split_dir, subset)
        if not os.path.isdir(sdir):
            continue
        out[subset] = {}
        for s in SPLIT_FILES:
            p = os.path.join(sdir, s + ".txt")
            if os.path.isfile(p):
                with open(p) as f:
                    out[subset][s] = [norm_name(l) for l in f if l.strip()]
    return out


def place(names, index, staging_dir, dst_dir, ext, label):
    """names を解決して dst_dir/<name><ext> に hardlink。戻り値: {name: status}"""
    status = {}
    n_linked = n_exist = 0
    samples = collections.defaultdict(list)
    for i, name in enumerate(names):
        dst = os.path.join(dst_dir, name + ext)
        if os.path.exists(dst):
            status[name] = "ok"
            n_exist += 1
            continue
        rel, st = resolve(name, index)
        if rel is None:
            status[name] = st
            if len(samples[st]) < 5:
                samples[st].append(name)
            continue
        link(os.path.join(staging_dir, rel + ext), dst)
        status[name] = "ok"
        n_linked += 1
        if (i + 1) % 200000 == 0:
            log(f"  {label}: {i+1}/{len(names)}")
    n_bad = sum(1 for v in status.values() if v != "ok")
    log(f"{label}: already={n_exist} linked={n_linked} unresolved={n_bad} / {len(names)}")
    for st, ns in samples.items():
        log(f"  sample {st}: {ns}")
    return status


def unresolved_breakdown(status):
    by_head = collections.Counter()
    for name, st in status.items():
        if st != "ok":
            by_head[name.split("/")[0] if "/" in name else "(no-dir)"] += 1
    return by_head


def write_avail_splits(split_dir, data_root, names_by_subset, m_status, t_status):
    avail_dir = os.path.join(data_root, "split", AVAIL_VERSION)
    for subset, splits in names_by_subset.items():
        need_text = subset.startswith("t2m")
        for s, names in splits.items():
            keep = [
                n for n in names
                if m_status.get(n) == "ok" and (not need_text or t_status.get(n) == "ok")
            ]
            out = os.path.join(avail_dir, subset, s + ".txt")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as f:
                f.write("\n".join(keep) + ("\n" if keep else ""))
            log(f"avail split {subset}/{s}: {len(keep)} / {len(names)}"
                + ("  (motion+text)" if need_text else "  (motion)"))
    log(f"フィルタ済み split: {avail_dir}  -> 学習時は --version {AVAIL_VERSION}/<subset>")


# ----------------------------------------------------------------------------
# verify
# ----------------------------------------------------------------------------
def verify(data_root, n_load=20, check_original=False):
    import numpy as np

    motion_dir = os.path.join(data_root, "motion_data", MOTION_TYPE)
    text_dir = os.path.join(data_root, "texts")
    ok = True
    for fn in ("mean.npy", "std.npy"):
        p = os.path.join(data_root, "mean_std", MOTION_TYPE, fn)
        if os.path.isfile(p):
            log(f"{p}: shape {np.load(p).shape}")
        else:
            log(f"MISSING: {p}"); ok = False

    versions = (SPLIT_VERSION, AVAIL_VERSION) if check_original else (AVAIL_VERSION,)
    for version in versions:
        split_dir = os.path.join(data_root, "split", version)
        if not os.path.isdir(split_dir):
            log(f"(skip) split dir not found: {split_dir}")
            continue
        for subset, splits in read_split_names(split_dir).items():
            for s, names in splits.items():
                n_m = sum(os.path.exists(os.path.join(motion_dir, n + ".npy")) for n in names)
                n_t = sum(os.path.exists(os.path.join(text_dir, n + ".txt")) for n in names)
                log(f"{version}/{subset}/{s}.txt: {len(names)} names | motion {n_m} | text {n_t}")
                if version == AVAIL_VERSION and n_m != len(names):
                    ok = False
                # 先頭 n_load 件をロードして形状確認
                step = max(1, len(names) // n_load)
                shapes = collections.Counter()
                for n in names[::step][:n_load]:
                    p = os.path.join(motion_dir, n + ".npy")
                    if os.path.exists(p):
                        a = np.load(p)
                        shapes[(a.ndim, a.shape[-1] if a.ndim else None, str(a.dtype))] += 1
                log(f"    sampled motion (ndim, last_dim, dtype): {dict(shapes)}")
                if any(k[1] != 272 for k in shapes):
                    log("    WARNING: 最終次元が 272 でないファイルがあります"); ok = False
    log("verify: " + ("OK" if ok else "問題あり (上のログを確認)"))
    return ok


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="dataset/MotionMillion")
    ap.add_argument("--staging-dir", default=None, help="tar.gz を展開した場所")
    ap.add_argument("--raw-dir", default=None, help="HF からの DL 先 (mean_std 取得用)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-original", action="store_true",
                    help="元の split/version1 も全件 stat する (遅い)")
    args = ap.parse_args()

    data_root = os.path.abspath(args.data_root)
    if args.verify_only:
        sys.exit(0 if verify(data_root, check_original=args.verify_original) else 1)

    staging_dir = os.path.abspath(args.staging_dir or os.path.join(data_root, "_extracted"))
    if not os.path.isdir(staging_dir):
        sys.exit(f"ERROR: staging dir not found: {staging_dir}")

    split_dir = install_split(staging_dir, data_root)
    install_mean_std(args.raw_dir, data_root)

    names_by_subset = read_split_names(split_dir)
    all_names = sorted({n for ss in names_by_subset.values() for ns in ss.values() for n in ns})
    log(f"split 名の総数 (重複除去): {len(all_names)}")
    if not all_names:
        sys.exit("ERROR: split が空です")

    split_staging = find_split_root(staging_dir)
    exclude = (split_staging,) if split_staging else ()
    motion_index = build_index(staging_dir, ".npy", exclude_dirs=exclude)
    text_index = build_index(staging_dir, ".txt", exclude_dirs=exclude)

    motion_dir = os.path.join(data_root, "motion_data", MOTION_TYPE)
    text_dir = os.path.join(data_root, "texts")
    m_status = place(all_names, motion_index, staging_dir, motion_dir, ".npy", "motion")
    t_status = place(all_names, text_index, staging_dir, text_dir, ".txt", "text")

    for label, st in (("motion", m_status), ("text", t_status)):
        bd = unresolved_breakdown(st)
        if bd:
            log(f"unresolved {label} by top-level dir: {bd.most_common(15)}")

    # staging 側のレイアウト把握用に数件だけ例示
    ex = [v[0] for v in list(motion_index.by_base.values())[:3]]
    log(f"staging 内 .npy の相対パス例: {ex}")

    write_avail_splits(split_dir, data_root, names_by_subset, m_status, t_status)


if __name__ == "__main__":
    main()
