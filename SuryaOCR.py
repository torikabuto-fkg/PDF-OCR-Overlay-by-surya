#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  pdf_ocr_overlay.py  ─  PDF を検索可能にする透明テキスト重ね合わせスクリプト
# ============================================================================
#
#  ■ 概要
#    任意の PDF（スキャン PDF・画像 PDF・普通の PDF 何でも可）を
#    Surya OCR で読み取り、元の見た目を一切変えずに「透明テキスト」を
#    各ページに重ねることで、テキスト検索・コピー可能な PDF を生成する。
#
#  ■ 対応入力
#    ・単一 PDF ファイル  (--input report.pdf)
#    ・フォルダ指定で一括 (--input ./pdfs/)
#
#  ■ 使い方
#    # 1 ファイル
#    python pdf_ocr_overlay.py --input exam.pdf --output exam_searchable.pdf
#
#    # フォルダ一括（出力先フォルダに _searchable.pdf を生成）
#    python pdf_ocr_overlay.py --input ./pdfs/ --output ./pdfs_out/
#
#    # ページ指定 + RTX 3080 向けバッチサイズ
#    python pdf_ocr_overlay.py --input exam.pdf --output exam_s.pdf \
#        --pages "1-5,10" --rec_batch 200 --det_batch 18 --dpi 200
#
#  ■ 依存パッケージ (surya-env venv にインストール済み)
#    pip install surya-ocr pikepdf reportlab pypdfium2
#
# ============================================================================

import argparse
import io
import os
import re
import sys
import time
from pathlib import Path

import pikepdf                              # PDF の読み書き・ページ合成
import pypdfium2 as pdfium                  # PDF → 画像レンダリング
from PIL import Image                       # 画像操作
from reportlab.pdfbase import pdfmetrics    # フォント管理
from reportlab.pdfbase.cidfonts import UnicodeCIDFont  # 日本語 CID フォント
from reportlab.pdfgen import canvas as rl_canvas       # PDF キャンバス描画

# ── 日本語対応 CID フォントを登録 ──────────────────────────────
# テキストは透明（見えない）なので書体は関係ないが、
# 日本語をエンコードするために CID フォントが必要。
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiKakuGo-W5"))
_JP_FONT = "HeiseiKakuGo-W5"    # 日本語テキスト用
_LATIN_FONT = "Helvetica"        # 英数テキスト用

# 日本語（ひらがな・カタカナ・漢字・CJK記号）を含むかの判定正規表現
_HAS_JP = re.compile(r"[\u3000-\u9FFF\uF900-\uFAFF]")


# ============================================================================
#  Surya OCR（遅延ロード）
# ============================================================================
# モデルは初回呼び出し時にだけ読み込む。
# これにより --help 表示やパス検証で無駄にGPUメモリを消費しない。

_rec_predictor = None   # テキスト認識モデル
_det_predictor = None   # テキスト検出（行検出）モデル


def _ensure_predictors():
    """Surya の認識/検出モデルを初回だけロードする"""
    global _rec_predictor, _det_predictor
    if _rec_predictor is not None:
        return  # 既にロード済み

    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    print("Loading Surya OCR models...")
    t0 = time.time()

    # FoundationPredictor は Recognition と Detection の共通基盤
    foundation = FoundationPredictor()
    _rec_predictor = RecognitionPredictor(foundation)
    _det_predictor = DetectionPredictor()

    print(f"  Models loaded in {time.time() - t0:.1f}s")


def ocr_image(pil_img: Image.Image) -> list[dict]:
    """
    PIL 画像 1 枚を Surya OCR にかけて行ごとの結果を返す。

    Returns:
        [
            {
                "text": "認識テキスト",
                "bbox": (x1, y1, x2, y2),   # 画像ピクセル座標
                "confidence": 0.95,
            },
            ...
        ]
    """
    _ensure_predictors()

    # Surya は画像のリストを受け取り、画像ごとの結果リストを返す
    results = _rec_predictor([pil_img], det_predictor=_det_predictor)
    page = results[0]  # 1 枚しか渡していないので [0]

    lines = []
    for tl in page.text_lines:
        text = tl.text.strip()
        if not text:
            continue
        lines.append({
            "text": text,
            "bbox": tuple(tl.bbox),        # (x1, y1, x2, y2)
            "confidence": tl.confidence,    # 0.0 ～ 1.0
        })
    return lines


# ============================================================================
#  PDF ページ → PIL 画像
# ============================================================================

def pdf_page_to_image(pdf_path: str, page_idx: int, dpi: int = 200) -> Image.Image:
    """
    pypdfium2 で PDF の指定ページをラスタライズして PIL Image を返す。

    Args:
        pdf_path:  PDF ファイルパス
        page_idx:  0-based ページ番号
        dpi:       レンダリング解像度（高いほど精度↑ だが遅い）
    """
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[page_idx]
    scale = dpi / 72   # PDF のデフォルト解像度は 72 DPI
    bitmap = page.render(scale=scale)
    img = bitmap.to_pil().convert("RGB")
    pdf.close()
    return img


# ============================================================================
#  透明テキストレイヤー生成（ReportLab）
# ============================================================================

def make_text_overlay(
    ocr_lines: list[dict],
    img_width_px: int,
    img_height_px: int,
    pdf_width_pt: float,
    pdf_height_pt: float,
) -> bytes:
    """
    OCR 結果の各行を、元 PDF ページと同サイズの透明テキスト PDF として生成。

    ポイント:
      - textRenderMode(3) = invisible → テキストは見えないが検索・コピー可能
      - 座標変換: 画像座標 (px, 左上原点) → PDF座標 (pt, 左下原点)
      - 水平スケーリングで bbox 幅にテキストをフィットさせる

    Returns:
        PDF バイナリデータ (bytes)
    """
    buf = io.BytesIO()
    c = rl_canvas.Canvas(buf, pagesize=(pdf_width_pt, pdf_height_pt))

    # 画像ピクセル → PDF ポイントへの変換倍率
    sx = pdf_width_pt / img_width_px    # 水平倍率
    sy = pdf_height_pt / img_height_px  # 垂直倍率

    # 塗り・線を完全透明に（念のため）
    c.setFillAlpha(0)
    c.setStrokeAlpha(0)

    for line in ocr_lines:
        text = line["text"]
        x1, y1, x2, y2 = line["bbox"]

        # ── 座標変換 ──
        # 画像: 左上原点、Y↓  →  PDF: 左下原点、Y↑
        pdf_x = x1 * sx                         # 左端
        pdf_y = pdf_height_pt - (y2 * sy)        # Y を反転（bbox 下端 → PDF 下端）
        box_w = (x2 - x1) * sx                  # テキスト領域の幅 (pt)
        box_h = (y2 - y1) * sy                  # テキスト領域の高さ (pt)

        if box_h <= 0 or box_w <= 0:
            continue

        # ── フォント選択 ──
        # 日本語を含む行は CID フォント、英数のみは Helvetica
        font_name = _JP_FONT if _HAS_JP.search(text) else _LATIN_FONT

        # フォントサイズ = bbox 高さの 85%（行間マージンを考慮）
        font_size = box_h * 0.85

        # ── 水平スケーリング ──
        # テキスト描画幅と bbox 幅を合わせるために水平方向を伸縮
        c.setFont(font_name, font_size)
        text_w = c.stringWidth(text, font_name, font_size)
        h_scale = (box_w / text_w * 100) if text_w > 0 else 100

        # ── 透明テキストを描画 ──
        c.saveState()
        c.translate(pdf_x, pdf_y)

        text_obj = c.beginText(0, 0)
        text_obj.setFont(font_name, font_size)
        text_obj.setHorizScale(h_scale)       # 水平スケーリング（%）
        text_obj.setTextRenderMode(3)          # 3 = invisible（見えないけど選択可能）
        text_obj.textOut(text)
        c.drawText(text_obj)

        c.restoreState()

    c.showPage()
    c.save()
    return buf.getvalue()


# ============================================================================
#  1 つの PDF を処理
# ============================================================================

def process_pdf(
    input_path: str,
    output_path: str,
    dpi: int = 200,
    page_range: str = "",
) -> dict:
    """
    メイン処理パイプライン:
      1. 元 PDF の各ページを画像にレンダリング (pypdfium2)
      2. Surya OCR でテキスト＋座標を認識
      3. ReportLab で透明テキスト PDF レイヤーを作成
      4. pikepdf で元ページの上にオーバーレイ（見た目は変わらない）
      5. 検索可能 PDF として保存

    Args:
        input_path:  入力 PDF パス
        output_path: 出力 PDF パス
        dpi:         OCR 用レンダリング解像度
        page_range:  処理ページ指定 ("1,3-5,10")、空文字=全ページ

    Returns:
        {"pages": 処理ページ数, "lines": OCR行数, "time": 処理秒数}
    """
    # 元 PDF を開く
    src = pikepdf.Pdf.open(input_path)
    total_pages = len(src.pages)

    # ページ範囲のパース（指定なし＝全ページ）
    if page_range:
        indices = _parse_page_range(page_range, total_pages)
    else:
        indices = list(range(total_pages))

    print(f"  Input  : {input_path}  ({total_pages} pages)")
    print(f"  Process: {len(indices)} page(s)  (DPI={dpi})")
    print(f"  Output : {output_path}")

    total_time = 0.0
    total_lines = 0

    for count, page_idx in enumerate(indices, 1):
        t0 = time.time()
        print(f"    [{count}/{len(indices)}] Page {page_idx + 1} ...", end=" ", flush=True)

        try:
            # Step 1: PDF ページ → PIL 画像
            img = pdf_page_to_image(input_path, page_idx, dpi=dpi)
            img_w, img_h = img.size

            # Step 2: Surya OCR でテキスト認識
            ocr_lines = ocr_image(img)

            # Step 3: 元ページの物理サイズ (pt) を取得
            page = src.pages[page_idx]
            mbox = page.mediabox
            pdf_w = float(mbox[2]) - float(mbox[0])
            pdf_h = float(mbox[3]) - float(mbox[1])

            # Step 4: 透明テキスト PDF レイヤーを生成
            overlay_bytes = make_text_overlay(
                ocr_lines, img_w, img_h, pdf_w, pdf_h
            )

            # Step 5: 元ページの上にオーバーレイ
            overlay_pdf = pikepdf.Pdf.open(io.BytesIO(overlay_bytes))
            page.add_overlay(overlay_pdf.pages[0])

            elapsed = time.time() - t0
            total_time += elapsed
            total_lines += len(ocr_lines)
            print(f"{len(ocr_lines)} lines  ({elapsed:.1f}s)")

        except Exception as e:
            # エラーが出てもそのページをスキップして続行
            elapsed = time.time() - t0
            total_time += elapsed
            print(f"ERROR: {type(e).__name__}: {e}")

    # 出力ディレクトリを作成して保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    src.save(output_path)
    src.close()

    return {"pages": len(indices), "lines": total_lines, "time": total_time}


# ============================================================================
#  フォルダ一括処理
# ============================================================================

def process_folder(
    input_dir: str,
    output_dir: str,
    dpi: int = 200,
    page_range: str = "",
    suffix: str = "_searchable",
):
    """
    フォルダ内の全 PDF を一括処理する。

    出力ファイル名は「元のファイル名 + suffix + .pdf」になる。
    例: report.pdf → report_searchable.pdf
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # フォルダ内の PDF を名前順で列挙
    pdfs = sorted(in_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    if not pdfs:
        print(f"ERROR: No PDF files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in {input_dir}\n")

    grand_pages = 0
    grand_lines = 0
    grand_time = 0.0

    for i, pdf_path in enumerate(pdfs, 1):
        # 出力ファイル名: 元の名前 + suffix
        out_name = pdf_path.stem + suffix + ".pdf"
        out_path = out_dir / out_name

        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")
        stats = process_pdf(
            input_path=str(pdf_path),
            output_path=str(out_path),
            dpi=dpi,
            page_range=page_range,
        )
        grand_pages += stats["pages"]
        grand_lines += stats["lines"]
        grand_time += stats["time"]
        print()

    # 全体サマリー
    print(f"{'=' * 55}")
    print(f"Total PDFs  : {len(pdfs)}")
    print(f"Total pages : {grand_pages}")
    print(f"Total lines : {grand_lines}")
    print(f"Total time  : {grand_time:.1f}s  "
          f"({grand_time / max(grand_pages, 1):.2f}s/page)")
    print(f"Output dir  : {output_dir}")


# ============================================================================
#  ページ範囲パーサ   "1,3-5,10"  →  [0, 2, 3, 4, 9]
# ============================================================================

def _parse_page_range(spec: str, total: int) -> list[int]:
    """
    カンマ区切り・ハイフン範囲のページ指定文字列を 0-based リストに変換。
    例: "1,3-5,10"  (total=20)  →  [0, 2, 3, 4, 9]
    """
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            # 1-based → 0-based、上限クランプ
            indices.extend(range(a - 1, min(b, total)))
        else:
            idx = int(part) - 1     # 1-based → 0-based
            if 0 <= idx < total:
                indices.append(idx)
    return sorted(set(indices))


# ============================================================================
#  CLI エントリポイント
# ============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="PDF に Surya OCR の透明テキストを重ねて検索可能 PDF を生成する",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 1 ファイル処理
  python pdf_ocr_overlay.py --input exam.pdf --output exam_searchable.pdf

  # フォルダ一括処理
  python pdf_ocr_overlay.py --input ./pdfs/ --output ./pdfs_searchable/

  # ページ指定 + GPU 最適化
  python pdf_ocr_overlay.py --input exam.pdf --output exam_s.pdf \\
      --pages "1-5,10" --dpi 200 --rec_batch 200 --det_batch 18
""",
    )

    ap.add_argument(
        "--input", required=True,
        help="入力パス（PDF ファイル or フォルダ）",
    )
    ap.add_argument(
        "--output", required=True,
        help="出力パス（PDF ファイル or フォルダ）",
    )
    ap.add_argument(
        "--dpi", type=int, default=200,
        help="OCR 用レンダリング解像度 (default: 200, 高精度なら 300)",
    )
    ap.add_argument(
        "--pages", default="",
        help="処理ページ指定 (例: 1,3-5,10)。省略=全ページ",
    )
    ap.add_argument(
        "--suffix", default="_searchable",
        help="フォルダ一括時の出力ファイル名接尾辞 (default: _searchable)",
    )
    ap.add_argument(
        "--rec_batch", type=int, default=0,
        help="RECOGNITION_BATCH_SIZE (0=自動, RTX 3080: 200)",
    )
    ap.add_argument(
        "--det_batch", type=int, default=0,
        help="DETECTOR_BATCH_SIZE (0=自動, RTX 3080: 18)",
    )

    args = ap.parse_args()
    input_path = Path(args.input)

    # 入力パスの存在チェック
    if not input_path.exists():
        print(f"ERROR: Path not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # ── Surya のバッチサイズを環境変数で設定 ──
    # Surya は内部で os.environ からこれらを読み取る
    if args.rec_batch > 0:
        os.environ["RECOGNITION_BATCH_SIZE"] = str(args.rec_batch)
    if args.det_batch > 0:
        os.environ["DETECTOR_BATCH_SIZE"] = str(args.det_batch)

    print()

    # ── フォルダ or 単一ファイルで分岐 ──
    if input_path.is_dir():
        # フォルダ一括モード
        process_folder(
            input_dir=str(input_path),
            output_dir=args.output,
            dpi=args.dpi,
            page_range=args.pages,
            suffix=args.suffix,
        )
    else:
        # 単一 PDF モード
        stats = process_pdf(
            input_path=str(input_path),
            output_path=args.output,
            dpi=args.dpi,
            page_range=args.pages,
        )
        print(f"\n{'=' * 55}")
        print(f"Processed : {stats['pages']} pages")
        print(f"OCR lines : {stats['lines']}")
        print(f"Total time: {stats['time']:.1f}s  "
              f"({stats['time'] / max(stats['pages'], 1):.2f}s/page)")
        print(f"Saved     : {args.output}")


if __name__ == "__main__":
    main()
