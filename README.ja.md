# sidon-rocm-windows

音声復元モデル [Sidon](https://github.com/sarulab-speech/Sidon) を、**Windows ネイティブの ROCm** で AMD GPU 上に動かすための非公式ツールです（WSL・CUDA は使いません）。

> **非公式です。** Sidon の作者や AMD とは関係ありません。2026年10月時点で、Sidon にも AMD にも Windows の AMD GPU で Sidon を動かす公式手順はありません。このリポジトリは公開済みの単一話者モデルを固定し、Windows ROCm で正しく速く動かすための回避策を加えたものです。

英語版の詳しい説明は [README.md](README.md)、技術的な調査結果は [docs/findings.md](docs/findings.md) にあります。

## できること

- 公開 TorchScript [`sarulab-speech/sidon-v0.1`](https://huggingface.co/sarulab-speech/sidon-v0.1) で、公式デモ `app.py` と同じ手順で音声を復元します（出力は 48kHz モノラル）。
- モデルファイルを revision と sha256 で固定し、毎回照合します。
- 既定で MIOpen を使わない畳み込みにします。確認した機種では GPU の処理が約6〜7倍速くなり（初回か2回目以降かで変わります）、Windows の GPU タイムアウト（ドライバーのリセット）も起きませんでした。
- 入力や既存の出力を上書きしません。GPU で失敗しても CPU に切り替えません。NaN を含む出力は書き出しません。

## 確認した環境

GMKtec EVO-X2（Ryzen AI Max+ 395 / Radeon 8060S、gfx1151）、Windows 11 Pro 25H2、ドライバー 32.0.32015.7029（Windows が報告する版）、Python 3.12.10、PyTorch 2.9.1+rocm7.2.1。ほかの環境は未確認です。

## 導入と使い方

手順は [README.md](README.md) の Setup と Usage を参照してください。要点：

```powershell
.\scripts\bootstrap.ps1
# AMD の PyTorch wheel と前処理の依存を入れる（README.md の手順 2〜4）
.\.venv\Scripts\sidon-rocm.exe doctor
.\.venv\Scripts\sidon-rocm.exe fetch-models
.\.venv\Scripts\sidon-rocm.exe restore --input data\input\in.wav --output data\output\out.wav
```

入力は 16bit PCM の WAV です。他の形式は `ffmpeg -i in.m4a -c:a pcm_s16le in.wav` などで変換してください。終了コードは `0` = 成功、`1` = 失敗（出力は書かない）、`2` = 引数の誤り・入力と同じパス・既存の出力です。GPU を多く使う他のアプリは止めておいてください。

## 確認した結果

日本語の音声、`--conv-backend native`、96秒チャンク、新しいプロセスでの1回目の推論時間：

| 入力 | GPU | CPU（基準） | GPU と CPU のコサイン類似度 |
| --- | --- | --- | --- |
| 60秒 | 8.4秒 | 69.3秒 | 0.9999999998 |
| 10分 | 87.5秒 | 741.0秒 | 0.9999999997 |

## ライセンス

このリポジトリは MIT です。Sidon のコードと `sidon-v0.1` モデルは作者により MIT で公開されています。PyTorch と ROCm の wheel は AMD のライセンスに従い、このリポジトリには含みません。モデルファイルも `fetch-models` で Hugging Face から取得し、同梱していません。出力を商用利用する場合は、モデルカードと学習データのライセンスをご自身で確認してください。
