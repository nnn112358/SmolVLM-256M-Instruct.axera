# SmolVLM-256M-Instructモデル変換

このモデルはVision EncoderとLanguage Modelの2つの部分に分かれており、それぞれ変換します。

## 一、Vision Encoderの変換
Vision Encoderをonnxにエクスポートし、`pulsar2 build`を通じてaxmodelモデルに変換します。

### 1. 仮想環境の作成
```
conda create -n smolvlm python=3.12 -y
conda activate smolvlm
```

### 2. 依存関係のインストール
```
pip install -r requirements.txt
```
このコードは`transformers==4.49.0`に依存しています。このバージョンの`transformers`を使用することを推奨します。他のバージョンでは互換性の問題が発生する可能性があります。

### 3. モデルのエクスポート（PyTorch -> ONNX）
onnxをエクスポートする前に、huggingfaceまたはmodel scopeからモデルをダウンロードする必要があります。ここではモデルの保存ディレクトリを`../SmolVLM-256M-Instruct/`と仮定します。
`bash export.sh`を実行して直接モデルをエクスポートすることができます。以下は詳細な手順です。

1). モデルを実行し、floatモデルの出力を確認する
```
python run.py ../SmolVLM-256M-Instruct/
```

2). onnxモデルのエクスポート
```
python export.py ../SmolVLM-256M-Instruct/
```
このステップでは、`Qwen2.5-VL-3B-Instruct_vision.onnx`と`Qwen2.5-VL-3B-Instruct_vision.onnx.data`が生成されます。計算グラフと重みパラメータが分離されます。

3). onnxモデルのテスト
```
python test_onnx.py ../SmolVLM-256M-Instruct/
```
このステップでは、onnxモデルを使用してvision encoderモジュールを置き換えて推論を行います。`run.py`の出力と一致するかどうか比較してください。

### 4.モデルの変換（ONNX -> Axera）
モデル変換ツール`Pulsar2`を使用して、ONNXモデルをAxeraのNPU実行用のモデルファイル形式`.axmodel`に変換します。通常、以下の2つのステップが必要です：

- このモデル用のPTQ量子化キャリブレーションデータセットの生成
- `Pulsar2 build`コマンドセットを使用したモデル変換（PTQ量子化、コンパイル）。詳細な使用説明は[AXera Pulsar2ツールチェーンガイド](https://pulsar2-docs.readthedocs.io/zh-cn/latest/index.html)を参照してください。

1). 量子化データセット  
既存のデータセット[imagenet-calib.tar](https://github.com/techshoww/SmolVLM-256M-Instruct.axera/releases/download/v1.0.0/imagenet-calib.tar)をダウンロードするか、自分で画像データをtarでパッケージ化します。
```
tar -cvf calib.tar {your images}
```

2). モデル変換
* 設定ファイルの修正

`config.json`の`calibration_dataset`フィールドを確認し、このフィールドで設定されたパスを前のステップでダウンロードした量子化データセットの保存パスに変更します。

* Pulsar2 build
参考コマンドは以下の通りです：
```
pulsar2 build --input SmolVLM-256M-Instruct_vision.onnx --config config.json --output_dir build-output --output_name SmolVLM-256M-Instruct_vision.axmodel --target_hardware AX650 --compiler.check 0
```
コンパイルが完了したら、ファイル`build-output/SmolVLM-256M-Instruct_vision.axmodel`を`../SmolVLM-256M-Instruct-AX650/`に配置します。

## 二、Language Modelの変換

### 1. Language Modelの変換
コマンドを実行します
```
pulsar2 llm_build --input_path ../SmolVLM-256M-Instruct/ --output_path ../SmolVLM-256M-Instruct-AX650/ --kv_cache_len 1023 --hidden_state_type bf16 --prefill_len 320 --parallel 32 --chip AX650
```
ここで、`prefill_len`の長さは`prefill`フェーズの最大トークン数です。実際の状況に応じてこの値を設定してください。
`kv_cache_len`は`prefill`と`decode`フェーズのトークンの合計長です。

### 2. Language modelからトークン埋め込みを抽出する
コマンドを実行します
```
bash tools/embed_process.sh ../SmolVLM-256M-Instruct/ ../SmolVLM-256M-Instruct-AX650/
```

### 3. 設定ファイルのコピー
```
cp ../SmolVLM-256M-Instruct/*.json ../SmolVLM-256M-Instruct-AX650/
```

これで、モデル変換全体が完了しました。`../SmolVLM-256M-Instruct-AX650/`をAxeraのデバイスにアップロードして実行準備をします。
