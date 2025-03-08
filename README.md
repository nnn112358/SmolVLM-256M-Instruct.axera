# SmolVLM-256M-Instruct.axera
SmolVLM-256M-Instruct DEMO on Axera
- 事前コンパイル済みモデルのダウンロード：[models](https://github.com/techshoww/SmolVLM-256M-Instruct.axera/releases/download/v1.0.0/models.tar.gz)、自分で変換したい場合は[モデル変換](/model_convert/README.md)を参照してください
- [cpp demo](./cpp)
このプロジェクトを再帰的にクローン
```
git clone --recursive https://github.com/techshoww/ax-llm.git
```

## サポートプラットフォーム
- [x] AX650N
- [ ] AX630C

## モデル変換
[モデル変換](./model_convert/README.md)

## ボード上への展開
- AX650NデバイスにはUbuntu22.04が事前インストールされています
- root権限でAX650Nボードデバイスにログインします
- インターネットに接続し、AX650Nデバイスが`apt install`、`pip install`などのコマンドを正常に実行できることを確認します
- 検証済みデバイス：AX650N DEMOボード、AichingPai Pro(AX650N)、AichingPai2(AX630C)

### Python API 実行
#### 必要条件
npu_python_llmをPython環境を持つAX650N開発ボードまたはAichingPai Proにコピーします  
以下のコマンドを実行してpyaxengineをインストールします
```
cd {your path to npu_python_llm}/axengein 
pip install -e .
``` 

#### 環境変数の追加
以下の2行を`/root/.bashrc`（実際に追加するパスは自分で確認してください）に追加し、ターミナルを再接続するか`source ~/.bashrc`を実行します
```
export LD_LIBRARY_PATH={your path to npu_python_llm}/engine_so/:$LD_LIBRARY_PATH
``` 

#### 実行
開発ボード上で以下のコマンドを実行します
```
python3 infer_axmodel.py
```  

**入力**
画像：
![demo.jpg](assets/demo.jpg)
テキスト：
```
Can you describe this image?
```

**出力**  
```
The image depicts a large, historic statue of Liberty, located in New York City. The statue is a prominent landmark in the city and is known for its iconic presence and historical significance. The statue is located on Liberty Island, which is a part of the Empire State Building complex. The statue is made of bronze and is mounted on a pedestal. The pedestal is rectangular and has a weathered look, suggesting it has been in use for a long time. The statue is surrounded by a large, open area, which is likely a plaza or a plaza park.
The statue is surrounded by a large, open space, which is likely a plaza or a plaza park. The space is filled with trees and other greenery, and there is a clear view of the city skyline, which includes the Empire State Building and other notable landmarks. The sky is clear and blue, indicating that it is a sunny day.
In the background, there are tall buildings that are part of the Empire State Building complex. These buildings are modern and have a sleek, contemporary design. The buildings are well-maintained and appear to be part of the same building complex.
The statue is positioned in the center of the image, with the plaza in the foreground. The plaza is surrounded by trees and other greenery, and there is a clear view of the city skyline. The sky is clear and blue, indicating that it is a sunny day.
The statue is made of bronze and is mounted on a pedestal. The pedestal is rectangular and has a weathered look, suggesting it has been in use for a long time. The statue is surrounded by a large, open area, which is likely a plaza or a plaza park.
The statue is surrounded by a large, open space, which is likely a plaza or a plaza park. The space is filled with trees and other greenery, and there is a clear view of the city skyline. The sky is clear and blue, indicating that it is a sunny day.
In summary, the image depicts the Statue of Liberty in New York City, surrounded by a large open plaza with trees and other greenery. The statue is positioned in the center of the image, surrounded by a large open space with trees and other greenery. The sky is clear and blue, indicating a sunny day.<end_of_utterance>
```

## モデル速度  
| ステージ | 時間 |
|------|------|
| Vision Encoder (PTQ U16) | 121 ms  | 
| Prefill |  512ms    |
| Decode  |  11.6 token/s |

実際のモデル処理時間はこれほど大きくありません。Python コードのパフォーマンスが良くないため、[C++ code](./cpp)を使うとより高速になります。

## 技術的な議論
- Github issues
- QQグループ: 139953715
