# pupiline
`pupiline`はDeepLabCutで瞳孔をトラッキングした後に、追跡点に楕円を当てはめてその面積を計測することで、瞳孔サイズを定量するためのpythonで書かれたプログラムです。  
瞳孔サイズの定量に加えて、動画中に埋め込まれたタイムスタンプを用いて、特定のイベントに関連した瞳孔の動態を簡易的に表示するためのRプログラムも付属しています。

# 使い方
まず、`poetry install --no-dev`を実行して、依存するライブラリをインストールします。インストールに成功したら、`poetry run python pupiline/init.py`を実行します。すると、プロジェクト直下に`data`というディレクトリが作成されます。この`data`には解析対象や解析済みのファイルを配置します。`poetry install --no-dev`がpythonのバージョンによっては失敗する可能性があります。`pupiline`の作成と動作チェックにはpython 3.10.12を使用しているため、必要に応じて同じバージョンを使用することを推奨します。あらかじめ`pyenv install 3.10.12`で該当するバージョンをインストールして、`poetry env use 3.10.12`を実行することで、`pupiline`で使用するpythonバージョンを3.10.12に変更できます。

次にDLCで瞳孔の縁をトラッキングして、各点の座標が保存されているcsvファイルを用意します。座標が記録されたcsvファイルを`pupiline`直下の`data/tracked`というディレクトリに配置します。  
楕円を当てはめる際に面積を計算するだけでなく、既存の動画に楕円を重ねて描画したい場合は、対応する動画ファイルも用意します。動画ファイルは`data/video`に配置します。csvと動画ファイルの名前は一部一致している必要があります。例えば、csvファイルが`hoge-fugaDLC_resnet50_TestDec6shuffle1_500000_labeled.csv`とすると、動画ファイルの名前は`hoge-fuga`を含んでいる必要があります。

ファイルを用意して適切なディレクトリに配置したら、`fit_elliplse.py`というプログラムを実行すると、`data/tracked`にあるcsvファイルを読み込んで瞳孔の大きさを計算します。  
`poetry run python pupiline/fit_ellipse.py`によってプログラムを実行します。デフォルトでは、csvファイル内の`pupil`を名前に含む点を使用して楕円の面積を計算します。その解析の途中経過の動画や、楕円を上書きした動画は生成されません。設定を変更し、対象とする部位を変更する、解析の経過を表示する、そして楕円を上書きした動画を作成する場合には、コマンドライン引数を指定する必要があります。例えば、`poetry run python pupiline/fit_ellipse.py -t eyelid -c -s`とすると、`eyelid`を名前に含む点で解析し、途中経過を表示、楕円を上書きしたファイルを作成します。`-t`は文字列を引数として対象とする体部位を指定します。`-c`は楕円を上書きしたデータを作成するかを設定します。`-s`は解析の途中経過の動画を表示するか設定します。

解析の際には、1ファイルずつ、`data/tracked`内の全てのファイルを解析します。そのため、解析対象外のデータなどは、`data/tracked`からあらかじめ除外しておく必要があります。既に解析したファイルを再解析しないようにするため、一度解析が完了したファイルは解析終了時に`data/analyzed`に自動的に移動されます。さらに、解析結果の瞳孔サイズを記録したファイルは`data/area`に解析終了時に自動的に保存されます。

`fit_ellipse`で解析が完了したら、動画の左上に赤い点がタイムスタンプとして埋め込まれている場合には、`align_pupil.R`でタイムスタンプに関連した瞳孔サイズの動態をプロットできます。  
Rstudioなどで、プログラムを開いて実行してください このプログラムは`tidyverse`と`comprexr`に依存しています。プログラム内で、それぞれのパッケージの存在を確認して、存在しない場合にはインストールします。インストールに失敗した場合には、別途インストールしてください。`comprexr`はgithubのレポジトリから直接インストールするため、`install.packages`は使用できないので、適切な方法を調べてインストールしてください。