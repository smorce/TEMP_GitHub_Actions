# FROM・・・使用するイメージを指定します
# python3.9のイメージをダウンロード
FROM python:3.9-buster
FROM tiangolo/uvicorn-gunicorn-fastapi
ENV PYTHONUNBUFFERED=1

# --mount オプション でバインドマウントする。-p のオプションは必要。 イメージ名は temp_github_actions(小文字じゃないといけない) と自分で決めた。latestというタグをつけた。
# docker run -p 4200:4200 --rm --mount type=bind,src=/Users/tamurasho/Documents/mlops/Docker-test3,dst=/home/projects temp_github_actions:latest
# 多分、上記は設定ファイルとは関係ない。本番環境でコンテナを展開するときに打つコマンド。

# WORKDIR・・・RUNコマンドを実行する際の作業ディレクトリを指定します。必須。
WORKDIR /home/projects

# 必須
# ファイルは全部必要なので、マウント先に全部コピーする
# これをやらないと /home/projects 下は空っぽだった。マウントするだけではなにもファイルが存在しない
COPY . /home/projects

# 修正したソースコードをGitリポジトリにアップするために
# コンテナ上にGitをインストールします。
# Visual Studio CodeのRemote Development 拡張機能を利用すると、
# Visual Studio CodeのGitの機能はコンテナ側で動作します。
# そのため、コンテナ側にもGitのインストールが必要になります。
RUN python -m pip install --upgrade pip && \
    pip install -r etc/requirements.txt && \
    apt-get update && apt-get -y install git && apt-get -y install zsh && \
    wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh || true && \
    chsh -s /usr/bin/zsh && \
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /root/.bash_profile && \
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia' >> /root/.bash_profile && \
    echo 'export PROMPT_COMMAND="history -a"' >> /root/.bash_profile && \
    echo 'export HISTFILE=/root/.zsh-history' >> /root/.bash_profile && \
    echo 'export PYTHONDONTWRITEBYTECODE=1' >> /root/.bash_profile && \
    echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> /root/.bash_profile

# Create the folder
# RUN mkdir -p api
# Paste the folder
# COPY ./api api

# どのポートを公開するかを明示します。4200ポートの公開を明示します。
EXPOSE 4200

# CMDは原則は末尾に記載。コンテナ起動時に実行するコマンドとオプションを変更できます。
# nohup & は無理だった
# localhost じゃ起動できない。0.0.0.0 が正しい
# CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]