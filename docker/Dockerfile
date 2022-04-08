# FROM・・・使用するイメージを指定します
# python3.9のイメージをダウンロード
FROM python:3.9-buster
FROM tiangolo/uvicorn-gunicorn-fastapi
ENV PYTHONUNBUFFERED=1

# WORKDIR・・・RUNコマンドを実行する際の作業ディレクトリを指定します。必須。
WORKDIR /home/projects

# 必須。これをやらないと requirements でインストールできない
# ファイルは全部必要なので、マウント先に全部コピーする
# COPY . /projects
# 全部じゃなくて下記に修正してみたけど問題ないっぽい
COPY etc/requirements.txt /tmp/requirements.txt

# 修正したソースコードをGitリポジトリにアップするために
# コンテナ上にGitをインストールします。
# Visual Studio CodeのRemote Development 拡張機能を利用すると、
# Visual Studio CodeのGitの機能はコンテナ側で動作します。
# そのため、コンテナ側にもGitのインストールが必要になります。
RUN python -m pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt && \
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
# EXPOSE 4200　←　devcontainer.json で指定しているので不要
# CMDは原則は末尾に記載。コンテナ起動時に実行するコマンドとオプションを変更できます。
# CMD ["uvicorn", "api.predictAPI:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
CMD ["uvicorn", "src.api.api:app", "--host", "localhost", "--port", "4200", "--reload"]