# FROM・・・使用するイメージを指定します
# python3.9のイメージをダウンロード
FROM python:3.9-buster
FROM tiangolo/uvicorn-gunicorn-fastapi
ENV PYTHONUNBUFFERED=1

# WORKDIR・・・RUNコマンドを実行する際の作業ディレクトリを指定します。
# WORKDIR /app

# Starship のインストール
# イメージのサイズを抑えるため、複数指定する場合は「&&」が推奨されています。
RUN sh -c "$(curl -fsSL https://starship.rs/install.sh)" -- --yes && \
    echo 'eval "$(starship init bash)"' >> .bashrc

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# 修正したソースコードをGitリポジトリにアップするために
# コンテナ上にGitをインストールします。
# Visual Studio CodeのRemote Development 拡張機能を利用すると、
# Visual Studio CodeのGitの機能はコンテナ側で動作します。
# そのため、コンテナ側にもGitのインストールが必要になります。
RUN apt-get update && apt-get -y install git

# Create the folder
RUN mkdir -p app
# Paste the folder
COPY ./app app
どのポートを公開するかを明示します。4200ポートの公開を明示します。
# EXPOSE 4200
# CMDは原則は末尾に記載。コンテナ起動時に実行するコマンドとオプションを変更できます。
# CMD ["uvicorn", "api.predictAPI:app", "--host", "0.0.0.0", "--port", "4200", "--reload"]
CMD ["uvicorn", "api.predictAPI:app", "--host", "0.0.0.0", "--reload"]