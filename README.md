Name
====

## Overview
なし<br>
<br>
## Description
なし<br>
<br>
## Demo
よく使うコマンド<br>
git status<br>
git pull<br>
git add -A<br>
git commit -m"xxx追加"<br>
git push origin main<br>
<br>
【その他よく使うコマンド (Git)】<br>
git branch # 現在のローカルのブランチ確認<br>
git fetch # ローカルにリモートブランチの状況を反映させる<br>
git branch -a # 反映されたか確認<br>
git checkout develop # developブランチに移動<br>
git branch # 新しくdevelopブランチが作成されたことを確認<br>
git stash # 変更を一時的に退避させる<br>
ブランチを切り替える際によく使う。単にcheckoutしてしまうと他ブランチに作業内容が中途半端に反映されることがある為、その予防策として。<br>
git stash apply # stashした作業内容を今いるブランチに反映する<br>
git stash drop # stashした作業内容を削除する<br>
<br>
## Demo
・GCS のバケットは Airbyte がアクセスできるように一般公開しておく<br>
・github-actions-account のサービスアカウントを作っておく<br>
　┗BigQuery データオーナー　←　必要<br>
　┗BigQuery ユーザー　←　必要<br>
　┗ストレージ管理者<br>
<br>
・train.yamlを実行する<br>
・train.yaml の first_training の待機時間中に Airbyte を手動で1回だけ実行させる → BigQuery にデータがインサートされる<br>
・10分後に Airbyte で「24時間ごとに起動する設定」を Enable に設定する<br>
<br>
## VS.

## Requirement

## Usage

## Install

## Contribution

## Licence

## Author
