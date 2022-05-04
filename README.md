Name
====

Overview
よく使うコマンド
git status
git pull
git add -A
git commit -m"xxx追加"
git push origin main
# ----------------------------
$ git branch #現在のローカルのブランチ確認
* main
$ git fetch #ローカルにリモートブランチの状況を反映させる
$ git branch -a #反映されたか確認
* main
  remotes/origin/HEAD -> origin/main
  remotes/origin/develop
  remotes/origin/main
$ git checkout develop #developブランチに移動
$ git branch #新しくdevelopブランチが作成されたことを確認
* develop
  main
# ----------------------------
【その他よく使うコマンド (Git)】
・$ git stash : 変更を一時的に退避させる。
ブランチを切り替える際によく使う。単にcheckoutしてしまうと他ブランチに作業内容が中途半端に反映されることがある為、その予防策として。
・$ git stash apply : stashした作業内容を今いるブランチに反映する。
・$ git stash drop : stashした作業内容を削除する

## Description

## Demo
・train.yamlを実行する
・欠損値の補完ができたら Airbyte を手動で1回だけ実行させる → BigQuery にデータがインサートされる
・直後に Airbyte「24時間ごとに起動する設定」を Enable に設定する
・その後10分後に predict.yaml を .github/workflows 下に配置してPUSHする

## VS.

## Requirement

## Usage

## Install

## Contribution

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## Author

[tcnksm](https://github.com/tcnksm)