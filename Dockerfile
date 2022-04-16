# Spring BootアプリをビルドするためMaven/Java8のDockerイメージを利用
FROM maven:3.5-jdk-8-alpine AS builder

# ビルド時のワークディレクトリの設定
WORKDIR /app/alpine

# Docker Run時にjavaコマンドでSpring Bootを起動(Embedded Tomcatを起動)
CMD ["java", "-Djava.security.egd=file:/dev/./urandom", "-jar", "/app.jar"]