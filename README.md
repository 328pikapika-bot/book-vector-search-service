# 图书向量化检索服务

[English](README.en.md) | 简体中文

将图书章节结构化、清洗、切片并向量化，使用 Milvus/MySQL 提供图书内容检索与数据维护接口。

## 主要功能

- 图书和章节内容入库、删除与检索，保留来源字段和过滤条件。
- 包含同书操作锁、并发控制、容量切换及失败补偿相关实现。
- 保留当前独立服务的源码和脚本，不把早期知识图谱/Web 工程混入此仓库。

## 源码与运行入口

唯一服务目录为 `源码/05_编码/skwx_ai-python/social_sciences_vector/`。其中 `main_library_api.py` 是 API 入口，`national_development/` 包含图书操作，`scripts/` 包含预检与验证工具。

Python、Milvus、MySQL、embedding 模型及相应运行时；模型文件按自己的机器路径提供。依赖以服务目录的 requirements.txt 为准。

先进入上述服务目录，核对 `config.py`、`constant.py` 和启动脚本。当前材料中的最新明确发布版本为 v2.2.0，但本次选择的是现有独立服务代码树，未把历史发布包拼成另一套工程。

## 配置

图书 MySQL 配置使用 `BOOK_VECTOR_MYSQL_HOST`、`BOOK_VECTOR_MYSQL_USER`、`BOOK_VECTOR_MYSQL_PASSWORD`、`BOOK_VECTOR_MYSQL_DATABASE`；Milvus、Neo4j 和 ZhipuAI 密钥按 `.env.example` 注入。代码读取环境变量，不会自动加载 `.env`。

配置示例位于源码中的 `.env.example`。真实凭证只保存在本机环境或不跟踪的 `.env` 中。`127.0.0.1` 与 `example.invalid` 为本地/不可解析示例地址，部署前替换为自己的配置。

## 验证与边界

本机模型绝对路径需要自行替换。真实模型加载、MySQL/Milvus 连接、并发及检索数据未在本次验证；历史日志、旧工程和发布包不进入拟公开范围。

本目录于 2026-11-04 完成公开范围整理与静态核验。代码发布不等于已部署或全部功能可运行。配置与依赖准备好后，应再执行适用于目标环境的测试。

公开范围及 GitHub/Gitee 接手规则见[仓库发布说明](仓库发布说明.md)。资料库原有的文档、会话、经验和管理记录继续保留在本目录，只供本地知识库使用。
