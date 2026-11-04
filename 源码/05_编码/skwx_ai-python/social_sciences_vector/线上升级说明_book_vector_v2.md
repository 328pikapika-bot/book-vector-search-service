# 向量服务线上升级说明

## 本次升级目的

解决新版本上传向量时，向旧 Milvus 集合写入失败的问题。旧集合缺少新版本需要的以下 6 个字段：

- `publish_direct`
- `chunk_index`
- `chunk_total`
- `chunk_start`
- `chunk_end`
- `chunk_strategy`

线上旧集合及其中的数据不删除、不迁移。本次新增的图书从新集合 `book_vector_v2` 开始写入。

向量记录中的图书字段使用 Milvus 的现有字段名：`book_id`（图书 ID）和 `title`（图书名称）。上传请求应在顶层提供 `book_id`、`title`；同时兼容服务端传入的 `bookId`、`bookName`。章节段落不需要重复携带图书 ID。

## 集合命名规则

在 `config.py` 中配置：

```python
milvus_collection_name = "book_vector_v2"
```

创建逻辑如下：

| 条件 | 使用的集合 |
| --- | --- |
| 新前缀下尚无集合 | `book_vector_v2` |
| 当前集合分区数达到 1024 | `book_vector_v2_2` |
| 后续集合继续用满 | `book_vector_v2_3`、`book_vector_v2_4`，依此类推 |
| 配置项不存在或为空 | 保持一期逻辑，使用 `book_partition_N` |

服务会校验待使用集合是否含有上述 6 个新字段。若集合是旧结构，服务会明确报错，不会继续写入导致后台上传失败。

## 升级包内容

以 `social_sciences_vector` 目录为服务包。至少需要随本次升级发布以下文件：

- `config.py`
- `national_development/create_library.py`
- `national_development/create_collection.py`
- `national_development/drop_collection.py`
- `national_development/field.py`
- `national_development/up_collection.py`

不要将本地的 `log/`、`log.txt`、`test.log`、`nohup.out`、`__pycache__/` 打入升级包。线上已有 Python 环境和依赖无需重复安装；若环境不存在，则使用 `requirements.txt` 安装依赖。

## 上线前确认

1. 备份线上当前服务目录和 `config.py`。
2. 在 `config.py` 保留线上 MySQL、Milvus 地址和账号，仅新增或修改：

   ```python
   milvus_collection_name = "book_vector_v2"
   ```

3. 确认服务账号可连接 MySQL 业务库和 Milvus 的 `ssc_1` 数据库。
4. 确认线上 Milvus 中不存在同名但旧结构的 `book_vector_v2` 集合；若存在，不能直接复用，应改用新的未占用前缀。

## 发布与验证

1. 停止旧服务进程：`main_library_api.py`。
2. 覆盖升级包中的文件。
3. 在服务目录执行：

   ```bash
   chmod +x main_library_run.sh
   ./main_library_run.sh
   ```

   服务监听 `8006` 端口。

4. 由服务端调用 `POST /up_collection/ImportFromText` 上传一条测试图书。
5. 验证返回成功；MySQL 的 `vector_library`、`vector_collection` 中出现对应记录；Milvus 的 `ssc_1` 数据库中创建 `book_vector_v2`，并有对应的 `book_{book_id}` 分区和向量数据。
6. 再执行一次已有图书的向量检索，确认一期旧集合仍可正常查询。

## 回退方式

停止服务，恢复备份的代码与 `config.py`，然后重新启动服务。由于本次不会修改或删除旧集合和旧数据，回退不会影响一期已入库内容。新建的 `book_vector_v2` 数据保留在 Milvus 中；如需清理，应在确认无业务引用后单独执行，不能在回退时直接删除。

## 服务端需配合的信息

- 服务地址和端口：向量服务 `8006`。
- 调用接口：`POST /up_collection/ImportFromText`。
- 请求中 `book_id` 必须唯一；重复上传同一图书前，先确认是否需要删除已有分区。
- 上传失败时，请将接口响应、服务 `log.txt` 中完整异常、`book_id` 和请求时间一并提供给算法侧排查。
