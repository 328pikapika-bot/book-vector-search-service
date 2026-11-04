# 向量服务线上并发修复升级说明

## 版本信息

- 版本：`book-vector-v2-concurrency-hotfix-20260723`
- 服务端口：`8006`
- 升级类型：增量代码升级，不更换 MySQL、Milvus、模型或历史向量数据

## 修复内容

本版本修复批量并发上传时出现的以下异常：

```text
collection not found[database=default][collection=book_vector_v2]
Configured vector_library 'book_vector_v2' exists in MySQL, but the Milvus collection is missing
```

根因是多个同步 FastAPI 请求在同一进程的线程池中并发执行，并反复重绑 PyMilvus 的全局 `default` 连接。一个请求连接 `ssc_1` 后，另一个请求可能将同一别名切回 `default`，导致前一个请求在错误数据库中创建分区或写入向量。

本版本包含以下修复：

1. 每个 Milvus 数据库使用独立、稳定的连接别名，不再重绑全局 `default`。
2. 上传、检索、删除接口的 `Collection` 和 `utility` 调用全部显式指定连接别名。
3. `vector_library` 选择和首次创建增加 MySQL 连接级锁。
4. 创建分区时使用 `SELECT ... FOR UPDATE` 锁定目标 `vector_library` 行。
5. `partition_count` 改为数据库原子递增、递减，避免并发覆盖。
6. Milvus 分区创建成功而 MySQL 提交失败时执行补偿删除。
7. 删除接口可兼容“数据库记录存在、Milvus 分区已经不存在”的失败残留状态。
8. 同一 `book_id` 使用 MySQL 命名锁覆盖切片、向量生成、向量写入和 `flush` 全流程。
9. 相同图书重复请求会等待首请求完成；首请求成功后返回“不重复入库”，首请求失败清理后允许重试。
10. 整书删除和段落删除与上传共用书级锁；同书检索在上传期间返回“处理中”，不读取半成品。

书级锁不新增数据库表，发布前预检中的 `mysql_named_lock` 必须通过。默认写锁等待 1800 秒、检索门禁等待 30 秒，可通过环境变量 `VECTOR_BOOK_LOCK_TIMEOUT_SECONDS` 和 `VECTOR_BOOK_READ_LOCK_TIMEOUT_SECONDS` 调整。

## 升级包文件

本包是增量升级包，需覆盖以下文件：

- `national_development/milvus_connection.py`（新增）
- `national_development/create_library.py`
- `national_development/create_collection.py`
- `national_development/up_collection.py`
- `national_development/drop_collection.py`
- `national_development/drop_paragraph.py`
- `national_development/search_vector.py`
- `national_development/search_scalar.py`
- `national_development/field.py`

本包不包含线上 `config.py`。必须保留线上 MySQL、Milvus 地址和账号，只确认存在：

```python
milvus_collection_name = "book_vector_v2"
```

## 上线前必须执行

1. 暂停批量上传任务。
2. 暂停失败自动重试。
3. 等待当前正在执行的上传请求结束或停止旧服务。
4. 备份完整服务目录、线上 `config.py` 和当前 `log.txt`。
5. 确认 `ssc_1` 中的 `book_vector_v2` 不要删除。
6. 不要手工修改 `vector_library.partition_count`。

只读检查：

```sql
SELECT id, name, library_id, partition_count_max, partition_count, create_time
FROM vector_library
WHERE name = 'book_vector_v2';

SELECT book_id, partition_name, library_id, data_collection_id, create_time
FROM vector_collection
WHERE name = 'book_vector_v2'
ORDER BY create_time;
```

## 发布步骤

1. 停止旧服务：

   ```bash
   pgrep -af main_library_api.py
   pkill -f main_library_api.py
   ```

2. 将升级包中的 `national_development/` 覆盖到线上服务的同名目录。

3. 检查新增文件存在：

   ```bash
   test -f national_development/milvus_connection.py
   ```

4. 运行语法检查：

   ```bash
   python -m compileall -q national_development
   ```

5. 保留线上 `config.py`，确认集合前缀仍为 `book_vector_v2`。

6. 启动服务：

   ```bash
   chmod +x main_library_run.sh
   ./main_library_run.sh
   ```

## 发布后验证

### 1. 单本验证

选择一个从未上传过的新 `book_id`，调用：

```text
POST /up_collection/ImportFromText
```

确认：

- HTTP 返回 `200`。
- `ssc_1.book_vector_v2` 中创建 `book_{book_id}` 分区。
- `vector_collection` 中出现对应记录。
- 日志包含“上传成功”。

### 2. 并发验证

使用 5 个不同的新 `book_id` 并发上传，确认全部成功。验证期间同时执行一次已有图书检索。

日志中不得再出现：

```text
database=default][collection=book_vector_v2
```

### 3. 逐步恢复

并发数按 `1 -> 5 -> 10` 逐步放开。观察一批成功后再提高，不要立即恢复故障时的全部并发量。

## 失败任务重新下发

故障期间接口返回 `500` 的图书不能直接全部重传，因为部分图书可能已经创建分区并写入了部分向量。

对每个确认失败的 `book_id`：

1. 先调用修复后服务的 `DELETE /book/{book_id}` 清理该书的分区和数据库记录。
2. 如果返回“不存在”，确认 MySQL 和 Milvus 均无该书记录后继续。
3. 再重新调用 `POST /up_collection/ImportFromText`。
4. 不要清理日志中已经明确“上传成功”并返回 `200` 的图书。

如果发现 Milvus 有分区但 MySQL 没有 `vector_collection` 记录，应先由开发人员核对后清理，不要批量执行无条件删除。

## 回退

1. 再次暂停上传。
2. 停止服务。
3. 恢复升级前备份的代码和 `config.py`。
4. 重新启动服务。

回退只恢复代码，不删除 `book_vector_v2`、已有分区或历史向量数据。
