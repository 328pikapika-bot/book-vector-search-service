import json
import random

import html
import re
import traceback
import uuid
import time

from datetime import datetime
import mysql.connector
import pypinyin
from fastapi import Body, FastAPI, APIRouter
from pymilvus import Collection, utility, FieldSchema, DataType, CollectionSchema
from pymilvus.exceptions import PartitionAlreadyExistException
from starlette.responses import JSONResponse

from config import mysql_host, mysql_user, mysql_password, mysql_database, milvus_host, milvus_port
from db_process.db_tool.embedding_model import (
    get_embedding_device_info,
    semantic_model,
    text2vec,
)
from national_development.create_collection import CollectionCapacityExceeded, create_collection
from national_development.create_library import create_library
from national_development.book_operation_lock import (
    BookOperationLockTimeout,
    book_operation_lock,
)
from national_development.drop_collection import _drop_collection_unlocked
from national_development.milvus_connection import connect_milvus
from log.logger import Logger
logger = Logger()
app_up_collection = APIRouter()

def connection_mysql():
    # 创建数据库连接
    db_connection = mysql.connector.connect(
        host=mysql_host,  # MySQL 服务器地址（例如：localhost、IP 地址等）
        user=mysql_user,  # 用户名
        password=mysql_password,  # 密码
        database=mysql_database  # 要连接的数据库名称
    )
    return db_connection

def convert_to_pinyin(name):
    # 将中文字符串转换为拼音
    return ''.join(pypinyin.lazy_pinyin(name))

def legacy_split_text(text, chunk_size=1000):
    # 按照指定大小切割文本
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


CHUNK_OVERLAP = 100
REFERENCE_CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 120
CHUNK_STRATEGY = "paragraph_sentence_overlap_v2"
REFERENCE_CHUNK_STRATEGY = "reference_overlap_v1"
SENTENCE_BOUNDARY_CHARS = "\n。！？；;!?"
REFERENCE_TITLE_KEYWORDS = ("参考文献", "References", "Bibliography")
VECTOR_BATCH_SIZE = 16


def normalize_text_for_embedding(value):
    if value is None:
        return ""

    text = html.unescape(str(value)).replace("\xa0", " ")
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"(?i)</\s*(p|div|li|tr|h[1-6])\s*>", "\n", text)
    text = re.sub(r"(?i)<\s*(p|div|li|tr|h[1-6])[^>]*>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n+ *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _text_units(text):
    units = []
    start = 0
    separators = set(SENTENCE_BOUNDARY_CHARS)
    for index, char in enumerate(text):
        if char not in separators:
            continue
        raw = text[start:index + 1]
        unit_start = start + len(raw) - len(raw.lstrip())
        unit_end = index + 1 - (len(raw) - len(raw.rstrip()))
        if unit_start < unit_end:
            units.append((unit_start, unit_end))
        start = index + 1

    raw = text[start:]
    unit_start = start + len(raw) - len(raw.lstrip())
    unit_end = len(text) - (len(raw) - len(raw.rstrip()))
    if unit_start < unit_end:
        units.append((unit_start, unit_end))
    return units


def _paragraph_spans(text):
    spans = []
    start = 0
    for match in re.finditer(r"\n+", text):
        raw_start, raw_end = start, match.start()
        unit_start = raw_start + len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].lstrip())
        unit_end = raw_end - (len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].rstrip()))
        if unit_start < unit_end:
            spans.append((unit_start, unit_end))
        start = match.end()

    raw_start, raw_end = start, len(text)
    unit_start = raw_start + len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].lstrip())
    unit_end = raw_end - (len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].rstrip()))
    if unit_start < unit_end:
        spans.append((unit_start, unit_end))
    return spans


def _sentence_spans(text, start, end):
    spans = []
    current = start
    separators = set(SENTENCE_BOUNDARY_CHARS.replace("\n", ""))
    for index in range(start, end):
        if text[index] not in separators:
            continue
        raw = text[current:index + 1]
        unit_start = current + len(raw) - len(raw.lstrip())
        unit_end = index + 1 - (len(raw) - len(raw.rstrip()))
        if unit_start < unit_end:
            spans.append((unit_start, unit_end))
        current = index + 1

    raw = text[current:end]
    unit_start = current + len(raw) - len(raw.lstrip())
    unit_end = end - (len(raw) - len(raw.rstrip()))
    if unit_start < unit_end:
        spans.append((unit_start, unit_end))
    return spans


def _reference_spans(text):
    spans = []
    current = 0
    pattern = re.compile(r"\n+|；|;|(?=(?:\n|^)\s*\[\d+\])")
    for match in pattern.finditer(text):
        raw_start, raw_end = current, match.start()
        if raw_start < raw_end:
            unit_start = raw_start + len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].lstrip())
            unit_end = raw_end - (len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].rstrip()))
            if unit_start < unit_end:
                spans.append((unit_start, unit_end))
        current = match.end() if match.end() > match.start() else match.start()

    raw_start, raw_end = current, len(text)
    if raw_start < raw_end:
        unit_start = raw_start + len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].lstrip())
        unit_end = raw_end - (len(text[raw_start:raw_end]) - len(text[raw_start:raw_end].rstrip()))
        if unit_start < unit_end:
            spans.append((unit_start, unit_end))
    return spans


def _skip_leading_space(text, index, end):
    while index < end and text[index].isspace():
        index += 1
    return index


def _aligned_overlap_start(text, chunk_start, chunk_end, overlap):
    if overlap <= 0:
        return chunk_end

    fixed_start = max(chunk_start, chunk_end - overlap)
    search_start = max(chunk_start, fixed_start - overlap)
    boundary_chars = set(SENTENCE_BOUNDARY_CHARS)
    for index in range(fixed_start, search_start - 1, -1):
        if text[index] in boundary_chars:
            candidate = _skip_leading_space(text, index + 1, chunk_end)
            if chunk_start < candidate < chunk_end:
                return candidate

    return _skip_leading_space(text, fixed_start, chunk_end)


def _make_chunk(text, start, end, strategy):
    return {
        "text": text[start:end].strip(),
        "chunk_start": start,
        "chunk_end": end,
        "chunk_strategy": strategy,
    }


def _hard_split_span(text, start, end, chunk_size, overlap, strategy):
    chunks = []
    step = max(1, chunk_size - overlap)
    current = start
    while current < end:
        chunk_end = min(end, current + chunk_size)
        chunks.append(_make_chunk(text, current, chunk_end, strategy))
        if chunk_end >= end:
            break
        current += step
    return chunks


def _expand_units_for_chunking(text, spans, chunk_size, overlap, strategy):
    units = []
    for start, end in spans:
        if end - start <= chunk_size:
            units.append((start, end))
            continue

        sentence_spans = _sentence_spans(text, start, end)
        for sentence_start, sentence_end in sentence_spans:
            if sentence_end - sentence_start <= chunk_size:
                units.append((sentence_start, sentence_end))
            else:
                units.extend(
                    (chunk["chunk_start"], chunk["chunk_end"])
                    for chunk in _hard_split_span(text, sentence_start, sentence_end, chunk_size, overlap, strategy)
                )
    return units


def _build_chunks_from_units(text, units, chunk_size, overlap, min_chunk_size, strategy):
    chunks = []
    current_start = None
    current_end = None

    for unit_start, unit_end in units:
        if current_start is None:
            current_start = unit_start
            current_end = unit_end
            continue

        if unit_end - current_start <= chunk_size:
            current_end = unit_end
            continue

        chunks.append(_make_chunk(text, current_start, current_end, strategy))

        new_start = _aligned_overlap_start(text, current_start, current_end, overlap)
        if unit_end - new_start > chunk_size:
            new_start = unit_start
        current_start = new_start
        current_end = unit_end

    if current_start is not None:
        chunks.append(_make_chunk(text, current_start, current_end, strategy))

    if len(chunks) >= 2 and len(chunks[-1]["text"]) < min_chunk_size:
        last = chunks.pop()
        previous = chunks[-1]
        merged_start = previous["chunk_start"]
        merged_end = last["chunk_end"]
        if merged_end - merged_start <= chunk_size:
            previous.update(_make_chunk(text, merged_start, merged_end, strategy))
        else:
            chunks.append(last)

    total = len(chunks)
    for index, chunk in enumerate(chunks):
        chunk["chunk_index"] = index
        chunk["chunk_total"] = total
    return chunks


def _is_reference_section(directory_name=""):
    return any(keyword.lower() in str(directory_name or "").lower() for keyword in REFERENCE_TITLE_KEYWORDS)


def split_text(text, chunk_size=1000, overlap=CHUNK_OVERLAP, min_chunk_size=MIN_CHUNK_SIZE, directory_name=""):
    chunk_size = int(chunk_size or 1000)
    overlap = min(int(overlap), max(0, chunk_size // 3))
    min_chunk_size = min(int(min_chunk_size), chunk_size)
    clean_text = normalize_text_for_embedding(text)
    if not clean_text:
        return []

    if _is_reference_section(directory_name):
        reference_overlap = min(REFERENCE_CHUNK_OVERLAP, max(0, chunk_size // 3))
        units = _expand_units_for_chunking(
            clean_text,
            _reference_spans(clean_text),
            chunk_size,
            reference_overlap,
            REFERENCE_CHUNK_STRATEGY,
        )
        return _build_chunks_from_units(
            clean_text,
            units,
            chunk_size,
            reference_overlap,
            min_chunk_size,
            REFERENCE_CHUNK_STRATEGY,
        )

    paragraph_units = _expand_units_for_chunking(
        clean_text,
        _paragraph_spans(clean_text),
        chunk_size,
        overlap,
        CHUNK_STRATEGY,
    )
    return _build_chunks_from_units(
        clean_text,
        paragraph_units,
        chunk_size,
        overlap,
        min_chunk_size,
        CHUNK_STRATEGY,
    )


def generate_unique_random_integers(n):
    unique_numbers = set()
    while len(unique_numbers) < n:
        random_int = random.randint(10**17, 10**18 - 1)
        unique_numbers.add(random_int)  # set 自动去重
    return list(unique_numbers)


def normalize_publish_direct(value):
    if value is None:
        return []
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]

    result = []
    for item in values:
        item = str(item).strip()
        if item and item not in result:
            result.append(item)
    return result


def truncate_utf8(value, max_bytes):
    if value is None:
        return ""
    text = str(value)
    encoded_text = text.encode("utf-8")
    if len(encoded_text) <= max_bytes:
        return text
    return encoded_text[:max_bytes].decode("utf-8", errors="ignore")


def data_process(field):
    publish_direct = normalize_publish_direct(field.get("publish_direct", []))
    content_text = truncate_utf8(field.get("content_text", ""), 65535)
    chunk_index = int(field.get("chunk_index", 0) or 0)
    chunk_total = int(field.get("chunk_total", 0) or 0)
    chunk_start = int(field.get("chunk_start", 0) or 0)
    chunk_end = int(field.get("chunk_end", 0) or 0)
    chunk_strategy = field.get("chunk_strategy", "")
    # clean_text = []
    # for key, value in field.items():
    #     if key not in ["title","book_id","directory_id","text"]:
    #         clean_text.append([field[key]])
    clean_text = [[field["literature_id"]], [field["directory_type"]], [field["directory_code"]], [field["directory_name"]], [field["directory_en_title"]],
                  [field["directory_inner_code"]], [field["directory_search_title"]], [field["directory_en_search_title"]], [field["directory_level"]],
                  [field["directory_keywords"]], [field["directory_e_keywords"]], [field["directory_subject_words"]], [field["directory_issue_date"]], [field["directory_doc_type"]], [field["directory_product_series"]],
                  [field["directory_doi"]], [field["directory_min_node_search"]], [field["directory_hit_count"]], [field["directory_down_count"]], [field["directory_book_publish_date"]], [field["theme"]],
                  [field["en_theme"]], [field["search_title"]], [field["directory_inter_code"]], [field["isbn"]], [field["language"]], [field["type"]], [content_text],
                  [field["publish_time"]], [field["en_search_title"]], [field["price"]], [field["author_name"]], [field["author_id"]], [field["college"]], [field["org_name"]], [field["country"]],
                  [field["position"]], [field["academic_title"]], [field["ethnic"]], [field["age"]], [field["disciplines"]], [field["research"]], [field["study_branch_id"]], [field["unit_address"]],
                  [field["research_region_values"]], [field["research_area_values"]], [publish_direct], [chunk_index], [chunk_total], [chunk_start], [chunk_end], [chunk_strategy]]

    return clean_text


def _append_insert_row(columns, chunk_record, vector, book_id, book_name):
    chunk = chunk_record["chunk"]
    paragraph = chunk_record["paragraph"]
    chunk_field = chunk_record["chunk_field"]
    data_field = data_process(chunk_field)
    row_values = [
        "".join(str(num) for num in generate_unique_random_integers(1)),
        book_name,
        vector,
        book_id,
        paragraph["directory_id"],
        chunk,
    ] + [field_column[0] for field_column in data_field]

    for index, value in enumerate(row_values):
        columns[index].append(value)


def _insert_chunk_batch(collection, partition_name, chunk_records, vectors, book_id, book_name):
    if not chunk_records:
        return 0
    column_count = 6 + len(data_process(chunk_records[0]["chunk_field"]))
    columns = [[] for _ in range(column_count)]
    for chunk_record, vector in zip(chunk_records, vectors):
        _append_insert_row(columns, chunk_record, vector, book_id, book_name)
    collection.insert(columns, partition_name=partition_name)
    return len(chunk_records)


def get_existing_library_id(book_id):
    with connection_mysql() as db_connection:
        cursor = db_connection.cursor()
        sql = """
            SELECT library_id FROM vector_collection
            WHERE partition_name = %s
            ORDER BY create_time DESC
            LIMIT 1;
        """
        cursor.execute(sql, (f"book_{book_id}",))
        row = cursor.fetchone()
    return row[0] if row else ""


def _up_collection_locked(data, lock_handle):
    # 图书级元数据以请求顶层字段为准，兼容服务端可能传入的驼峰命名。
    data = dict(data)
    data["book_id"] = str(data.get("book_id") or data.get("bookId") or "").strip()
    data["title"] = str(data.get("title") or data.get("bookName") or data.get("name") or "").strip()
    if not data["book_id"]:
        raise ValueError("book_id/bookId cannot be empty")
    if not data["title"]:
        raise ValueError("title/bookName cannot be empty")

    now = datetime.now()
    formatted_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')
    library_id = ""
    try:
        logger.info(f"正在运行{data['book_id']}")
        lock_handle.ensure_held()
        for selection_attempt in range(3):
            library_id = create_library(
                data["title"],
                data["book_id"],
                data["max_text_total"],
                data["vector_model"],
                data["semantic_model"],
                data.get("force_new_collection", False)
            )["data"]["id"]
            try:
                dataCollectionId_partition_name = create_collection(
                    data["title"], data["book_id"], library_id, data["paragraphs"]
                )["data"]
                break
            except CollectionCapacityExceeded as exc:
                logger.info(f"{data['book_id']}检测到集合已满，重新选择新集合: {exc}")
                if selection_attempt == 2:
                    raise
        partition_name = dataCollectionId_partition_name["partition_name"]
        dataCollectionId = dataCollectionId_partition_name["id"]

        # try:
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()

            sql = "SELECT embedding_model_max_token_total, library_id FROM vector_library WHERE id = %s;"
            cursor.execute(sql, (library_id,))
            max_token_library_id = cursor.fetchone()  # 获取分块大小

            sql = "SELECT name FROM vector_library_base WHERE id = %s;"
            cursor.execute(sql, (max_token_library_id[1],))
            database = cursor.fetchone()[0]  # 获取库名

            sql = "SELECT name FROM vector_collection WHERE data_collection_id = %s;"
            cursor.execute(sql, (dataCollectionId,))
            collection_name = cursor.fetchone()[0]  # 获取表名

        milvus_alias = connect_milvus(database)

        # 文本转换为向量
        name_zh = data['title']
        # 拼音转换
        data['title'] = convert_to_pinyin(data['title'])

        # 生成唯一ID


        collection = Collection(collection_name, using=milvus_alias)

        chunk_records = []
        split_start_time = time.perf_counter()
        for paragraph in data["paragraphs"]:
            if data["semantic_model"] != "":
                text_chunks = semantic_model() or []
            else:
                directory_context = " ".join([
                    str(paragraph.get("directory_name", "")),
                    str(paragraph.get("directory_search_title", "")),
                    str(paragraph.get("search_title", "")),
                ])
                text_chunks = split_text(
                    paragraph['text'],
                    chunk_size=max_token_library_id[0],
                    directory_name=directory_context,
                )

            for chunk_info in text_chunks:
                if isinstance(chunk_info, dict):
                    chunk = chunk_info.get("text", "")
                    chunk_field = paragraph.copy()
                    chunk_field.update({
                        "chunk_index": chunk_info.get("chunk_index", 0),
                        "chunk_total": chunk_info.get("chunk_total", 0),
                        "chunk_start": chunk_info.get("chunk_start", 0),
                        "chunk_end": chunk_info.get("chunk_end", 0),
                        "chunk_strategy": chunk_info.get("chunk_strategy", CHUNK_STRATEGY),
                    })
                else:
                    chunk = str(chunk_info)
                    chunk_field = paragraph.copy()
                if not chunk.strip():
                    continue
                chunk_records.append({
                    "chunk": chunk,
                    "paragraph": paragraph,
                    "chunk_field": chunk_field,
                })

        split_elapsed = time.perf_counter() - split_start_time
        total_chunks = len(chunk_records)
        device_info = get_embedding_device_info()
        device_reason = (
            f"，原因={device_info['fallback_reason']}"
            if device_info["device_type"] == "CPU" and device_info["fallback_reason"]
            else ""
        )
        logger.info(
            f"{data['book_id']}开始向量化: 设备={device_info['display']}, "
            f"批大小={VECTOR_BATCH_SIZE}{device_reason}"
        )
        logger.info(
            f"{data['book_id']}切片完成: 章节数={len(data['paragraphs'])}, chunk数={total_chunks}, "
            f"切片耗时={split_elapsed:.2f}s, 向量批大小={VECTOR_BATCH_SIZE}"
        )

        if total_chunks == 0:
            logger.info(f"{data['book_id']}没有可入库chunk，直接返回成功")
        else:
            import_start_time = time.perf_counter()
            processed_count = 0

            for batch_start in range(0, total_chunks, VECTOR_BATCH_SIZE):
                batch_records = chunk_records[batch_start:batch_start + VECTOR_BATCH_SIZE]
                batch_chunks = [record["chunk"] for record in batch_records]
                batch_vectors = text2vec(
                    batch_chunks,
                    data["vector_model"],
                    batch_size=VECTOR_BATCH_SIZE,
                    log_context=f"图书{data['book_id']}",
                ).tolist()
                # A named lock is connection-scoped.  Detect a lost MySQL
                # session before writing another vector batch.
                lock_handle.ensure_held()
                _insert_chunk_batch(
                    collection,
                    partition_name,
                    batch_records,
                    batch_vectors,
                    data["book_id"],
                    name_zh,
                )

                processed_count += len(batch_records)
                elapsed = time.perf_counter() - import_start_time
                speed = processed_count / elapsed if elapsed > 0 else 0
                remain = (total_chunks - processed_count) / speed if speed > 0 else 0
                percent = processed_count * 100 / total_chunks
                logger.info(
                    f"{data['book_id']}向量入库进度: {processed_count}/{total_chunks} "
                    f"({percent:.1f}%), 耗时={elapsed:.1f}s, 速度={speed:.2f}chunk/s, "
                    f"预计剩余={remain:.1f}s"
                )

        lock_handle.ensure_held()
        collection.flush()
        total_elapsed = time.perf_counter() - split_start_time
        logger.info(f"{data['book_id']}上传成功: chunk数={total_chunks}, 总耗时={total_elapsed:.2f}s")

        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "name": name_zh,
                "type": "Standard",
                "raw": data['paragraphs'],
                "book_id": data["book_id"],
                "total": 0,
                "readonly": "false",
                "ready": "false",
                "createTime": formatted_time,
                "modifyTime": "null",
                "libraryId": library_id
            }
        }

        return JSONResponse(result)
    except PartitionAlreadyExistException as e:
        existing_library_id = get_existing_library_id(data["book_id"]) or library_id
        logger.info(f"{data['book_id']}已存在，不做修改, 返回成功, libraryId={existing_library_id}")
        result_exist = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "已存在，不重复入库",
            "data": {
                "name": data['title'],
                "type": "Standard",
                "raw": data['paragraphs'],
                "book_id": data["book_id"],
                "total": 0,
                "readonly": "false",
                "ready": "false",
                "createTime": formatted_time,
                "modifyTime": "null",
                "libraryId": existing_library_id
            }
        }
        return JSONResponse(result_exist)
    except Exception as err:
        error_details = traceback.format_exc()  # 获取完整的错误堆栈信息
        logger.info(f"{data['book_id']}上传失败, 失败原因为: {err}, 错误详细信息: {error_details}")
        # 下撤图书
        # The outer request already owns this book's lock.  Re-entering the
        # public DELETE endpoint here would deadlock on a second connection.
        _drop_collection_unlocked(data["book_id"])
        return JSONResponse({"authorization": False, "stateCode": 1, "state": False, "message": str(error_details), "data": None}, status_code=500)


@app_up_collection.post("/up_collection/ImportFromText")
def up_collection(data=Body()):
    book_id = str(data.get("book_id", "") or "").strip()
    if not book_id:
        return JSONResponse(
            {
                "authorization": False,
                "stateCode": 1,
                "state": False,
                "message": "book_id cannot be empty",
                "data": None,
            },
            status_code=400,
        )
    try:
        with book_operation_lock(book_id) as lock_handle:
            logger.info(
                f"{book_id}获得全流程锁, 等待耗时={lock_handle.wait_seconds:.3f}s"
            )
            return _up_collection_locked(data, lock_handle)
    except BookOperationLockTimeout as exc:
        logger.info(f"{book_id}等待全流程锁超时: {exc}")
        return JSONResponse(
            {
                "authorization": False,
                "stateCode": 1,
                "state": False,
                "message": str(exc),
                "data": {"book_id": book_id, "status": "processing"},
            },
            status_code=409,
        )
    except Exception as exc:
        error_details = traceback.format_exc()
        logger.info(f"{book_id}获取全流程锁失败: {exc}, 错误详细信息: {error_details}")
        return JSONResponse(
            {
                "authorization": False,
                "stateCode": 1,
                "state": False,
                "message": error_details,
                "data": None,
            },
            status_code=500,
        )

