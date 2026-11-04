from pymilvus import connections, Collection

from config import milvus_host, milvus_port


def main():
    # 连接到 Milvus 服务
    connections.connect(host=milvus_host, port=milvus_port, db_name="zhongguojinxiandaishigangyao_ziyuan")

    collection = Collection("zhongguojinxiandaishigangyao_shipin")


    # 查询并统计去重后的 label 数量
    results = collection.query(
        expr="",  # 可以添加筛选条件，这里为空表示查询所有数据
        output_fields=["resourceId"],
        limit=10000
    )
    unique_labels = set()
    for result in results:
        unique_labels.add(result["resourceId"])
    total_unique_labels = len(unique_labels)
    print(f"Total unique labels: {total_unique_labels}")


if __name__ == "__main__":
    main()