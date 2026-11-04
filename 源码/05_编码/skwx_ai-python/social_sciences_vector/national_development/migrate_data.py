from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from pymilvus import connections, utility
from concurrent.futures import ThreadPoolExecutor
connections.connect(host="127.0.0.1", port=19530, db_name="ssc_1")

connections_list = utility.list_collections()



def multi_threading(results, partition_name):
    connections.connect(host="ip", port=19530, db_name="ssc_1")
    collection_1 = Collection("book_partition_1")
    collection_1.create_partition(partition_name)
    for field in results:
        clean_text = [[field["literature_id"]], [field["directory_type"]], [field["directory_code"]],
                      [field["directory_name"]],
                      [field["directory_en_title"]],
                      [field["directory_inner_code"]], [field["directory_search_title"]],
                      [field["directory_en_search_title"]],
                      [field["directory_level"]],
                      [field["directory_keywords"]], [field["directory_e_keywords"]],
                      [field["directory_subject_words"]],
                      [field["directory_issue_date"]], [field["directory_doc_type"]],
                      [field["directory_product_series"]],
                      [field["directory_doi"]], [field["directory_min_node_search"]],
                      [field["directory_hit_count"]],
                      [field["directory_down_count"]], [field["directory_book_publish_date"]], [field["theme"]],
                      [field["en_theme"]], [field["search_title"]], [field["directory_inter_code"]],
                      [field["isbn"]],
                      [field["language"]], [field["type"]], [field["content_text"]],
                      [field["publish_time"]], [field["en_search_title"]], [field["price"]], [field["author_name"]],
                      [field["author_id"]], [field["college"]], [field["org_name"]], [field["country"]],
                      [field["position"]], [field["academic_title"]], [field["ethnic"]], [field["age"]],
                      [field["disciplines"]],
                      [field["research"]], [field["study_branch_id"]], [field["unit_address"]],
                      [field["research_region_values"]], [field["research_area_values"]]]
        ins = [[field["id"]], [field["title"]], [field["embedding"]], [field["book_id"]], [field["directory_id"]],
               [field["text"]]] + clean_text
        collection_1.insert(ins, partition_name=partition_name)
    collection_1.flush()


def precess_data():
    for collection_name in connections_list:
        if collection_name == "book_partition_1":
            collection = Collection(collection_name)
            partitions = collection.partitions
            print(collection_name)
            total_entities = collection.num_entities
            for partition in partitions:
                partition_name = partition.name
                if partition.name == "_default":
                    continue
                expr = f"book_id == '{partition_name.split('_')[-1]}'"
                # 查询整个 collection 中的所有数据
                results = collection.query(expr=expr, output_fields=["*"], partition=partition_name)
                print(results)
                tasks = []
                with ThreadPoolExecutor(max_workers=200) as executor:
                    task = executor.submit(multi_threading, results, partition_name)

                    tasks.append(task)
                # 等待所有任务完成
                for task in tasks:
                    task.result()  # 获取任务的结果，确保任务已经完成


if __name__ == '__main__':
    precess_data()

