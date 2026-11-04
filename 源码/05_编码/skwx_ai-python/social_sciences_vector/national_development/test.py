# # #
# # # from db_process.db_tool.embedding_model import text2vec
# # #
# # # vec_query = text2vec(["你好，明天回家吗？"]).tolist()[0][:128]
# # #
# # #
# # from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
# # from pymilvus.orm import utility
# #
# # from config import milvus_host, milvus_port
# #
# # from pymilvus import connections, Collection
# #
# # # 连接到 Milvus 服务
# # connections.connect(host=milvus_host, port=milvus_port, db_name="ssc_1")
# # # 获取所有 collection 名称
# # collection_names = utility.list_collections()
# #
# # # 输出所有 collection 名称
# # print(collection_names)
# # # # 假设已经创建了一个名为 'jindaishigangyao' 的集合
# # # collection = Collection("jindaishigangyao")
# # #
# # # # 获取集合的所有字段
# # # fields = collection.schema.fields
# # #
# # # # 从字段中去除 'embedding' 字段
# # # field_names = [field.name for field in fields if field.name != 'embedding']
# # # print(field_names)
# # # # 构建查询条件
# # # expression = "title == '近代史纲要'"
# # # # 执行查询，排除 'embedding' 字段
# # # results = collection.query(expr=expression, output_fields=field_names)
# # #
# # # # 输出查询结果
# # # for result in results:
# # #     print(result)
# import numpy as np

field_all = {
  "vector_model": "m3e-base",
  "semantic_model": "",
  "book_id": "7310128796108800_1",
  "title": "近代史纲要_1",
  "max_text_total": "500"
}

field = {
      "college": [],
      "country": [],
      "directory_doc_type": "",
      "directory_e_keywords": [
        ""
      ],
      "unit_address": [],
      "directory_min_node_search": "",
      "language": "",
      "type": "",
      "research": [],
      "directory_en_title": "",
      "price": "0.0",
      "publish_time": "",
      "directory_subject_words": [
        ""
      ],
      "directory_inner_code": "135124",
      "text": "红军不怕远征难。万水千山只等闲。武林威后。红二。红四方面军和红二十五军也各自从立足的革命。根据地撤出。踏上。漫漫长征路。长征途中。英勇的红军在中国共产党领导下。面对险恶的自然条件和敌人穷凶极恶的围追堵截。以非凡的智慧和大无畏的英雄气概。血战相将。智取尊义。死渡赤水。巧渡金沙江。强渡大渡河。飞渡卢定桥。突破辣子口。血战独树镇。敖战嘉陵江。转战。乌蒙山。各路红军历经千辛万险。纵横十余省。突破了数十万敌人的围追堵截。跨越了数十条滔滔极流。征服了数十座矮矮雪山。穿越了数百里的茫茫草地。可仪腾袭浪。巫蒙磅礴走你晚。每当人们吟诵起毛泽东主席的七律长征时总会心潮澎湃。热泪盈眶。思绪穿越历史的烟云。回到那段惊心动魄的历史中。在硝烟与鲜血中感受英雄的气概。在苦难中。见证辉煌。1934年10月。由于党内左倾教条主义的错误领导。中央红军第五次反围剿势力。不得不离开赣南。闽西根据地。进行战略转移。此谓。每走一步都险象还生。1936年10月。三大红军主力会师甘肃惠宁。胜利完成了弃吞山河。标秉史册的。长征。长征。是人类历史上绝无仅有的奇迹。是一座让中国人民世代铭记的巍峨峰碑。毛泽东曾豪迈的宣告。长征。是历史记录上的第一次。长征是宣言书。长征是宣传队。长征是播种机。自从盘古开天帝。三皇五帝到如今。历史上曾经有过我们这样的长征吗。长征的胜利。宣告了国民党反动派。妄图消灭中国共产党。和红军阴谋的彻底失败。实现了中国共产党。北上抗日的战略转移。从此。中国共产党。以陕甘宁革命根据地为中心。开始了新的革命征程。并不断奏效。新的胜利。据统计。长征时期。红军指挥员平均年龄不足25岁。红军战士平均年龄不足20岁。",
      "directory_inter_code": "135124",
      "directory_id": "DIR_12430256",
      "ethnic": [],
      "search_title": "《城市学论丛》编辑委员会",
      "directory_search_title": "《城市学论丛》编辑委员会",
      "directory_issue_date": "",
      "content_text": "<p><b>《城市学论丛》编辑委员会</b>：（以姓氏笔画为序）</p><p>丁峻 石向实 张卫良 秦均平 黄宛峰</p>",
      "directory_name": "《城市学论丛》编辑委员会",
      "academic_title": [],
      "literature_id": "12430119",
      "research_area_values": [],
      "directory_en_search_title": "",
      "directory_down_count": 0,
      "position": [],
      "author_id": [],
      "directory_book_publish_date": "2012-04-01",
      "author_name": [],
      "research_region_values": [],
      "directory_keywords": [
        ""
      ],
      "isbn": "",
      "en_theme": "",
      "en_search_title": "",
      "title": "《城市学论丛》编辑委员会",
      "directory_code": "",
      "theme": "",
      "org_name": [],
      "study_branch_id": [],
      "directory_doi": "",
      "directory_hit_count": 0,
      "directory_level": 0,
      "book_id": "12430118",
      "disciplines": [],
      "directory_type": "formerAidText",
      "directory_product_series": "",
      "age": []
    }
from db_process.db_tool.embedding_model import text2vec, semantic_model
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, connections
from config import milvus_host, milvus_port

connections.connect(host=milvus_host, port=milvus_port, db_name="ssc_1")
collection = Collection("book_7310128796108800_1")
# # partition_name = "test_2"
# # collection.create_partition(partition_name)
# #
# # clean_text = [[field["literature_id"]], [field["directory_type"]], [field["directory_code"]], [field["directory_name"]],
# #               [field["directory_en_title"]],
# #               [field["directory_inner_code"]], [field["directory_search_title"]], [field["directory_en_search_title"]],
# #               [field["directory_level"]],
# #               [field["directory_keywords"]], [field["directory_e_keywords"]], [field["directory_subject_words"]],
# #               [field["directory_issue_date"]], [field["directory_doc_type"]], [field["directory_product_series"]],
# #               [field["directory_doi"]], [field["directory_min_node_search"]], [field["directory_hit_count"]],
# #               [field["directory_down_count"]], [field["directory_book_publish_date"]], [field["theme"]],
# #               [field["en_theme"]], [field["search_title"]], [field["directory_inter_code"]], [field["isbn"]],
# #               [field["language"]], [field["type"]], [field["content_text"]],
# #               [field["publish_time"]], [field["en_search_title"]], [field["price"]], [field["author_name"]],
# #               [field["author_id"]], [field["college"]], [field["org_name"]], [field["country"]],
# #               [field["position"]], [field["academic_title"]], [field["ethnic"]], [field["age"]], [field["disciplines"]],
# #               [field["research"]], [field["study_branch_id"]], [field["unit_address"]],
# #               [field["research_region_values"]], [field["research_area_values"]]]
# #
# # # 定义要插入的数据
# # embedding = text2vec([field["text"]], "m3e-base").tolist()
# #
# # ins = [["32234435"], [field_all["title"]], embedding, [field_all["book_id"]], [field["directory_id"]], [field["text"]]] + clean_text
# # print(ins)
# #
# # # 向指定分区插入数据
# # collection.insert(ins, partition_name="test_2")



# 查询指定分区的数据
embedding = text2vec([field["text"]], "m3e-base").tolist()

search_params = {
    "nprobe": 16  # 查询时搜索的倒排列表数量
}

exer = "id=='123223445'"

# 查询指定分区的向量数据
results = collection.search(embedding, "embedding", param=search_params, limit=10, partition_names=None, expr=exer)

# 输出查询结果
for result in results[0]:
    print(f"ID: {result.id}, 距离: {result.distance}")

