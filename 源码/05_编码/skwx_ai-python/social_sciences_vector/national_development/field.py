from pymilvus import connections, Collection, utility, FieldSchema, DataType, CollectionSchema
# 初始化 fields 列表
fields = []

# 添加其他字段
field1 = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=255)  # 设置主键字段
field2 = FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=255)  # 设置图书名称字段
field3 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # 设置向量字段
field4 = FieldSchema(name="book_id", dtype=DataType.VARCHAR, max_length=255)  # 设置图书ID字段
field5 = FieldSchema(name="directory_id", dtype=DataType.VARCHAR, max_length=255)  # 设置章节ID字段
field6 = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)  # 设置文本字段
field7 = FieldSchema(name="literature_id", dtype=DataType.VARCHAR, max_length=255)  # 设置文献ID字段
field8 = FieldSchema(name="directory_type", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置目录类型字段
field9 = FieldSchema(name="directory_code", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置目录编号字段
field10 = FieldSchema(name="directory_name", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置目录名称字段
field11 = FieldSchema(name="directory_en_title", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置目录英文名称字段
field12 = FieldSchema(name="directory_inner_code", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置内部编码字段
field13 = FieldSchema(name="directory_search_title", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置搜索标题字段
field14 = FieldSchema(name="directory_en_search_title", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置英文搜索标题字段
field15 = FieldSchema(name="directory_level", dtype=DataType.INT64, max_length=255, nullable=True)  # 设置目录级别字段
field16 = FieldSchema(name="directory_keywords", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=1000, nullable=True)  # 设置关键词字段
field17 = FieldSchema(name="directory_e_keywords", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=1000, nullable=True)  # 设置英文关键词字段
field18 = FieldSchema(name="directory_subject_words", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=1000, nullable=True)  # 设置主题词字段
field19 = FieldSchema(name="directory_issue_date", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置出版时间字段
field20 = FieldSchema(name="directory_doc_type", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置类型字段
field21 = FieldSchema(name="directory_product_series", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置产品系列字段
field22 = FieldSchema(name="directory_doi", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置DOI同方标识符字段
field23 = FieldSchema(name="directory_min_node_search", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置检索最小节点字段
field24 = FieldSchema(name="directory_hit_count", dtype=DataType.INT64, max_length=255, nullable=True)  # 设置点击量字段
field25 = FieldSchema(name="directory_down_count", dtype=DataType.INT64, max_length=255, nullable=True)  # 设置下载统计字段
field26 = FieldSchema(name="directory_book_publish_date", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置图书出版日期字段
field27 = FieldSchema(name="theme", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置主题字段
field28 = FieldSchema(name="en_theme", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置英文主题字段
field29 = FieldSchema(name="search_title", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置检索标题字段
field30 = FieldSchema(name="directory_inter_code", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置对应目录内部编码字段
field31 = FieldSchema(name="isbn", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置国际标准书号字段
field32 = FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置语言字段
field33 = FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置资源类型字段
field34 = FieldSchema(name="content_text", dtype=DataType.VARCHAR, max_length=65535, nullable=True)  # 设置文献正文字段
field35 = FieldSchema(name="publish_time", dtype=DataType.VARCHAR, max_length=255, nullable=True)  # 设置发布时间字段
field36 = FieldSchema(name="en_search_title", dtype=DataType.VARCHAR, max_length=1000, nullable=True)  # 设置英文检索标题字段
field37 = FieldSchema(name="price", dtype=DataType.VARCHAR, max_length=50, nullable=True)  # 设置章节单价字段
field38 = FieldSchema(name="author_name", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=255, nullable=True)  # 设置作者名称字段
field39 = FieldSchema(name="author_id", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=255, nullable=True)  # 设置作者ID字段
field40 = FieldSchema(name="college", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设毕业院校字段
field41 = FieldSchema(name="org_name", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置所在单位字段
field42 = FieldSchema(name="country", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置国籍字段
field43 = FieldSchema(name="position", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置职务字段
field44 = FieldSchema(name="academic_title", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置职称字段
field45 = FieldSchema(name="ethnic", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=255, nullable=True)  # 设置民族字段
field46 = FieldSchema(name="age", dtype=DataType.ARRAY, element_type=DataType.INT64, max_capacity=50, max_length=255, nullable=True)  # 设置年龄字段
field47 = FieldSchema(name="disciplines", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置专业字段
field48 = FieldSchema(name="research", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置研究方向字段
field49 = FieldSchema(name="study_branch_id", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=255, nullable=True)  # 设置研究机构ID字段
field50 = FieldSchema(name="unit_address", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置单位地址字段
field51 = FieldSchema(name="research_region_values", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置研究区域Value字段
field52 = FieldSchema(name="research_area_values", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=65535, nullable=True)  # 设置研究领域Value字段


# 将其他字段添加到字段列表中
field53 = FieldSchema(name="publish_direct", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=50, max_length=255, nullable=True)
field54 = FieldSchema(name="chunk_index", dtype=DataType.INT64, max_length=255, nullable=True)
field55 = FieldSchema(name="chunk_total", dtype=DataType.INT64, max_length=255, nullable=True)
field56 = FieldSchema(name="chunk_start", dtype=DataType.INT64, max_length=255, nullable=True)
field57 = FieldSchema(name="chunk_end", dtype=DataType.INT64, max_length=255, nullable=True)
field58 = FieldSchema(name="chunk_strategy", dtype=DataType.VARCHAR, max_length=255, nullable=True)

fields.append(field1)
fields.append(field2)
fields.append(field3)
fields.append(field4)
fields.append(field5)
fields.append(field6)
fields.append(field7)
fields.append(field8)
fields.append(field9)
fields.append(field10)
fields.append(field11)
fields.append(field12)
fields.append(field13)
fields.append(field14)
fields.append(field15)
fields.append(field16)
fields.append(field17)
fields.append(field18)
fields.append(field19)
fields.append(field20)
fields.append(field21)
fields.append(field22)
fields.append(field23)
fields.append(field24)
fields.append(field25)
fields.append(field26)
fields.append(field27)
fields.append(field28)
fields.append(field29)
fields.append(field30)
fields.append(field31)
fields.append(field32)
fields.append(field33)
fields.append(field34)
fields.append(field35)
fields.append(field36)
fields.append(field37)
fields.append(field38)
fields.append(field39)
fields.append(field40)
fields.append(field41)
fields.append(field42)
fields.append(field43)
fields.append(field44)
fields.append(field45)
fields.append(field46)
fields.append(field47)
fields.append(field48)
fields.append(field49)
fields.append(field50)
fields.append(field51)
fields.append(field52)
fields.append(field53)
fields.append(field54)
fields.append(field55)
fields.append(field56)
fields.append(field57)
fields.append(field58)
