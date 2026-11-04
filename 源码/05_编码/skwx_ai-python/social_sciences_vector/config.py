import os

# 本地配置
# mysql 服务配置
mysql_host = os.getenv("BOOK_VECTOR_MYSQL_HOST", "127.0.0.1")  # MySQL 服务器地址（例如：localhost、IP 地址等）
mysql_user = os.getenv("BOOK_VECTOR_MYSQL_USER", "")  # 用户名
mysql_password = os.getenv("BOOK_VECTOR_MYSQL_PASSWORD", "")  # 密码
mysql_database = os.getenv("BOOK_VECTOR_MYSQL_DATABASE", "book_vector_demo")  # 要连接的数据库名称

# 连接到 Milvus 服务器
milvus_host = os.getenv("MILVUS_HOST", "127.0.0.1")
milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

# Milvus 集合名称前缀；集合用满后自动创建 _2、_3 等后续集合，留空时使用 book_partition_N。
milvus_collection_name = "book_vector_v2"


# 线上配置
# mysql 服务配置
# mysql_host = "127.0.0.1"  # MySQL 服务器地址（例如：localhost、IP 地址等）
# mysql_user = "root"  # 用户名
# mysql_password = os.getenv("BOOK_VECTOR_MYSQL_PASSWORD", "")  # 密码
# mysql_database = "sk_hymax"  # 要连接的数据库名称
#
# 连接到 Milvus 服务器
# milvus_host = "127.0.0.1"
# milvus_port = 19530
