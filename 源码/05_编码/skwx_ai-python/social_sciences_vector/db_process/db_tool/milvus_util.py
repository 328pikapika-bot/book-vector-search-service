#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2023/10/12 16:37
# @Author : yzq
# @File : db_util.py
# @Software: PyCharm
import traceback

from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db, MilvusClient
from db_process.config.constant_config import CONSISTENCY_LEVEL, METRIC_TYPE
from log.logger import Logger

logger = Logger()

class MilvusUtils:

    def __init__(self, ip, port='19530', alias="default", db_name="default", collection_name=''):
        self.conn = connections.connect(host=ip, port=port, alias=alias)
        if not db.list_database().__contains__(db_name):
            db.create_database(db_name)
        db.using_database(db_name)
        self.collection = None
        if collection_name:
            self.collection = Collection(collection_name)
            self.collection.load()

    def create_collection(self, collection_name, fields_list, num_partitions=1024):
        """
        创建数据库集合
        :param collection_name: 集合名称
        :param fields_list: 字段信息
        :param num_partitions: 分区数
        :return:
        """
        if utility.has_collection(collection_name):
            return True

        fields = []
        index_params = []
        for field in fields_list:
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description']
            )

            if field.__contains__('dim'):
                temp_params = FieldSchema(
                    name=field['name'],
                    dtype=field['dtype'],
                    description=field['description'],
                    dim=field['dim'],
                )

            if field.__contains__('max_length'):
                temp_params = FieldSchema(
                    name=field['name'],
                    dtype=field['dtype'],
                    description=field['description'],
                    max_length=field['max_length'],
                )
            if field.__contains__('is_primary'):
                temp_params.is_primary = field['is_primary']
                temp_params.auto_id = field['auto_id']
            fields.append(temp_params)
            # 判断是否创建索引
            if field.__contains__('index_params'):
                # 向量索引
                if field['index_params']:
                    temp_params = {
                        "metric_type": field['index_params']['metric_type'],  # 'IP'
                        "index_type": field['index_params']['index_type'],  # "IVF_FLAT",
                        "params": {"nlist": field['index_params']['nlist']}  #
                    }
                # 标量索引
                else:
                    temp_params = None
                index_params.append((field['name'], 'index_' + field['name'], temp_params))

        schema = CollectionSchema(
            fields=fields,
            description="nstl serach"
        )
        # 创建集合
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=2,  # 要创建的集合的分片数。	[1,16]
            # num_partitions=num_partitions,  # 要创建的集合的逻辑分区数。	[1,4096]
            consistency_level=CONSISTENCY_LEVEL
        )
        # 集合创建索引
        for field_name, index_name, index_param in index_params:
            collection.create_index(
                field_name=field_name,
                index_params=index_param,
                index_name=index_name
            )
        return True

    def create_index(self, collection_name, field_name, index_name, **kwargs):
        """
        创建索引
        :param collection_name:
        :param field_name:
        :param index_name:
        :param kwargs:
        :return:
        """
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return False
        try:
            collection = Collection(collection_name)
            temp_params = {
                "metric_type": kwargs.get('metric_type', 'IP'),  # 'IP'
                "index_type": kwargs.get('index_type', "IVF_FLAT"),
                "params": {"nlist": kwargs.get('index_type', 128)}
            }
            # 集合创建索引
            collection.create_index(
                field_name=field_name,
                index_params=temp_params,
                index_name=index_name
            )
        except:
            logger.error("创建索引异常")
            return False
        return True

    def insert_datas(self, datas, collection_name, partition_name=None):
        """
        插入数据
        :param collection_name:
        :param datas: 数据集 list/tuple/pandas.DataFrame
        :param partition_name: 分区名称
        :return:
        """
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return 0
        try:
            collection = Collection(collection_name)
            res = collection.insert(datas, partition_name=partition_name)
            # collection.disconnect("default")
            return res.insert_count
        except Exception as e:
            logger.error(collection_name + ":导入数据异常")
            logger.error(e)
            return 0

    def delete_datas(self, collection_name, id_list):
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return False
        try:
            expr = "id in " + str(id_list)
            collection = Collection(collection_name)
            collection.delete(expr)
            return True
        except:
            logger.error(collection_name + ":数据删除异常")
            return False

    def search_datas(self, collection_name, vec_field, vec_text, output_fields, start_num=0, top_n=20,
                     nprobe=32, threshold=0):
        """
        :param collection_name:
        :param vec_field:
        :param vec_text:
        :param output_fields:
        :param expr:
        :param start_num:
        :param top_n:
        :param nprobe:要搜索的群集单元数
        :return:
        """
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return None

        search_params = {
            "metric_type": METRIC_TYPE,
            "offset": start_num,  # the sum of `offset` in `param` and `limit` should be less than 16384.
            "ignore_growing": False,
            "param": {"nprobe": nprobe}  # , "range_filter": 0.85 radius<distance <= range_filter
        }
        if threshold:
            search_params['param']['radius'] = threshold
        try:
            # collection = Collection(collection_name)
            # collection.load()
            results = self.collection.search(
                data=vec_text,  # 查询向量
                anns_field=vec_field,  # 检索向量字段
                param=search_params,
                limit=top_n,  # 输出结果数量 # the sum of `limit` should be less than 16384.
                output_fields=output_fields,  # 输出字段名称
                consistency_level=CONSISTENCY_LEVEL,  # 一致性级别
                timeout=5   # 超时时间, 5秒钟
            )
            # 结果转换
            res_list = []
            if results:
                for item in results[0]:
                    temp_dict = dict()
                    temp_dict['distance'] = item.distance
                    for key in output_fields:
                        temp_dict[key] = item.entity.get(key)
                    res_list.append(temp_dict)
            return res_list
        except Exception:
            logger.error(collection_name + ":数据查询异常")
            logger.error(traceback.print_exc())
            return []


    def get_datas_by_id(self, collection_name, id_list, output_fields):
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return None
        try:
            res = self.collection.query(
                expr="id in %s" % id_list,
                output_fields = output_fields
            )
        except Exception as e:
            logger.error(collection_name + ":数据查询异常")
            logger.debug(e)
            return None
        return res


class MilvusUtils1:
    def __init__(self, user, pwd, ip, port='19530', alias="default", db_name="default"):
        self.user = user
        self.pwd = pwd
        self.ip = ip
        self.port = port
        self.alias = alias
        self.db_name = db_name
        self.client = MilvusClient(uri=r"http://" + ip + ':' + port, user=user, password=pwd, db_name=db_name)

    def __get_connect(self):
        connections.connect(user=self.user, password=self.pwd, host=self.ip, port=self.port, alias=self.alias)
        if not db.list_database().__contains__(self.db_name):
            db.create_database(self.db_name)
        db.using_database(self.db_name)

    def create_collection(self, collection_name, fields_list, num_partitions=1024):
        """
        创建数据库集合
        :param collection_name: 集合名称
        :param fields_list: 字段信息
        :param num_partitions: 分区数
        :return:
        """
        self.__get_connect()
        if utility.has_collection(collection_name):
            return True
        fields = []
        index_params = []
        for field in fields_list:
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description']
            )

            if field.__contains__('dim'):
                temp_params = FieldSchema(
                    name=field['name'],
                    dtype=field['dtype'],
                    description=field['description'],
                    dim=field['dim'],
                )

            if field.__contains__('max_length'):
                temp_params = FieldSchema(
                    name=field['name'],
                    dtype=field['dtype'],
                    description=field['description'],
                    max_length=field['max_length'],
                )
            if field.__contains__('is_primary'):
                temp_params.is_primary = field['is_primary']
                temp_params.auto_id = field['auto_id']
            fields.append(temp_params)
            # 判断是否创建索引
            if field.__contains__('index_params'):
                # 向量索引
                if field['index_params']:
                    temp_params = {
                        "metric_type": field['index_params']['metric_type'],  # 'IP'
                        "index_type": field['index_params']['index_type'],  # "IVF_FLAT",
                        "params": {"nlist": field['index_params']['nlist']}  #
                    }
                # 标量索引
                else:
                    temp_params = None
                index_params.append((field['name'], 'index_' + field['name'], temp_params))

        schema = CollectionSchema(
            fields=fields,
            description="nstl serach"
        )
        # 创建集合
        collection = Collection(
            name=collection_name,
            schema=schema,
            using='default',
            shards_num=2,  # 要创建的集合的分片数。	[1,16]
            # num_partitions=num_partitions,  # 要创建的集合的逻辑分区数。	[1,4096]
            consistency_level=CONSISTENCY_LEVEL
        )
        # 集合创建索引
        for field_name, index_name, index_param in index_params:
            collection.create_index(
                field_name=field_name,
                index_params=index_param,
                index_name=index_name
            )
        collection.release()
        return True

    def create_index(self, collection_name, field_name, index_name, **kwargs):
        """
        创建索引
        :param collection_name:
        :param field_name:
        :param index_name:
        :param kwargs:
        :return:
        """
        self.__get_connect()
        if not utility.has_collection(collection_name):
            logger.warning(collection_name + ':集合不存在')
            return False
        collection = Collection(collection_name)
        try:
            temp_params = {
                "metric_type": kwargs.get('metric_type', 'IP'),  # 'IP'
                "index_type": kwargs.get('index_type', "IVF_FLAT"),
                "params": {"nlist": kwargs.get('index_type', 128)}
            }
            # 集合创建索引
            collection.create_index(
                field_name=field_name,
                index_params=temp_params,
                index_name=index_name
            )
        except:
            logger.error("创建索引异常")
            return False
        finally:
            collection.release()
        return True

    def insert_datas(self, datas, collection_name, partition_name=None):
        """
        插入数据
        :param collection_name:
        :param datas: 数据集 list/tuple/pandas.DataFrame
        :param partition_name: 分区名称
        :return:
        """
        try:
            res = self.client.insert(collection_name, datas)
            return res.__len__()
        except Exception as e:
            logger.error(collection_name + ":导入数据异常")
            logger.error(e)
            return 0

    def delete_datas(self, collection_name, id_list):
        try:
            self.client.delete(collection_name, id_list)
            return True
        except:
            logger.error(collection_name + ":数据删除异常")
            return False

    def search_datas(self, collection_name, vec_field, vec_text, output_fields, expr=None, start_num=0, top_n=20,
                     nprobe=32):
        """

        :param collection_name:
        :param vec_field:
        :param vec_text:
        :param output_fields:
        :param expr:
        :param start_num:
        :param top_n:
        :param nprobe:要搜索的群集单元数
        :return:
        """
        search_params = {
            "metric_type": METRIC_TYPE,
            "offset": start_num,  # the sum of `offset` in `param` and `limit` should be less than 16384.
            "ignore_growing": False,
            "params": {"nprobe": nprobe}
        }
        try:
            results = self.client.search(
                collection_name=collection_name,
                data=vec_text,  # 查询向量
                # anns_field=vec_field,  # 检索向量字段
                filter=expr,
                limit=start_num + top_n,  # 输出结果数量 # the sum of `limit` should be less than 16384.
                output_fields=output_fields,  # 输出字段名称
                search_params=search_params
                # consistency_level=CONSISTENCY_LEVEL  # 一致性级别
            )
            return results
        except Exception as e:
            logger.error(collection_name + ":数据查询异常")
            logger.debug(e)
            return None
