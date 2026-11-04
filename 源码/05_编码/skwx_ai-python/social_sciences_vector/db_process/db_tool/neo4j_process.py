#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/28 10:39
@Author  : st
@File    : neo4j_process.py
@Project : LLM_and_mlivus_api_medical
@Software: PyCharm
@文件说明: 

"""
import json
import traceback

from neo4j import GraphDatabase

from constant import NEO4J_URI, NEO4J_USER, NEO4J_PWD
from log.logger import Logger

log = Logger()


class Neo4jConnection:

    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(user, pwd))
        except Exception as e:
            log.error("Failed to create the driver: %s" % e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, parameters=None, db=None):
        # assert self.__driver is not None, "Driver not initialized!"
        # session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            log.error("Query failed: %s" % traceback.print_exc())
        # finally:
        # if session is not None:
        #     session.close()
        return response


conn = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PWD)


def get_neo4j_datas(keywords, docter_name, limit=100, level=1):
    """
    搜索节点
    :param keywords: 关键词
    :param docter_name: 医生名称
    :param level: 层级
    :param limit: 限制
    :param kwargs: 其他参数
    :return: 结果
    """
    triples = []
    for keyword in keywords:
        try:
            query = f"MATCH (n) -[r*1..{level}]-> (m)  WHERE n.name CONTAINS '{keyword}' AND n.doctor_name = '{docter_name}' RETURN n,r,m "  # LIMIT {limit} "
            result = conn.query(query)
            for item in result:
                for rel in item['r']:

                    start_node = dict(item['n'])
                    start_node['label'] = set(item['n'].labels).pop()
                    end_node = dict(item['m'])
                    end_node['label'] = set(item['m'].labels).pop()
                    if rel.type == '有医案' or start_node['label'] == '医案' or end_node['label'] == '医案':
                        continue
                    if len(triples) >= limit:
                        break

                    triples.append((start_node, rel.type, end_node))
        except Exception:
            log.error(traceback.print_exc())
    log.debug('neo4j results: ' + str(triples))
    return triples


def expand_nodes(results, node, level, docter_name, limit=20):
    """
    展开节点
    :param results: 结果
    :param node: 节点
    :param level: 层级
    :param kwargs: 其他参数
    :return: 结果
    """
    try:
        if level <= 0:
            return
        query = f"MATCH (n)-[r]-(m) WHERE id(n) = {node.id} AND m.doctor_name = '{docter_name}' RETURN n, r, m LIMIT {limit}"
        relations = conn.query(query)
        if not relations:
            return
        for node_n, rel, node_m in relations:
            rel_dict = rel_node_2_dict(rel, "RELATION")
            start_node_dict = rel_node_2_dict(node_n, "NODE")
            end_node_dict = rel_node_2_dict(node_m, "NODE")
            if not (rel_dict and start_node_dict and end_node_dict):
                continue
            results.append({"开始节点": start_node_dict, "关系": rel_dict, "结束节点": end_node_dict})
            expand_nodes(results, node_m, level - 1, docter_name)
    except Exception:
        log.error(traceback.print_exc())


def rel_node_2_dict(object, obj_type):
    """
    转换关系节点
    :param object: 关系节点
    :param obj_type: 节点类型
    :return: 关系节点字典
    """
    temp_dict = dict()
    try:
        obj_properties = object._properties
        if not obj_properties:
            return None
        if obj_type == "NODE":
            # temp_dict["节点ID"] = object.element_id
            temp_dict["节点ID"] = object.id
            temp_dict["节点标签"] = next(iter(object.labels))
            temp_dict["节点名称"] = obj_properties["name"]
        elif obj_type == "RELATION":
            # temp_dict["关系ID"] = object.element_id
            temp_dict["关系ID"] = object.id
            temp_dict["关系类型"] = object.type
        for key in obj_properties.keys():
            if key in ["name", "date", "id", "description", "doctor_name"]:
                continue
            value = obj_properties[key].replace('\n', '\\n').strip()
            if not value:
                continue
            temp_dict[key] = value
    except Exception:
        log.error(traceback.print_exc())
    finally:
        return temp_dict


# 测试代码
if __name__ == "__main__":
    keywords = ["消痈汤"]
    level = 1
    results = get_neo4j_datas(keywords, "赵炳南")
    print(json.dumps(results, ensure_ascii=False))
