#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/5/31 10:26
@Author  : st
@File    : graph_input_zbn.py
@Project : LLM_and_mlivus_api_medical
@Software: PyCharm
@文件说明: 

"""

# from neo4j import Graph, Node, Relationship
from py2neo import Graph, Node, Relationship
import csv

# graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "admin"))
graph = Graph("bolt://127.0.0.1:7688", auth=("neo4j", "neo4j@123"))
# graph = Graph("bolt://127.0.0.1:7687", auth=("neo4j", "neo4j@123"))

# 清空整个Neo4j数据库
graph.run("MATCH (n) DETACH DELETE n")
print("已清空整个Neo4j数据库")

# 查找或创建节点
def get_or_create_node(label, name, doctor_name='赵炳南'):
    '''
    该函数用于根据给定的标签（label）和名称（name）查找或创建节点。
    首先，它尝试在数据库中查找已存在的具有指定标签和名称的节点。
    如果找到匹配的节点，它将返回找到的节点。
    如果未找到匹配的节点，它将创建一个带有指定标签和名称的新节点，然后将其添加到数据库中，并返回新节点。
    '''
    # 查找已存在的节点
    existing_node = graph.nodes.match(label, name=name, doctor_name=doctor_name).first()
    if existing_node:
        return existing_node
    else:
        # 如果节点不存在，则创建一个新节点
        new_node = Node(label, name=name, doctor_name=doctor_name)
        graph.create(new_node)
        return new_node



def establish_relationship(graph, subject_node_label, property_name, relation_name, reverse_property_name, object_node_label, doctor_name='赵炳南'):
    # 获取所有具有给定属性的节点
    # nodes_with_property = graph.nodes.match(subject_node_label).where(f"exists(_.`{property_name}`)")
    nodes_with_property = graph.nodes.match(subject_node_label).where(f"_.`{property_name}` IS NOT NULL")

    # 遍历具有属性的节点
    for node in nodes_with_property:
        # 获取节点的属性值
        property_value = node[property_name]

        object_node = graph.nodes.match(object_node_label, name=property_value, doctor_name=doctor_name).first()

        # 检查 object_node 不为 None 且关系不存在
        if object_node and not graph.exists(Relationship(node, relation_name, object_node, doctor_name=doctor_name)):
            # 创建关系
            relationship = Relationship(node, relation_name, object_node, doctor_name=doctor_name)
            graph.create(relationship)

            reverse_relationship = Relationship(object_node, reverse_property_name, node, doctor_name=doctor_name)
            graph.create(reverse_relationship)

def create_node_and_properties(graph, label, name_column, excluded_columns, row, doctor_name='赵炳南'):
    # 获取或创建节点
    node = get_or_create_node(label, row[name_column], doctor_name=doctor_name)

    # 添加属性
    for key, value in row.items():
        if key not in excluded_columns:
            node[key] = value

    # 将节点推送到图数据库
    graph.push(node)



def load_nodes_from_csv_file(graph, csv_file_path, label, name_column, excluded_columns):
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            create_node_and_properties(graph, label, name_column, excluded_columns, row)


def input_graph():
    # 读取CSV文件
    csv_file_path = "./zhaobingnan/graph.csv"

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 创建节点
            # subject_node = Node(row["subject_type"], name=row["subject"])
            # value_node = Node(row["value_type"], name=row["value"])
            subject_node = get_or_create_node(row["subject_type"], row["subject"], doctor_name='赵炳南')
            value_node = get_or_create_node(row["value_type"], row["value"], doctor_name='赵炳南')

            # 创建关系
            relationship_type = row["property"]
            relationship = Relationship(subject_node, relationship_type, value_node, doctor_name='赵炳南')

            # 添加属性
            for key, value in row.items():
                if key not in ["subject_type", "subject", "property", "value_type", "value"]:
                    relationship[key] = value
            relationship['doctor_name'] = '赵炳南'

            # 将节点和关系添加到图中
            graph.create(subject_node | value_node | relationship)
    print("已创建节点和关系:graph")

def input_jyf():
    # 读取CSV文件
    csv_file_path = "./zhaobingnan/经验方.csv"

    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 创建经验方节点
            experience_node = get_or_create_node("经验方", row["方名"])
            # 添加经验方的属性
            for key, value in row.items():
                if key not in ["ID", "方名"]:
                    experience_node[key] = value
            # existing_node.update({key: value for key, value in row.items() if key not in ["ID", 方名"]})
            # graph.push(existing_node)
            graph.push(experience_node)
            doctor_node = graph.nodes.match("名医", name=row["名医"], doctor_name='赵炳南').first()
            if doctor_node:
                # 创建关系
                expertise_relationship = Relationship(experience_node, "名医", doctor_node, doctor_name='赵炳南')
                # 将节点和关系添加到图中
                graph.create(expertise_relationship)
            # 查找已存在的方剂节点
            existing_formulae_node = graph.nodes.match("方剂", name=row["方名"], doctor_name='赵炳南').first()
            if existing_formulae_node:
                # 创建关系
                formulae_relationship = Relationship(experience_node, "相关方剂", existing_formulae_node,
                                                     doctor_name='赵炳南')
                # 将节点和关系添加到图中
                graph.create(formulae_relationship)
    print("已创建节点和关系:经验方")


input_graph()
input_jyf()

load_nodes_from_csv_file(graph, "./zhaobingnan/临床研究.csv", "临床研究", "研究名称", ["ID", "研究名称"])
establish_relationship(graph, "临床研究", "名医", "名医", "相关研究", "名医")
print("已创建节点和关系:临床研究")
load_nodes_from_csv_file(graph, "./zhaobingnan/临证经验.csv", "临证经验", "ID", ["ID"])
establish_relationship(graph, "临证经验", "人名", "名医", "有临证经验", "名医")
print("已创建节点和关系:临证经验")
load_nodes_from_csv_file(graph, "./zhaobingnan/特色用药.csv", "特色用药", "ID", ["ID"])
establish_relationship(graph, "特色用药", "名医", "名医", "有用药经验", "名医")
print("已创建节点和关系:特色用药")
load_nodes_from_csv_file(graph, "./zhaobingnan/学术思想.csv", "学术思想", "学术思想核心", ["ID"])
establish_relationship(graph, "学术思想", "名医", "名医", "有学术思想", "名医")
print("已创建节点和关系:学术思想")
load_nodes_from_csv_file(graph, "./zhaobingnan/医案.csv", "医案", "ID", ["ID"])
establish_relationship(graph, "医案", "名医", "名医", "有医案", "名医")
print("已创建节点和关系P:医案")