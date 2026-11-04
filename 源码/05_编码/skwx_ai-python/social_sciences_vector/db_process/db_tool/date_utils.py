# -*- coding: utf-8 -*-
import re
import time as tm
import time


def time_to_str(ut, f=None):
    """
    时间转字符串
    :param f:
    :param ut:秒
    :return: 时间戳字符串
    """
    if f is None:
        f = '%Y-%m-%d %H:%M:%S'
    t = time.strftime(f, tm.localtime(ut))
    # t = datetime.fromtimestamp(ut, f)
    return str(t)


def str_to_time(s, f=None):
    """
    时间戳字符串转换为时间秒数
    :param s:字符串
    :param f:时间戳格式
    :return:
    """
    if f is None:
        f = '%Y-%m-%d %H:%M:%S'
    return time.mktime(time.strptime(s, f))


def time_format(s, old_f, new_f):
    """
    时间戳格式化
    :param s: 时间字符串
    :param old_f: 当前格式
    :param new_f: 转换的格式
    :return:
    """
    if not (old_f or new_f):
        return s
    return time_to_str(str_to_time(s, old_f), new_f)


def get_time_str():
    return time_to_str(time.time())

year_pat = r'(?P<Year>\d{4})年(?:\d{1,2}月)?(?:\d{1,2}日)?'


def date_fromat_str(s, ct=''):
    """格式格式的日期格式化称中文 年月日"""
    if not s:
        return ''
    s = s.replace('年', '/').replace('月', '/').replace('日', '')
    if s.endswith('/'):
        s = s[:-1]
    pat1 = r'^\d{4}/\d{1,2}/\d{1,2}'
    rs = re.findall(pat1, s)
    if rs:
        return __date_exc(time_format(rs[0], '%Y/%m/%d', '%YYear%mMon%dDay'))
    pat2 = r'^\d{4}/\d{1,2}$'
    rs = re.findall(pat2, s)
    if rs:
        return __date_exc(time_format(rs[0], '%Y/%m', '%YYear%mMon'))
    pat3 = r'^\d{4}$'
    rs = re.findall(pat3, s)
    if rs:
        return __date_exc(time_format(rs[0], '%Y', '%YYear'))
    pat4 = r'^\d{4}-\d{1,2}-\d{1,2}'
    rs = re.findall(pat4, s)
    if rs:
        return __date_exc(time_format(rs[0], '%Y-%m-%d', '%YYear%mMon%dDay'))
    pat5 = r'^\d{4}-\d{1,2}$'
    rs = re.findall(pat5, s)
    if rs:
        return __date_exc(time_format(rs[0], '%Y-%m', '%YYear%mMon'))
    # 实体中没有 年份，获取创建时间中的年份
    years = re.match(year_pat, ct)
    if not years:
        return ''
    year = years.group('Year')
    pat6 = r'^\d{1,2}/\d{1,2}'
    rs = re.findall(pat6, s)
    if rs:
        return __date_exc(time_format(year + '/' + rs[0], '%Y/%m/%d', '%YYear%mMon%dDay'))
    pat7 = r'^\d{1,2}-\d{1,2}'
    rs = re.findall(pat7, s)
    if rs:
        return __date_exc(time_format(year + '-' + rs[0], '%Y-%m-%d', '%YYear%mMon%dDay'))
    return ''


def date_fromat_time(s):
    """格式格式的日期格式化称中文 年月日"""
    if not s:
        return None
    s = s.replace('年', '/').replace('月', '/').replace('日', '').replace('-', '/').replace('.', '/')
    s = s[:-1] if s.endswith('/') else s
    rs = re.findall(r'^\d{4}/\d{1,2}/\d{1,2}', s)
    if rs:
        return str_to_time(rs[0], '%Y/%m/%d')
    rs = re.findall(r'^\d{4}/\d{1,2}$', s)
    if rs:
        return str_to_time(rs[0], '%Y/%m')
    rs = re.findall(r'^\d{4}$', s)
    if rs:
        return str_to_time(rs[0], '%Y')
    rs = re.findall(r'^\d{4}-\d{1,2}-\d{1,2}', s)
    if rs:
        return str_to_time(rs[0], '%Y-%m-%d')
    rs = re.findall(r'^\d{4}-\d{1,2}$', s)
    if rs:
        return str_to_time(rs[0], '%Y-%m')
    return None


def __date_exc(d):
    return d.replace('Year', '年').replace('Mon', '月').replace('Day', '日')


if __name__ == '__main__':
    print(get_time_str())

