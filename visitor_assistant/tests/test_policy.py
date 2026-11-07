from pathlib import Path

from ingest.policy import ContentPolicy
from models import Audience


POLICY_PATH = Path(__file__).resolve().parents[1] / 'config' / 'content_policy.yaml'


def test_public_and_internal_mapping():
    policy = ContentPolicy.from_file(POLICY_PATH)
    public_ticket = policy.classify('1-产品与服务详情/迪士尼乐园与度假区 票务信息.docx')
    public_hotel = policy.classify('2-运营流程与标准作业程序/迪士尼乐园酒店信息 包括各家迪士尼酒店的房型.docx')
    public_booking = policy.classify('2-运营流程与标准作业程序/迪士尼官网订票：如何轻松规划你的梦幻之旅.docx')
    internal_complaint = policy.classify('2-运营流程与标准作业程序/迪士尼乐园度假区客诉处理流程.docx')
    internal_crm = policy.classify('2-运营流程与标准作业程序/迪士尼乐园的客户关系管理有哪些.docx')
    internal_staff = policy.classify('5-内部知识与工具/迪士尼员工票使用规则.doc')
    emergency = policy.classify('3-特殊情况与应急预案/迪士尼乐园「紧急情况处理」一站式速查表.docx')
    navigation = policy.classify('迪士尼RAG知识库.docx')

    assert public_ticket.audience is Audience.PUBLIC
    assert public_hotel.audience is Audience.PUBLIC
    assert public_booking.audience is Audience.PUBLIC
    assert internal_complaint.audience is Audience.INTERNAL
    assert internal_crm.audience is Audience.INTERNAL
    assert internal_staff.audience is Audience.INTERNAL
    assert emergency.audience is Audience.PUBLIC
    assert navigation.index is False


def test_redaction_removes_internal_details():
    policy = ContentPolicy.from_file(POLICY_PATH)
    text = '''游客安全指引：请前往服务台。
内部电话：021-12345678
责任人：张三
升级路径：值班经理 -> 宾客关系部
员工联系方式：工号 8899 企业微信 abc
普通票价信息保留。'''
    redacted, records = policy.redact(text)
    assert '021-12345678' not in redacted
    assert '张三' not in redacted
    assert '宾客关系部' not in redacted
    assert '8899' not in redacted
    assert '游客安全指引' in redacted
    assert '普通票价信息保留' in redacted
    assert {record['type'] for record in records} >= {
        'internal_phone', 'responsible_person', 'escalation_path', 'staff_contact'
    }
