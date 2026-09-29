"""Boundary tests for the annotation-only negative-pool contract."""
import numpy as np

from scripts.experiments import tts_negative_pool as p


def test_rank_ties_and_common_distance_translation():
    assert p.rank(np.array([1., 2., 3.]), 2.) == 0.5
    x = np.array([1., 2.])
    y = np.array([[2., 4.], [0., 2.]])
    np.testing.assert_allclose(p.dist(x,y),p.dist(x+100,y+100))


def test_categories_distinguish_occurrence_and_phone():
    query={"event":"phone_0","phone":"a"}
    assert p.category(query,query)=="same_occurrence"
    assert p.category(query,{"event":"phone_1","phone":"a"})=="same_phone_other"
    assert p.category(query,{"event":"phone_1","phone":"b"})=="different_phone"
    assert p.category(query,{"event":"gap_0","phone":"__gap__"})=="silence_gap"


def test_support_requires_both_clocks_and_annotation_duration(monkeypatch):
    # The second arm has twice the duration: donor inclusion must satisfy BOTH.
    records={"n":[("a",i*.2,(i+1)*.2) for i in range(20)],
             "t":[("a",i*.4,(i+1)*.4) for i in range(20)]}
    monkeypatch.setattr(p,"phones",lambda path:records[path])
    s=p.make_support({"id":"synthetic","speaker":"s"},{"N":{"path":"n"},"T":{"path":"t"}}, {"N":100,"T":200},1)
    assert s["eligible"]
    for query in s["queries"]:
        node=s["nodes"][query["node"]]
        for dn in query["donors"]:
            donor=s["nodes"][dn]
            assert all(3<=abs(node["j"][a]-donor["j"][a])<=15 for a in p.ARMS)
    assert s["rejected"]["event_duration"]==19  # zero-length gaps


def test_short_phone_has_no_fallback_node(monkeypatch):
    monkeypatch.setattr(p,"phones",lambda path:[("a",0.,.079)])
    s=p.make_support({"id":"synthetic","speaker":"s"},{a:{"path":a} for a in p.ARMS},{a:100 for a in p.ARMS},1)
    assert not s["nodes"] and not s["eligible"]
