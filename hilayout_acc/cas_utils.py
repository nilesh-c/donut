from collections import namedtuple
import re

token = namedtuple("cas_token", ['begin', 'end', 'id', 'token_text', 'x0', 'y0', 'x1', 'y1'])
entity = namedtuple("cas_entity", ['begin', 'end', 'label'])
relation = namedtuple("cas_relation", ['subject', 'object', 'predicate'])

def get_covered_tokens(cas, page):
    cas_tokens = cas.select_covered('de.fraunhofer.iais.kd.textmining.types.Position', page)
    for i, t in enumerate(cas_tokens):
        token_text = cas.sofa_string[t.begin:t.end]
        annotation = token(t.begin - page.begin, t.end - page.begin, i, token_text,
                           t.x, t.y, t.x + t.width, t.y + t.height)
        yield annotation

def get_tokens(cas):
    cas_tokens = cas.select('de.fraunhofer.iais.kd.textmining.types.Token')
    for i, t in enumerate(cas_tokens):
        token_text = cas.sofa_string[t.begin:t.end]
        annotation = token(t.begin, t.end, i, token_text,
                           0,0,0,0)
        yield annotation

def get_tokens_re(cas):
    for i, match in enumerate(re.finditer("\w+", cas.sofa_string)):
        token_text = match.group()
        begin, end = match.span()
        annotation = token(begin, end, i, token_text,
                           0,0,0,0)
        yield annotation

def get_pages(cas):
    return cas.select('de.fraunhofer.iais.kd.textmining.types.Page')

def get_covered_entities(cas, page):
    cas_entities = cas.select_covered('de.fraunhofer.iais.kd.textmining.types.NamedEntity', page)
    for x in cas_entities:
        entity = (x.begin - page.begin, x.end - page.begin, x.entityType)
        yield entity

def get_entities(cas):
    cas_entities = cas.select('de.fraunhofer.iais.kd.textmining.types.NamedEntity')
    for x in cas_entities:
        e = entity(x.begin, x.end, x.entityType)
        if x.entityType == 'Page':
            continue
        yield e

def get_relations(cas):
    for rel in cas.select("de.fraunhofer.iais.kd.textmining.types.BiRelation"):
        make_entity_tuple = lambda x: entity(x.begin, x.end, x.entityType)
        subject_entity = make_entity_tuple(rel.relata1)
        object_entity = make_entity_tuple(rel.relata2)
        yield relation(subject_entity, object_entity, rel.predicate.predicateType)

def get_entities_with_scores(cas):
    cas_entities = cas.select('de.fraunhofer.iais.kd.textmining.types.NamedEntity')
    for x in cas_entities:
        entity = (x.begin, x.end, x.entityType, x.score)
        if x.entityType == 'Page':
            continue
        yield entity
