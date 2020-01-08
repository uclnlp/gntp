# -*- coding: utf-8 -*-

from gntp.parse import clauses


def parse_clause(text):
    parsed = clauses.grammar.parse(text)
    return clauses.ClauseVisitor().visit(parsed)
