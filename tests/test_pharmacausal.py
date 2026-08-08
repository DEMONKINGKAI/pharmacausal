"""Unit tests for the pure, dependency-light logic in the pipeline.

Deliberately narrow: these cover the functions where a subtle bug would
silently corrupt downstream results without ever raising an error --
drug-name normalization, age-unit conversion, background-knowledge tier
assignment, and PC adjacency-matrix edge parsing. They don't touch the
FAERS download, parquet I/O, or the causal-learn discovery calls
themselves, which need real data or heavy dependencies to exercise
meaningfully.
"""

import numpy as np
import pandas as pd
import pytest

from pharmacausal.discovery import node_tier
from pharmacausal.features import age_to_years, normalize_ingredient
from pharmacausal.graph_export import drug_event_edges, pc_graph_to_edges


class TestNormalizeIngredient:
    def test_plain_ingredient_uppercased(self):
        assert normalize_ingredient("aspirin") == ["ASPIRIN"]

    def test_salt_suffix_stripped(self):
        assert normalize_ingredient("AMLODIPINE BESYLATE") == ["AMLODIPINE"]
        assert normalize_ingredient("METFORMIN HYDROCHLORIDE") == ["METFORMIN"]

    def test_combo_drug_exploded_on_backslash(self):
        assert normalize_ingredient("CARBIDOPA\\LEVODOPA") == ["CARBIDOPA", "LEVODOPA"]

    def test_combo_with_salts_on_each_component(self):
        assert normalize_ingredient("AMLODIPINE BESYLATE\\ATORVASTATIN CALCIUM") == \
            ["AMLODIPINE", "ATORVASTATIN"]

    def test_internal_whitespace_collapsed(self):
        assert normalize_ingredient("  vitamin   d  ") == ["VITAMIN D"]

    def test_bare_salt_name_drops_to_empty(self):
        # a component that's nothing but a salt suffix shouldn't produce
        # a spurious empty-string node
        assert normalize_ingredient("SODIUM") == []


class TestAgeToYears:
    def test_years_passthrough(self):
        s = age_to_years(pd.Series(["45"]), pd.Series(["YR"]))
        assert s.iloc[0] == pytest.approx(45.0)

    def test_decades_converted(self):
        s = age_to_years(pd.Series(["4"]), pd.Series(["DEC"]))
        assert s.iloc[0] == pytest.approx(40.0)

    def test_months_converted(self):
        s = age_to_years(pd.Series(["24"]), pd.Series(["MON"]))
        assert s.iloc[0] == pytest.approx(2.0)

    def test_missing_age_is_nan(self):
        s = age_to_years(pd.Series([None]), pd.Series(["YR"]))
        assert pd.isna(s.iloc[0])

    def test_unrecognized_unit_is_nan_not_silently_wrong(self):
        s = age_to_years(pd.Series(["10"]), pd.Series(["XYZ"]))
        assert pd.isna(s.iloc[0])


class TestNodeTier:
    """Background-knowledge tiers: a wrong tier here would silently let
    causal discovery orient a reaction as a cause of its own drug exposure
    again, which is the exact bug this ordering was added to fix."""

    def test_demographics_and_report_metadata_are_tier_0(self):
        assert node_tier("age_bin") == 0
        assert node_tier("sex_female") == 0
        assert node_tier("reporter_us") == 0
        assert node_tier("report_expedited") == 0

    def test_indication_and_polypharmacy_are_tier_1(self):
        assert node_tier("indication__Asthma") == 1
        assert node_tier("n_drugs_bin") == 1
        assert node_tier("n_suspect_drugs_bin") == 1

    def test_drug_is_tier_2(self):
        assert node_tier("drug__ASPIRIN") == 2

    def test_event_is_tier_3_and_therefore_terminal(self):
        assert node_tier("event__Headache") == 3
        assert node_tier("event__Headache") > node_tier("drug__ASPIRIN")

    def test_unrecognized_column_raises_instead_of_defaulting(self):
        with pytest.raises(ValueError):
            node_tier("some_unmapped_column")


class TestPcGraphToEdges:
    def test_directed_edge_orientation_preserved(self):
        # per CausalGraph convention: graph[j,i]==1 and graph[i,j]==-1 means i --> j
        names = ["drug__X", "event__Y"]
        graph = np.array([[0, -1], [1, 0]])
        edges = pc_graph_to_edges(names, graph)
        assert len(edges) == 1
        assert edges.iloc[0]["relation"] == "drug__X --> event__Y"

    def test_undirected_edge(self):
        names = ["A", "B"]
        graph = np.array([[0, -1], [-1, 0]])
        edges = pc_graph_to_edges(names, graph)
        assert edges.iloc[0]["relation"] == "A --- B"

    def test_no_edge_when_matrix_entries_are_zero(self):
        names = ["A", "B"]
        graph = np.zeros((2, 2), dtype=int)
        assert len(pc_graph_to_edges(names, graph)) == 0

    def test_drug_event_filter_extracts_clean_names(self):
        df = pd.DataFrame({
            "node1": ["drug__ASPIRIN", "drug__A", "event__X"],
            "node2": ["event__Headache", "drug__B", "event__Y"],
        })
        out = drug_event_edges(df)
        assert len(out) == 1
        assert out.iloc[0]["drug"] == "ASPIRIN"
        assert out.iloc[0]["event"] == "Headache"
