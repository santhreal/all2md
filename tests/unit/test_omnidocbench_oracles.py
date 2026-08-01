"""Behavioral tests for direct annotation and AST fidelity projections."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from all2md.ast.nodes import (
    Code,
    CodeBlock,
    Document,
    Heading,
    MathBlock,
    MathInline,
    Paragraph,
    Table,
    TableCell,
    TableRow,
    Text,
)
from benchmarks.omnidocbench import oracles
from benchmarks.omnidocbench.oracles import (
    FormulaProjection,
    PageProjection,
    TableProjection,
    content_similarity,
    project_annotation,
    project_ast,
    score_page,
)

pytestmark = pytest.mark.unit


def test_annotation_projection_uses_order_and_supported_ground_truth() -> None:
    """External facts must come from annotation fields, not from an all2md renderer."""
    record = {
        "page_info": {"image_path": "nested/page-7.jpg"},
        "layout_dets": [
            {
                "category_type": "text_block",
                "order": 3,
                "text": "Body",
                "ignore": False,
                "line_with_spans": [
                    {"category_type": "text_span", "text": "Body"},
                    {"category_type": "equation_inline", "latex": "$a$"},
                    {"category_type": "equation_inline", "latex": "$ignored$", "ignore": True},
                ],
            },
            {"category_type": "code_txt", "order": 2, "text": "print(1)", "ignore": False},
            {
                "category_type": "table",
                "order": 4,
                "html": "<table><tr><th colspan='2'>Head</th></tr><tr><td>A</td><td>B</td></tr></table>",
                "ignore": False,
            },
            {"category_type": "title", "order": 1, "text": "Title", "ignore": False},
            {
                "category_type": "equation_isolated",
                "order": 5,
                "latex": "$$x + y$$",
                "ignore": False,
            },
            {
                "category_type": "figure",
                "order": 6,
                "ignore": False,
                "line_with_spans": [{"category_type": "equation_inline", "latex": "$b$"}],
            },
            {"category_type": "footer", "order": 7, "ignore": True},
        ],
    }

    truth = project_annotation(record)

    assert truth.page_id == "page-7"
    assert truth.projection.text_blocks == ("Title", "print(1)", "Body", "Head A B")
    assert truth.projection.block_kinds == (
        "title",
        "text_block",
        "text_block",
        "table",
        "equation_isolated",
    )
    assert truth.projection.tables == (TableProjection(2, 2, 4, "Head A B"),)
    assert truth.projection.formulas == (
        FormulaProjection("inline", "$a$"),
        FormulaProjection("block", "$$x + y$$"),
        FormulaProjection("inline", "$b$"),
    )
    assert truth.unscored_categories == {"figure": 1}
    assert truth.explicitly_ignored == 2


def test_ast_projection_reads_nodes_directly_in_document_order() -> None:
    """Scoring must inspect one AST and preserve its semantic block order."""
    table = Table(
        header=TableRow(cells=[TableCell(content=[Text(content="Head")], colspan=2)]),
        rows=[
            TableRow(
                cells=[
                    TableCell(content=[Text(content="A")]),
                    TableCell(content=[Text(content="B")]),
                ]
            )
        ],
    )
    document = Document(
        children=[
            Heading(level=1, content=[Text(content="Title")]),
            CodeBlock(content="print(1)"),
            Paragraph(
                content=[
                    Text(content="Body"),
                    MathInline(
                        content="<math>a</math>",
                        notation="mathml",
                        representations={"latex": "a"},
                    ),
                ]
            ),
            table,
            MathBlock(
                content="<math>x + y</math>",
                notation="mathml",
                representations={"latex": "x + y"},
            ),
        ]
    )

    projection = project_ast(document)

    assert projection.text_blocks == ("Title", "print(1)", "Body <math>a</math>", "Head A B")
    assert projection.block_kinds == (
        "title",
        "text_block",
        "text_block",
        "table",
        "equation_isolated",
    )
    assert projection.tables == (TableProjection(2, 2, 4, "Head A B"),)
    assert projection.formulas == (
        FormulaProjection("inline", "a"),
        FormulaProjection("block", "x + y"),
    )


def test_page_scores_have_exact_independent_dimension_semantics() -> None:
    """Each metric must react to the fact it names rather than a shared output-length proxy."""
    expected = PageProjection(
        text_blocks=("A B",),
        block_kinds=("title", "text_block", "table"),
        tables=(TableProjection(2, 2, 4, "A B"),),
        formulas=(FormulaProjection("block", "$$x$$"),),
    )
    actual = PageProjection(
        text_blocks=("AB",),
        block_kinds=("title", "table"),
        tables=(TableProjection(1, 2, 2, "AB"),),
        formulas=(),
    )

    scores = score_page(expected, actual)

    assert scores == {
        "text_content_similarity": pytest.approx(2 / 3),
        "reading_order_similarity": pytest.approx(2 / 3),
        "formula_presence_accuracy": 0.0,
        "table_structure_similarity": pytest.approx(2 / 3),
        "table_content_similarity": pytest.approx(2 / 3),
        "formula_content_similarity": 0.0,
    }


def test_latin_word_boundary_loss_is_penalized_but_cjk_glyph_spacing_is_not() -> None:
    """Losing every inter-word space must cost score; OCR's inter-glyph CJK spaces must not.

    Deleting all whitespace made 'thequickbrownfox' score a perfect 1.0 against
    'the quick brown fox' on the 316 predominantly-Latin pinned pages, so a parser that
    regressed to space-free extraction kept full credit. The CJK rationale is real -- OCR
    inserts spurious spaces between ideographs -- so whitespace is deleted only where it
    touches ideographic text and collapsed to a single space everywhere else.
    """
    latin = PageProjection(("the quick brown fox",), ("text_block",), (), ())
    spaceless = PageProjection(("thequickbrownfox",), ("text_block",), (), ())
    assert score_page(latin, spaceless)["text_content_similarity"] < 0.85

    cjk = PageProjection(("发票金额合计",), ("text_block",), (), ())
    spaced_cjk = PageProjection(("发 票 金 额 合 计",), ("text_block",), (), ())
    assert score_page(cjk, spaced_cjk)["text_content_similarity"] == 1.0


def test_every_text_bearing_annotation_category_is_ground_truth() -> None:
    """A parser must not score higher for dropping captions, headers, footers, or references.

    The dataset annotates text in ten categories beyond text_block/title/code_txt. Scoring
    only that subset compared a filtered ground truth against the whole-page AST, so losing
    9.3% of the corpus text scored 1.0 while perfect fidelity scored 0.82 -- and on 63 pinned
    pages perfect fidelity scored exactly 0.0. The projection must cover the whole page.
    """
    assert {
        "equation_caption",
        "figure_caption",
        "figure_footnote",
        "footer",
        "header",
        "page_footnote",
        "page_number",
        "reference",
        "table_caption",
        "table_footnote",
    } <= oracles.TEXT_CATEGORIES

    record = {
        "page_info": {"page_no": 0, "image_path": "p.jpg", "height": 10, "width": 10},
        "layout_dets": [
            {"category_type": "header", "text": "Journal of Things", "order": 0},
            {"category_type": "text_block", "text": "Body", "order": 1},
            {"category_type": "figure_caption", "text": "Figure 1. A plot", "order": 2},
            {"category_type": "page_number", "text": "7", "order": 3},
        ],
    }

    truth = oracles.project_annotation(record)

    assert truth.projection.text_blocks == ("Journal of Things", "Body", "Figure 1. A plot", "7")
    assert truth.projection.block_kinds == ("text_block",) * 4
    assert truth.unscored_categories == {}

    dropped = PageProjection(("Body",), ("text_block",), (), ())
    assert score_page(truth.projection, dropped)["text_content_similarity"] < 0.3


def test_formula_similarity_preserves_latex_case_and_semantic_whitespace() -> None:
    """Text normalization must not conflate distinct case-sensitive LaTeX."""
    expected = PageProjection((), (), (), (FormulaProjection("block", r"\Gamma"),))

    case_score = score_page(
        expected,
        PageProjection((), (), (), (FormulaProjection("block", r"\gamma"),)),
    )
    whitespace_score = score_page(
        PageProjection((), (), (), (FormulaProjection("block", r"\text{a b}"),)),
        PageProjection((), (), (), (FormulaProjection("block", r"\text{ab}"),)),
    )
    delimited_score = score_page(
        PageProjection((), (), (), (FormulaProjection("block", "$$x + y$$"),)),
        PageProjection((), (), (), (FormulaProjection("block", "x + y"),)),
    )

    assert case_score["formula_content_similarity"] < 1.0
    assert whitespace_score["formula_content_similarity"] < 1.0
    assert delimited_score["formula_content_similarity"] == 1.0


def test_table_alignment_recovers_later_exact_matches_without_reordering() -> None:
    """An early omission must not shift every later table comparison, and swaps must not score as exact."""
    alpha = TableProjection(1, 1, 1, "alpha")
    beta = TableProjection(1, 1, 1, "beta")
    expected = PageProjection((), ("table", "table"), (alpha, beta), ())

    omitted = score_page(expected, PageProjection((), ("table",), (beta,), ()))
    reversed_order = score_page(expected, PageProjection((), ("table", "table"), (beta, alpha), ()))
    exact = score_page(expected, PageProjection((), ("table", "table"), (alpha, beta), ()))

    assert omitted["table_content_similarity"] == pytest.approx(0.5)
    assert exact["table_content_similarity"] == 1.0
    assert reversed_order["table_content_similarity"] == pytest.approx(0.2)
    assert reversed_order["table_structure_similarity"] == 1.0


def test_formula_metrics_are_absent_when_no_page_side_has_a_formula() -> None:
    """A parser with no math capability must not bank credit from formula-free pages."""
    plain = PageProjection(("Body",), ("text_block",), (), ())

    assert score_page(plain, plain) == {
        "text_content_similarity": 1.0,
        "reading_order_similarity": 1.0,
    }

    missed = score_page(
        PageProjection((), (), (), (FormulaProjection("block", "a"),)),
        plain,
    )
    assert missed["formula_presence_accuracy"] == 0.0
    assert missed["formula_content_similarity"] == 0.0


def test_formula_alignment_matches_only_like_kinds_in_document_order() -> None:
    """Inline and block formulas are distinct even when their LaTeX is identical."""
    expected = PageProjection(
        (),
        (),
        (),
        (
            FormulaProjection("inline", "a"),
            FormulaProjection("block", "b"),
        ),
    )

    omitted_inline = score_page(
        expected,
        PageProjection((), (), (), (FormulaProjection("block", "b"),)),
    )
    wrong_kind = score_page(
        PageProjection((), (), (), (FormulaProjection("inline", "x"),)),
        PageProjection((), (), (), (FormulaProjection("block", "x"),)),
    )

    assert omitted_inline["formula_content_similarity"] == pytest.approx(0.5)
    assert wrong_kind["formula_presence_accuracy"] == 1.0
    assert wrong_kind["formula_content_similarity"] == 0.0


def test_text_similarity_normalizes_unicode_and_layout_whitespace() -> None:
    """CJK OCR spacing and compatibility glyphs must not masquerade as content loss."""
    assert content_similarity("Ａ 中\n文", "a中文") == 1.0
    assert content_similarity("abc", "") == 0.0
    assert content_similarity("", "") == 1.0


def test_text_similarity_is_exact_normalized_levenshtein() -> None:
    """Substitutions and insertions must have the documented edit-distance cost."""
    assert content_similarity("kitten", "sitting") == pytest.approx(4 / 7)


def _reference_edit_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


@pytest.mark.fuzzing
@settings(deadline=None)
@given(
    st.text(alphabet="abc中文", max_size=8),
    st.text(alphabet="abc中文", max_size=8),
)
def test_bit_parallel_edit_distance_matches_reference(left: str, right: str) -> None:
    """The optimized Unicode oracle must equal the simple dynamic-programming contract.

    The example budget is deliberately left to the repository's Hypothesis profile, so a pull
    request draws the ``dev`` count and the scheduled ``HYPOTHESIS_PROFILE=ci`` run draws more.
    Pinning ``max_examples`` here would override that profile on every leg, which is what #230
    measured as roughly 160 times more expensive in CI than locally.

    Short draws cannot exercise the bit-parallel pattern's carry behaviour, so
    :func:`test_edit_distance_is_exact_across_machine_word_boundaries` covers length explicitly.
    """
    assert oracles._edit_distance(left, right) == _reference_edit_distance(left, right)


@pytest.mark.parametrize("length", [1, 31, 32, 33, 63, 64, 65, 66, 127, 128, 129, 200])
def test_edit_distance_is_exact_across_machine_word_boundaries(length: int) -> None:
    """Length must not change the answer, including where a bit-parallel pattern would carry.

    ``_edit_distance`` is Myers' bit-vector algorithm over Python integers. Those are unbounded,
    so there is no 64-bit block to get wrong, but nothing proved that: the property test above
    draws at most 8 characters, so no number of examples ever crossed a word boundary. The pinned
    corpus carries strings up to 2482 characters, so the untested range is the operational one.
    """
    left = "ab中文x" * (length // 5 + 1)
    left = left[:length]
    assert oracles._edit_distance(left, left) == 0
    assert oracles._edit_distance(left, "") == length
    substituted = ("z" if left[-1] != "z" else "y") + left[1:]
    assert oracles._edit_distance(left, substituted) == _reference_edit_distance(left, substituted)
    deleted = left[:-1]
    assert oracles._edit_distance(left, deleted) == _reference_edit_distance(left, deleted)
    reversed_text = left[::-1]
    assert oracles._edit_distance(left, reversed_text) == _reference_edit_distance(left, reversed_text)


def test_recovering_table_text_beats_deleting_the_table() -> None:
    """A parser must never score higher for dropping a table than for recovering its cells.

    Ground truth put table cell text only in ``tables``, while the AST side left it in
    ``text_blocks`` whenever no real ``Table`` node was built. On all 317 pinned pages with table
    ground truth, deleting the table beat emitting its cells as a paragraph -- by a full point on
    some pages. Cell text now belongs to the text stream on both sides.
    """
    html = "<table><tr><td>Alpha</td><td>Beta</td></tr></table>"
    truth = oracles.project_annotation(
        {
            "page_info": {"page_no": 0, "image_path": "p.jpg", "height": 100, "width": 100},
            "layout_dets": [
                {"category_type": "text_block", "text": "Body", "order": 0, "poly": [0, 10] * 4},
                {"category_type": "table", "html": html, "order": 1, "poly": [0, 50] * 4},
            ],
        }
    ).projection

    assert truth.text_blocks == ("Body", "Alpha Beta")

    kept_as_text = PageProjection(("Body", "Alpha Beta"), ("text_block", "text_block"), (), ())
    deleted = PageProjection(("Body",), ("text_block",), (), ())

    kept = score_page(truth, kept_as_text)["text_content_similarity"]
    dropped = score_page(truth, deleted)["text_content_similarity"]
    assert kept == 1.0
    assert dropped < kept


def test_unordered_running_content_is_placed_by_geometry() -> None:
    """Dropping a header must not outscore emitting it in its true position.

    Every header, footer, page number, and page footnote in the pinned dataset carries
    ``order: null``. Sorting those last appended running content after the body, so a parser that
    emitted the header first mismatched the ground truth and a parser that deleted the header
    matched it better -- a mean inversion on 663 of 981 pages. A category-only rank is not enough:
    146 of the 669 pinned page_number detections, about one in five, sit above the page midline.
    """
    record = {
        "page_info": {"page_no": 0, "image_path": "p.jpg", "height": 1000, "width": 800},
        "layout_dets": [
            {"category_type": "text_block", "text": "Body", "order": 5, "poly": [0, 400] * 4},
            {"category_type": "header", "text": "Journal", "order": None, "poly": [0, 40] * 4},
            {"category_type": "footer", "text": "Copyright", "order": None, "poly": [0, 960] * 4},
            {"category_type": "page_number", "text": "7", "order": None, "poly": [0, 20] * 4},
        ],
    }

    truth = oracles.project_annotation(record).projection

    assert truth.text_blocks == ("7", "Journal", "Body", "Copyright")

    visual = PageProjection(truth.text_blocks, truth.block_kinds, (), ())
    without_header = PageProjection(("7", "Body", "Copyright"), ("text_block",) * 3, (), ())
    assert score_page(truth, visual)["text_content_similarity"] == 1.0
    assert score_page(truth, without_header)["text_content_similarity"] < 1.0


def test_reversed_output_cannot_score_a_perfect_reading_order() -> None:
    """Reading order must detect reordering, not just block-category coverage.

    Eleven text categories collapse to ``text_block``, so an edit similarity over the kind
    sequence returned exactly 1.0 for fully reversed output on 153 pinned pages, and every
    permutation was free on the 113 pages whose kinds are a single repeated token.
    """
    expected = PageProjection(("Alpha", "Beta", "Gamma"), ("text_block",) * 3, (), ())
    forward = PageProjection(("Alpha", "Beta", "Gamma"), ("text_block",) * 3, (), ())
    reversed_blocks = PageProjection(("Gamma", "Beta", "Alpha"), ("text_block",) * 3, (), ())

    assert score_page(expected, forward)["reading_order_similarity"] == 1.0
    assert score_page(expected, reversed_blocks)["reading_order_similarity"] == 0.0


def test_spurious_spaces_around_fullwidth_punctuation_are_ignored() -> None:
    """Correct CJK extraction must not be penalized for the OCR artefact the rule targets.

    NFKC folds fullwidth punctuation to ASCII, so running the whitespace deletion only after
    normalization left spurious spaces beside those glyphs in place and depressed the score on
    588 of the 743 pinned pages containing them; it now costs score on none. The substitution runs
    on both sides of NFKC.
    """
    tight = PageProjection(("（6）计算：",), ("text_block",), (), ())
    spaced = PageProjection((" （ 6 ） 计算 ： ",), ("text_block",), (), ())
    assert score_page(tight, spaced)["text_content_similarity"] == 1.0

    latin = PageProjection(("the quick brown fox",), ("text_block",), (), ())
    spaceless = PageProjection(("thequickbrownfox",), ("text_block",), (), ())
    assert score_page(latin, spaceless)["text_content_similarity"] < 0.85


def test_inline_equations_inside_code_are_not_ground_truth() -> None:
    """A ``CodeBlock`` holds a plain string, so a formula inside code can never be matched.

    Harvesting that span created ground truth no AST can satisfy, which showed up as the last
    residual monotonicity inversion: a degraded parser outscored perfect fidelity on formula
    presence for one pinned page.

    The span shape here is the one the pinned dataset actually uses: ``line_with_spans`` is a flat
    list of spans carrying ``category_type`` directly. All 69718 spans in the pinned annotation
    are that shape and none nests them under a ``spans`` key, so a nested fixture would be skipped
    before reaching the guard and would pass with the guard deleted.
    """
    record = {
        "page_info": {"page_no": 0, "image_path": "p.jpg", "height": 100, "width": 100},
        "layout_dets": [
            {
                "category_type": "code_txt",
                "text": "x = $a$",
                "order": 0,
                "poly": [0, 10] * 4,
                "line_with_spans": [{"category_type": "equation_inline", "latex": "$a$"}],
            }
        ],
    }

    truth = oracles.project_annotation(record)

    assert truth.projection.formulas == ()
    assert truth.projection.text_blocks == ("x = $a$",)


def test_identical_output_scores_a_perfect_reading_order_despite_repeated_blocks() -> None:
    """Byte-identical output must be the ceiling, so the order term cannot self-penalize.

    Matching each emitted block to its most similar ground-truth block by unconstrained argmax
    sent every repetition of one string to the same index, which registered as an inversion and
    capped self-comparison at 0.9973 across the pinned corpus. Ties now resolve towards the
    block's own position.

    The repeats must straddle a distinct block. With them adjacent, first-index argmax collapses
    them all to index 0 and the sequence 0 <= 0 <= 0 <= 3 carries no inversion, so the fixture
    passes with the tie-break deleted and pins nothing. Interleaving them makes the untie-broken
    argmax emit 0, 1, 0, which inverts: this fixture scores 1.0 with the tie-break and 0.6667
    without it.
    """
    page = PageProjection(("Same", "Tail", "Same"), ("text_block",) * 3, (), ())
    assert score_page(page, page)["reading_order_similarity"] == 1.0
    adjacent = PageProjection(("Same", "Same", "Same", "Tail"), ("text_block",) * 4, (), ())
    assert score_page(adjacent, adjacent)["reading_order_similarity"] == 1.0


def test_a_paragraph_holding_table_cells_costs_exactly_one_kind_substitution() -> None:
    """Recovering a table as a paragraph must be charged for segmentation and nothing else.

    The expected cost is one substitution in the block-kind coverage term, ``1 - 1/n``. Because
    table cell text now sits in both text streams, that variant's blocks are byte-identical to
    exact reproduction and sit at the same indices, so the order-agreement factor must be exactly
    1.0. Anything lower would mean the agreement term double-charges a variant whose order is
    right, and the ratchet would red the first time all2md starts recovering table cells.
    """
    truth = PageProjection(
        ("Intro", "A B"),
        ("text_block", "table"),
        (TableProjection(1, 2, 2, "A B"),),
        (),
    )
    paragraph = PageProjection(("Intro", "A B"), ("text_block", "text_block"), (), ())

    scores = score_page(truth, paragraph)
    assert scores["text_content_similarity"] == 1.0
    assert scores["reading_order_similarity"] == 1 - 1 / len(truth.block_kinds)


def test_table_structure_counts_spanning_cells_by_span_not_by_cell_object() -> None:
    """Pins colspan and rowspan arithmetic in both the AST and the HTML projection.

    The suite's other table fixture has a colspan in a row that is not the widest and uses no
    rowspan at all, so a colspan-blind column count and a rowspan-blind slot count both happen to
    produce the right answer there. Three independent mistakes therefore went unnoticed: counting
    ``len(row.cells)`` for columns, dropping rowspan from ``cell_slots``, and incrementing the HTML
    parser's column count by one per cell. This is load-bearing arithmetic: 109 of the 428 pinned
    tables carry a colspan above 1 and 96 carry a rowspan above 1. Here the spanning row IS the
    widest, so a span-blind count reports 2 columns instead of 3, and the rowspan makes cell_slots
    differ from the cell count.
    """
    table = Table(
        header=TableRow(cells=[TableCell(content=[Text(content="Wide")], colspan=3)]),
        rows=[
            TableRow(
                cells=[
                    TableCell(content=[Text(content="Tall")], rowspan=2),
                    TableCell(content=[Text(content="B")]),
                ]
            )
        ],
    )
    projected = project_ast(Document(children=[table])).tables[0]
    # 3 columns from the spanning header, 2 rows, slots = 3 + (1*2) + 1 = 6.
    assert (projected.rows, projected.columns, projected.cell_slots) == (2, 3, 6)

    html = "<table><tr><th colspan='3'>Wide</th></tr>" "<tr><td rowspan='2'>Tall</td><td>B</td></tr></table>"
    record = {
        "page_info": {"page_no": 0, "image_path": "p.jpg", "height": 1000, "width": 800},
        "layout_dets": [{"category_type": "table", "order": 0, "html": html, "ignore": False}],
    }
    from_html = project_annotation(record).projection.tables[0]
    assert (from_html.rows, from_html.columns, from_html.cell_slots) == (2, 3, 6)


def test_two_block_pages_are_still_ordered() -> None:
    """The order term must engage at two blocks, which is the smallest orderable page.

    39 pinned pages carry exactly two text blocks. Raising the ``< 2`` guard to ``< 3`` would make
    swapping them free on every one of those pages while every other test still passed, because
    the reversal test uses three blocks. Two blocks is the boundary where order first exists.
    """
    truth = PageProjection(("First", "Second"), ("text_block",) * 2, (), ())
    swapped = PageProjection(("Second", "First"), ("text_block",) * 2, (), ())

    assert score_page(truth, truth)["reading_order_similarity"] == 1.0
    assert score_page(truth, swapped)["reading_order_similarity"] == 0.0


def test_the_page_text_stream_keeps_a_separator_between_blocks() -> None:
    """Joining blocks without a separator would fuse adjacent words into one token.

    ``score_page`` concatenates a page's blocks into one text stream. Joining with the empty
    string makes a parser that splits one block into two score identically to one that keeps it
    whole, and changes the score on 845 of the 981 pinned pages under a block-splitting
    degradation. The separator therefore has to be a real space.
    """
    truth = PageProjection(("alpha", "beta"), ("text_block",) * 2, (), ())
    fused = PageProjection(("alphabeta",), ("text_block",), (), ())

    assert score_page(truth, truth)["text_content_similarity"] == 1.0
    assert score_page(truth, fused)["text_content_similarity"] < 1.0


def test_normalization_strips_surrounding_whitespace() -> None:
    """Layout padding around a block must not cost score.

    208 projected ground-truth blocks carry leading or trailing whitespace, and 351 pinned strings
    normalize differently with and without the strip. Without it ``' x'`` scores 0.5 against
    ``'x'``, so pure padding would look like a content defect.
    """
    assert oracles.normalize_text("  padded text \n") == "padded text"
    assert content_similarity("  padded text \n", "padded text") == 1.0


def test_inline_code_spans_contribute_their_text() -> None:
    """Monospace runs must reach the compared text stream.

    ``src/all2md/parsers/pdf.py`` emits an inline ``Code`` node for a monospace span, so this is a
    live PDF-lane path that had no coverage. Measured, not assumed: narrowing ``_node_text``'s
    isinstance check to ``Text`` alone does NOT change the result, because ``Code`` has no children
    and the ``getattr(node, "content")`` fallback returns the same string. This test therefore pins
    the observable contract (the text arrives) rather than the isinstance fast path, and it fails if
    a future change routes inline nodes without a plain ``content`` attribute.
    """
    document = Document(
        children=[Paragraph(content=[Text(content="run"), Code(content="ls -la")])],
    )

    assert project_ast(document).text_blocks == ("run ls -la",)


def test_formula_comparison_folds_compatibility_equivalent_spellings() -> None:
    """A fullwidth and an ASCII spelling of the same character must be one formula.

    The pinned dataset writes the trailing comma of 20 spans as fullwidth U+FF0C in the span's
    ``latex`` field and as ASCII in the detection's ``text`` field. The text metric NFKC-folds and
    forgives this; without the same folding here, perfect text fidelity and perfect formula
    fidelity were mutually unsatisfiable on 9 pinned pages, capping formula_content_similarity at
    0.99924404 forever. NFKC neither casefolds nor collapses whitespace, so the two properties
    below still hold.
    """
    truth = PageProjection((), (), (), (FormulaProjection("inline", "$ R^{2}\uff0c $"),))
    ascii_spelling = PageProjection((), (), (), (FormulaProjection("inline", "$ R^{2}, $"),))
    assert score_page(truth, ascii_spelling)["formula_content_similarity"] == 1.0

    # NFKC must not erase the distinctions the metric relies on.
    assert oracles.normalize_formula("$\\Gamma$") != oracles.normalize_formula("$\\gamma$")
    assert oracles.normalize_formula("$\\text{a b}$") != oracles.normalize_formula("$\\text{ab}$")
