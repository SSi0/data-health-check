import pytest

from data_health_check import masking

SENSITIVE = [
    "ADDRESS",
    "BAN",
    "CREDIT_CARD",
    "DRIVERS_LICENSE",
    "EMAIL_ADDRESS",
    "PERSON",
    "PHONE_NUMBER",
    "SSN",
    "IPV4",
    "IPV6",
    "MAC_ADDRESS",
]


def test_sensitive_labels_match_the_specification():
    assert frozenset(SENSITIVE) == masking.SENSITIVE_LABELS


@pytest.mark.parametrize("label", SENSITIVE)
def test_every_sensitive_label_is_masked(label):
    assert masking.mask_samples(["a", "b"], label) == [f"[masked: {label}]"]


@pytest.mark.parametrize(
    "label", ["INTEGER", "FLOAT", "DATE", "UNKNOWN", "URL", "could not determine"]
)
def test_other_labels_are_not_masked(label):
    assert masking.mask_samples(["a", "b"], label) == ["a", "b"]


@pytest.mark.parametrize("label", [None, ""])
def test_missing_label_is_not_masked(label):
    assert not masking.is_sensitive(label)
    assert masking.mask_samples(["a"], label) == ["a"]


def test_combined_label_is_masked_if_any_part_is_sensitive():
    assert masking.is_sensitive("UNKNOWN|ADDRESS")
    assert masking.mask_placeholder("UNKNOWN|ADDRESS") == "[masked: ADDRESS]"
    assert masking.mask_placeholder("SSN|PHONE_NUMBER") == "[masked: SSN|PHONE_NUMBER]"
    assert not masking.is_sensitive("INTEGER|FLOAT")


def test_label_matching_ignores_case_and_spaces():
    assert masking.is_sensitive(" ssn ")


def test_mask_samples_returns_a_copy():
    samples = ["x"]
    result = masking.mask_samples(samples, "INTEGER")
    result.append("y")
    assert samples == ["x"]


@pytest.mark.parametrize("label", [None, "INTEGER", "SSN"])
def test_lite_mode_hides_samples_whatever_the_label(label):
    hidden = masking.mask_samples(["a", "b"], label, labels_checked=False)
    assert hidden == [masking.LITE_MODE_PLACEHOLDER] == ["[hidden: Lite mode]"]


@pytest.mark.parametrize("checked", [True, False])
def test_empty_samples_stay_empty(checked):
    assert masking.mask_samples([], "SSN", labels_checked=checked) == []


def numeric_column(**overrides):
    column = {
        "column_name": "id",
        "data_type": "int",
        "data_label": "INTEGER",
        "min": 900112222,
        "max": 999887777,
        "median": 950000000.5,
        "mean": 949000000.0,
        "stddev": 1000.0,
        "unique_count": 150,
        "sample_values": ["901112222", "902223333"],
    }
    column.update(overrides)
    return column


def test_sensitive_number_columns_mask_values_copied_from_the_data():
    protected = masking.protect_column(
        numeric_column(data_label="SSN"), labels_checked=True
    )
    # Every number statistic: with one or two distinct values even the mean
    # and stddev give the values away, and they mean nothing for IDs.
    for key in ("min", "max", "median", "mean", "stddev"):
        assert protected[key] == "[masked: SSN]"
    assert protected["sample_values"] == ["[masked: SSN]"]


def test_safe_columns_are_unchanged():
    column = numeric_column()
    assert masking.protect_column(column, labels_checked=True) == column


def test_lite_mode_hides_number_statistics_but_not_dates():
    lite = masking.protect_column(numeric_column(), labels_checked=False)
    assert lite["min"] == lite["max"] == lite["median"] == "[hidden: Lite mode]"
    assert lite["mean"] == 949000000.0
    dates = masking.protect_column(
        {
            "data_type": "datetime",
            "min": "2025-01-01",
            "max": "2025-12-31",
            "sample_values": ["2025-02-03"],
        },
        labels_checked=False,
    )
    assert (dates["min"], dates["max"]) == ("2025-01-01", "2025-12-31")
    assert dates["sample_values"] == ["[hidden: Lite mode]"]


def test_protect_column_does_not_modify_its_input():
    column = numeric_column(data_label="SSN")
    masking.protect_column(column, labels_checked=True)
    assert column["min"] == 900112222


@pytest.mark.parametrize(
    ("unique_count", "hidden"), [(1, True), (2, True), (None, True), (3, False)]
)
def test_lite_mode_hides_averages_of_columns_with_few_distinct_values(
    unique_count, hidden
):
    # A constant column's mean is its value; two values follow from mean ± sd.
    lite = masking.protect_column(
        numeric_column(unique_count=unique_count), labels_checked=False
    )
    expected = "[hidden: Lite mode]" if hidden else (949000000.0, 1000.0)
    assert (lite["mean"], lite["stddev"]) == (
        (expected, expected) if hidden else expected
    )
