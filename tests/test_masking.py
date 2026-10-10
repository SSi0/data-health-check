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
