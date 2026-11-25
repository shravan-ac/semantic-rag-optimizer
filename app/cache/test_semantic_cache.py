
import pytest

from app.cache import semantic_cache


TEST_COLLECTION = "semantic_cache_test"

TEST_SOURCES = ["leave_policy.txt"]

TEST_ANSWER = (
    "The reporting manager must approve leave requests."
)

TEST_CASES = [
    {
        "name": "Exact match",
        "cached_question": "Who approves leave requests?",
        "incoming_question": "Who approves leave requests?",
        "expected_hit": True,
    },
    {
        "name": "Genuine paraphrase",
        "cached_question": "Is annual leave paid?",
        "incoming_question": "Do employees receive paid annual leave?",
        "expected_hit": True,
    },
    {
        "name": "Different topic",
        "cached_question": "Is annual leave paid?",
        "incoming_question": "What is the internet reimbursement policy?",
        "expected_hit": False,
    },
    {
        "name": "Different numbers",
        "cached_question": "Is a minimum of 3 days required?",
        "incoming_question": "Is a minimum of 5 days required?",
        "expected_hit": False,
    },
    {
        "name": "Negation",
        "cached_question": "Are employees allowed to submit requests?",
        "incoming_question": "Are employees not allowed to submit requests?",
        "expected_hit": False,
    },
    {
        "name": "Modality",
        "cached_question": "Can employees request leave?",
        "incoming_question": "Must employees request leave?",
        "expected_hit": False,
    },
]


def reset_test_collection():
    collections = semantic_cache.client.get_collections().collections

    existing = [collection.name for collection in collections]

    if TEST_COLLECTION in existing:
        semantic_cache.client.delete_collection(
            collection_name=TEST_COLLECTION
        )

    semantic_cache.create_cache_collection(
        collection_name=TEST_COLLECTION
    )


@pytest.fixture
def isolated_cache_collection():
    """Create and clean up only the isolated test collection."""
    reset_test_collection()

    yield

    collections = semantic_cache.client.get_collections().collections
    existing = [collection.name for collection in collections]

    if TEST_COLLECTION in existing:
        semantic_cache.client.delete_collection(
            collection_name=TEST_COLLECTION
        )


@pytest.mark.parametrize(
    "test_case",
    TEST_CASES,
    ids=[case["name"] for case in TEST_CASES],
)
def test_semantic_cache_behavior(test_case, isolated_cache_collection):
    semantic_cache.save_to_cache(
        query=test_case["cached_question"],
        answer=TEST_ANSWER,
        sources=TEST_SOURCES,
        collection_name=TEST_COLLECTION,
    )

    result = semantic_cache.search_cache(
        test_case["incoming_question"],
        collection_name=TEST_COLLECTION,
    )

    actual_hit = result is not None

    assert actual_hit == test_case["expected_hit"], (
        f"{test_case['name']}: expected "
        f"{'HIT' if test_case['expected_hit'] else 'MISS'}, "
        f"got {'HIT' if actual_hit else 'MISS'}"
    )

    if result:
        assert result["answer"] == TEST_ANSWER
        assert result["sources"] == TEST_SOURCES
