"""
End-to-end test: List of articles with authors and timestamps (from AGENTS.md spec).

This test validates array generation with complex item schemas.
"""

import json
import pytest
from pathlib import Path


# Load schema fixture
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
with open(FIXTURES_DIR / "schemas" / "articles.json") as f:
    ARTICLES_SCHEMA = json.load(f)

with open(FIXTURES_DIR / "prompts" / "prompts.json") as f:
    PROMPTS = json.load(f)["articles"]


@pytest.mark.e2e
@pytest.mark.slow
class TestArticlesList:
    """Test articles list generation (spec requirement)."""

    @pytest.fixture(scope="class")
    def generator(self):
        """Create generator fixture."""
        from grammar_guard import GrammarConstrainedGenerator

        # Force CPU to avoid device-specific issues during testing
        return GrammarConstrainedGenerator(
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            backend="transformers",
            device="cpu"
        )

    def test_articles_list_generation(self, generator):
        """Test generating list of articles."""
        result = generator.generate(
            prompt=PROMPTS[0],  # "Generate 3 technology articles..."
            schema=ARTICLES_SCHEMA,
            max_tokens=300,
            max_retries=3
        )

        assert result.output

        # Parse JSON
        try:
            parsed = json.loads(result.output)
        except json.JSONDecodeError as e:
            pytest.fail(f"Generated invalid JSON: {e}\nOutput: {result.output}")

        # Should be an array
        assert isinstance(parsed, list), "Output should be an array"
        assert len(parsed) >= 1, "Array should have at least 1 item"
        assert len(parsed) <= 10, "Array should have at most 10 items"

        # Validate each article
        for i, article in enumerate(parsed):
            assert isinstance(article, dict), f"Article {i} should be an object"

            # Required fields
            assert "title" in article, f"Article {i} missing required field: title"
            assert "author" in article, f"Article {i} missing required field: author"
            assert "timestamp" in article, f"Article {i} missing required field: timestamp"

            # Type validation
            assert isinstance(article["title"], str), f"Article {i} title should be string"
            assert isinstance(article["author"], str), f"Article {i} author should be string"
            assert isinstance(article["timestamp"], str), f"Article {i} timestamp should be string"

            # Length constraints
            assert len(article["title"]) >= 5, f"Article {i} title too short"
            assert len(article["title"]) <= 200, f"Article {i} title too long"
            assert len(article["author"]) >= 2, f"Article {i} author too short"

            # Optional tags
            if "tags" in article:
                assert isinstance(article["tags"], list), f"Article {i} tags should be array"

        print(f"✓ Generated {len(parsed)} valid articles")

    def test_articles_with_tags(self, generator):
        """Test generating articles with tags."""
        result = generator.generate(
            prompt=PROMPTS[2],  # "Generate articles about machine learning with tags"
            schema=ARTICLES_SCHEMA,
            max_tokens=250,
            max_retries=3
        )

        assert result.output

        parsed = json.loads(result.output)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1

        # Check first article
        article = parsed[0]
        assert "title" in article
        assert "author" in article
        assert "timestamp" in article

        print(f"✓ Generated articles with metadata")
