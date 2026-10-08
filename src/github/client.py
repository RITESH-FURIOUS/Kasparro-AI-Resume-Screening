from dataclasses import dataclass
from datetime import datetime, timezone

import httpx


@dataclass
class GitHubRepository:
    name: str
    url: str
    description: str | None
    language: str | None
    stars: int
    updated_at: str | None


@dataclass
class GitHubSummary:
    username: str
    profile_url: str
    public_repositories: int
    recent_activity_score: float
    relevant_repository_score: float
    score: float
    repositories: list[GitHubRepository]
    error: str | None = None


class GitHubClient:
    """Small GitHub API client used only for enrichment."""

    API_BASE = "https://api.github.com"

    def __init__(self, token: str | None = None) -> None:
        self.token = token

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "kasparro-resume-screening",
        }

        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def extract_username(self, github_url: str) -> str | None:
        """Extract a GitHub username from a profile URL."""

        if not github_url:
            return None

        cleaned = github_url.rstrip("/")

        parts = cleaned.split("/")

        if len(parts) < 2:
            return None

        username = parts[-1].strip()

        if not username or username.lower() in {
            "github.com",
            "repositories",
        }:
            return None

        return username

    def get_profile(
        self,
        username: str,
    ) -> dict:
        response = httpx.get(
            f"{self.API_BASE}/users/{username}",
            headers=self._headers(),
            timeout=10.0,
        )

        response.raise_for_status()

        return response.json()

    def get_repositories(
        self,
        username: str,
    ) -> list[dict]:
        response = httpx.get(
            f"{self.API_BASE}/users/{username}/repos",
            headers=self._headers(),
            params={
                "per_page": 100,
                "sort": "updated",
            },
            timeout=10.0,
        )

        response.raise_for_status()

        return response.json()

    def summarize(
        self,
        github_url: str | None,
    ) -> GitHubSummary | None:
        """
        Enrich a candidate from their public GitHub profile.

        Any GitHub failure returns a summary with an error instead of
        failing the resume.
        """

        if not github_url:
            return None

        username = self.extract_username(github_url)

        if not username:
            return None

        try:
            profile = self.get_profile(username)
            repositories = self.get_repositories(username)

            repo_objects = [
                GitHubRepository(
                    name=repo.get("name", ""),
                    url=repo.get("html_url", ""),
                    description=repo.get("description"),
                    language=repo.get("language"),
                    stars=repo.get("stargazers_count", 0),
                    updated_at=repo.get("updated_at"),
                )
                for repo in repositories
            ]

            recent_activity_score = self._recent_activity_score(
                repo_objects
            )

            relevant_repository_score = (
                self._relevant_repository_score(repo_objects)
            )

            score = min(
                10.0,
                recent_activity_score
                + relevant_repository_score,
            )

            return GitHubSummary(
                username=username,
                profile_url=profile.get(
                    "html_url",
                    github_url,
                ),
                public_repositories=profile.get(
                    "public_repos",
                    len(repo_objects),
                ),
                recent_activity_score=recent_activity_score,
                relevant_repository_score=relevant_repository_score,
                score=round(score, 2),
                repositories=repo_objects[:10],
            )

        except Exception as exc:
            return GitHubSummary(
                username=username,
                profile_url=github_url,
                public_repositories=0,
                recent_activity_score=0,
                relevant_repository_score=0,
                score=0,
                repositories=[],
                error=str(exc),
            )

    @staticmethod
    def _recent_activity_score(
        repositories: list[GitHubRepository],
    ) -> float:
        """
        Up to 5 points for recent repository activity.

        We intentionally keep this conservative:
        public activity is evidence of activity, not proof of code quality.
        """

        now = datetime.now(timezone.utc)

        recent_count = 0

        for repo in repositories[:20]:
            if not repo.updated_at:
                continue

            try:
                updated = datetime.fromisoformat(
                    repo.updated_at.replace("Z", "+00:00")
                )

                days_old = (now - updated).days

                if days_old <= 180:
                    recent_count += 1

            except ValueError:
                continue

        return min(
            5.0,
            recent_count * 0.5,
        )

    @staticmethod
    def _relevant_repository_score(
        repositories: list[GitHubRepository],
    ) -> float:
        """
        Up to 5 points for repositories relevant to backend/AI engineering.
        """

        keywords = {
            "python",
            "ai",
            "llm",
            "agent",
            "agents",
            "rag",
            "langchain",
            "langgraph",
            "crewai",
            "machine-learning",
            "machine_learning",
            "backend",
            "api",
            "fastapi",
            "flask",
            "tensorflow",
            "pytorch",
        }

        relevant = 0

        for repo in repositories:
            text = " ".join(
                [
                    repo.name or "",
                    repo.description or "",
                    repo.language or "",
                ]
            ).lower()

            if any(keyword in text for keyword in keywords):
                relevant += 1

        return min(
            5.0,
            relevant * 1.0,
        )