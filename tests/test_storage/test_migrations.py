from __future__ import annotations

from agentkb.storage.migrations import MIGRATIONS


def test_migrations_are_strictly_ordered_and_have_unique_checksums():
    versions = [migration.version for migration in MIGRATIONS]
    checksums = [migration.checksum for migration in MIGRATIONS]

    assert versions == sorted(versions)
    assert len(versions) == len(set(versions))
    assert len(checksums) == len(set(checksums))
    assert all(len(checksum) == 64 for checksum in checksums)
