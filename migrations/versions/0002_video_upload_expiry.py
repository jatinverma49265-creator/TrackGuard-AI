"""Track upload URL expiry on video rows."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_video_upload_expiry"
down_revision: Union[str, None] = "0001_backend_foundation"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("upload_expires_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "upload_expires_at")
