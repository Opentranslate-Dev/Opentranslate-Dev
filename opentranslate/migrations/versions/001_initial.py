"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-03-26 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Create translations table
    op.create_table(
        'translations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('source_text', sa.Text(), nullable=False),
        sa.Column('target_text', sa.Text(), nullable=False),
        sa.Column('source_lang', sa.String(), nullable=False),
        sa.Column('target_lang', sa.String(), nullable=False),
        sa.Column('domain', sa.String(), nullable=False),
        sa.Column('translator_address', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False)
    )

    # Create validations table
    op.create_table(
        'validations',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('translation_id', sa.String(), sa.ForeignKey('translations.id')),
        sa.Column('validator_address', sa.String(), nullable=False),
        sa.Column('score', sa.Integer(), nullable=False),
        sa.Column('feedback', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False)
    )

    # Create rewards table
    op.create_table(
        'rewards',
        sa.Column('id', sa.String(), primary_key=True),
        sa.Column('address', sa.String(), nullable=False),
        sa.Column('amount', sa.Numeric(), nullable=False),
        sa.Column('type', sa.String(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False)
    )

def downgrade():
    op.drop_table('rewards')
    op.drop_table('validations')
    op.drop_table('translations') 