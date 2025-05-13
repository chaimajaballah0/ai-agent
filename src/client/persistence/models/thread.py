from sqlalchemy import Column, String, JSON, DateTime, PrimaryKeyConstraint, select
from datetime import datetime
from client.persistence.db.database import Base, AsyncSessionLocal

class UserThread(Base):
    __tablename__ = "user_threads"

    user_id = Column(String, nullable=False)
    thread_id = Column(String, nullable=False)
    store = Column(JSON)
    checkpoint = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        PrimaryKeyConstraint("user_id", "thread_id"),
    )

    @classmethod
    async def save_or_update(cls, user_id: str, thread_id: str, store: dict, checkpoint: dict):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(cls).where(cls.user_id == user_id, cls.thread_id == thread_id)
            )
            existing = result.scalar_one_or_none()
            if existing:
                existing.store = store
                existing.checkpoint = checkpoint
            else:
                session.add(cls(
                    user_id=user_id,
                    thread_id=thread_id,
                    store=store,
                    checkpoint=checkpoint
                ))
            await session.commit()

    @classmethod
    async def load(cls, user_id: str, thread_id: str):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(cls).where(cls.user_id == user_id, cls.thread_id == thread_id)
            )
            return result.scalar_one_or_none()

    @classmethod
    async def list_threads(cls, user_id: str):
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(cls).where(cls.user_id == user_id)
            )
            return result.scalars().all()
