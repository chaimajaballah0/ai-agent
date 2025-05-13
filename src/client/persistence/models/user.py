from sqlalchemy import Column, String, DateTime, select
from datetime import datetime
from client.persistence.db.database import Base, AsyncSessionLocal

class User(Base):
    __tablename__ = "users"

    user_id = Column(String, primary_key=True)  # Google 'sub'
    email = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    @classmethod
    async def save_if_not_exists(cls, user_id: str, email: str):
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(cls).where(cls.user_id == user_id))
            user = result.scalar_one_or_none()
            if not user:
                session.add(cls(user_id=user_id, email=email))
                await session.commit()

    @classmethod
    async def get_by_email(cls, email: str):
        async with AsyncSessionLocal() as session:
            result = await session.execute(select(cls).where(cls.email == email))
            return result.scalar_one_or_none()
