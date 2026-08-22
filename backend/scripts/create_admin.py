import asyncio
import sys
from pathlib import Path

from sqlalchemy import select


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401
from app.core.database import Base, SessionLocal, engine
from app.core.security import hash_password
from app.models.profile import UserProfile
from app.models.user import User


DEFAULT_ADMIN_EMAIL = "admin@example.com"
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "123456"


async def create_or_reset_admin(
    email: str = DEFAULT_ADMIN_EMAIL,
    password: str = DEFAULT_ADMIN_PASSWORD,
    username: str = DEFAULT_ADMIN_USERNAME,
) -> None:
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        async with SessionLocal() as db:
            user = await db.scalar(select(User).where(User.email == email))
            if user:
                user.username = username or user.username
                user.password_hash = hash_password(password)
                user.is_admin = True
                action = "reset"
            else:
                user = User(
                    email=email,
                    username=username,
                    password_hash=hash_password(password),
                    is_admin=True,
                )
                db.add(user)
                await db.flush()
                db.add(UserProfile(user_id=user.id))
                action = "created"
            await db.commit()
            print({"action": action, "email": email, "username": username, "is_admin": True})
    finally:
        await engine.dispose()


if __name__ == "__main__":
    email = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_ADMIN_EMAIL
    password = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_ADMIN_PASSWORD
    username = sys.argv[3] if len(sys.argv) >= 4 else DEFAULT_ADMIN_USERNAME
    asyncio.run(create_or_reset_admin(email=email, password=password, username=username))
