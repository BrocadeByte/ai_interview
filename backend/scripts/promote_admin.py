import asyncio
import sys
from pathlib import Path

from sqlalchemy import select


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import app.models  # noqa: F401
from app.core.database import SessionLocal
from app.models.user import User


async def promote_admin(email: str) -> None:
    async with SessionLocal() as db:
        user = await db.scalar(select(User).where(User.email == email))
        if not user:
            raise SystemExit(f"user not found: {email}")
        user.is_admin = True
        await db.commit()
        print({"email": user.email, "is_admin": user.is_admin})


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/promote_admin.py <email>")
    asyncio.run(promote_admin(sys.argv[1]))
