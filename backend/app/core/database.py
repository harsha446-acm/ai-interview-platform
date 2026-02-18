from motor.motor_asyncio import AsyncIOMotorClient
from app.core.config import settings

client: AsyncIOMotorClient = None
db = None


async def connect_to_mongo():
    global client, db
    mongo_url = settings.MONGODB_URL or "mongodb://localhost:27017"
    extra = {}
    if "mongodb+srv" in mongo_url:
        extra["tls"] = True
    client = AsyncIOMotorClient(
        mongo_url,
        serverSelectionTimeoutMS=60000,
        connectTimeoutMS=30000,
        socketTimeoutMS=30000,
        retryWrites=True,
        **extra,
    )
    db = client[settings.DATABASE_NAME]

    # Create indexes
    await db.users.create_index("email", unique=True)
    await db.candidates.create_index("unique_token", unique=True)
    await db.interview_sessions.create_index("session_token", unique=True)
    print("✅ Connected to MongoDB")


async def close_mongo_connection():
    global client
    if client:
        client.close()
        print("🔌 MongoDB connection closed")


def get_database():
    return db
