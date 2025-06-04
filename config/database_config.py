import os
from dotenv import load_dotenv

load_dotenv()

# Oracle Database Configuration (using modern oracledb driver)
ORACLE_CONFIG = {
    'user': os.getenv('ORACLE_USER'),
    'password': os.getenv('ORACLE_PASSWORD'),
    'dsn': os.getenv('ORACLE_DSN'),
    'encoding': os.getenv('ORACLE_ENCODING', 'UTF-8')
}

# Connection string for SQLAlchemy (using oracledb driver)
ORACLE_CONNECTION_STRING = f"oracle+oracledb://{ORACLE_CONFIG['user']}:{ORACLE_CONFIG['password']}@{ORACLE_CONFIG['dsn']}"

# PostgreSQL fallback configuration
POSTGRES_CONFIG = {
    'host': 'localhost',
    'database': 'bank_reviews',
    'user': 'postgres',
    'password': 'postgres',
    'port': 5432
}

POSTGRES_CONNECTION_STRING = f"postgresql://{POSTGRES_CONFIG['user']}:{POSTGRES_CONFIG['password']}@{POSTGRES_CONFIG['host']}:{POSTGRES_CONFIG['port']}/{POSTGRES_CONFIG['database']}"

# Database selection - can be overridden by environment variable
USE_ORACLE = os.getenv('USE_ORACLE', 'true').lower() == 'true' 