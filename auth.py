import sqlite3
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends, Header, Request
from pydantic import BaseModel
from passlib.context import CryptContext
from jose import JWTError, jwt

SECRET_KEY = "change_this_secret_in_prod"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

DB_PATH = "users.db"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
router = APIRouter()


class AuthData(BaseModel):
    username: str
    password: str


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_tables():
    conn = _get_db()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


@router.on_event("startup")
def startup_event():
    _ensure_tables()


@router.post("/register")
def register(auth: AuthData):
    if not auth.username or not auth.password:
        raise HTTPException(status_code=400, detail="username and password required")
    conn = _get_db()
    cur = conn.cursor()
    hashed = get_password_hash(auth.password)
    try:
        cur.execute("INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
                    (auth.username, hashed, datetime.utcnow().isoformat()))
        conn.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="username already exists")
    finally:
        conn.close()
    return {"ok": True, "username": auth.username}


@router.post("/login")
async def login(request: Request):
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        username = body.get("username")
        password = body.get("password")
    else:
        form = await request.form()
        username = form.get("username")
        password = form.get("password")

    if not username or not password:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    conn = _get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if not row or not verify_password(password, row["password"]):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header missing or invalid")
    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


@router.get("/me")
def me(username: str = Depends(get_current_user)):
    return {"username": username}
