from typing import TypedDict


class User(TypedDict):
    name: str
    age: int
    email: str


def create_user(name: str, age: int, email: str) -> User:
    return {"name": name, "age": age, "email": email}


if __name__ == "__main__":
    # 올바른 사용
    user1 = create_user("Alice", 30, "alice@example.com")

    # 타입 오류 (age에 문자열 할당)
    user2 = create_user("Bob", "25", "bob@example.com")

    # 타입 오류 (추가 필드 할당)
    user3: User = {
        "name": "Charlie",
        "age": 35,
        "email": "charlie@example.com",
        "extra": "info",
    }

    # 타입 오류 (필수 필드 누락)
    user4: User = {"name": "David", "age": 40}
