import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver import ChromeOptions

from typing import List
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query

app = FastAPI(
    title="교보문고 의 베스트셀러 책의 목록을 검색하고, 책의 목차를 가져오는 API",
    description="API를 활용하여 교보문고의 베스트셀러 책의 목록을 검색하고, 책의 목차를 가져오는 API",
    version="0.0.1",
    servers=[
        {
            "url": "https://7736-118-216-157-176.ngrok-free.app",
            "description": "Best seller books and table of contents fetch API",
        }
    ],
)


class BookKeyword(BaseModel):
    keyword: str = Field(..., title="Keyword to search")


class BookURL(BaseModel):
    title: str = Field(title="Title of the book")
    url: str = Field(..., title="URL of the book")


@app.get("/search", response_model=List[BookURL])
def get_bestseller_list_by_keyword(
    keyword: str = Query(..., title="keyword to search")
) -> list:
    """Search best seller book lists by keyword and then return the book lists"""
    url = f"https://search.kyobobook.co.kr/search?keyword={keyword}&gbCode=TOT&target=total"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        book_items = soup.find_all("a", attrs={"class": "prod_info"})
        book_list = [
            # {"title": prod.text.strip(), "link": prod.get("href")}
            BookURL(title=prod.text.strip(), url=prod.get("href"))
            for prod in book_items[:5]
        ]
        return book_list
    else:
        return []


@app.get("/info")
def get_table_of_content_by_url(
    bookurl: str = Query(..., title="url of the book"), response_model=str
):
    """Get book's table of contents by url"""
    options = ChromeOptions()
    options.add_argument("--headless=new")
    driver = webdriver.Chrome(options=options)
    driver.get(bookurl)
    contents = driver.find_element(By.CLASS_NAME, "book_contents_item").text
    driver.quit()
    return contents
