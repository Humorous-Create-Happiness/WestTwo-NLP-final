from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin
import time

headers = {
    #'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'
    'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1'

}


def getASection(url, bookName):
    bookName += ".txt"
    f = open(bookName, "a", encoding='utf-8')
    rsp = requests.get(url, headers=headers)
    rsp.encoding = 'utf-8'
    bs = BeautifulSoup(rsp.text, 'html.parser')

    title = bs.select('h1')
    if not title:
        print("无法找到章节标题：", url)
        return

    f.write(title[0].text)
    f.write("\n")

    body = bs.select('div.content')
    if not body:
        print("无法找到章节内容：", url)
        return

    paragraphs = body[0].find_all('p')
    content = []
    for p in paragraphs:
        content.append(p.text)
        content.append("\n")

    f.writelines(content)


    f.close()


def getSections(url, bookName):
    rsp = requests.get(url, headers=headers)
    rsp.encoding = 'utf-8'
    bs = BeautifulSoup(rsp.text, 'html.parser')
    sections = bs.select('ul.list')[0]
    links = sections.select('a')
    #print(links)
    #del links[0]  # 第一个a标签为“收起”，将其删除
    for link in links:
        if link.attrs["href"] is not None:
            newUrl = urljoin(url, link.attrs['href'])
            getASection(newUrl, bookName)


def getBooks(url):
    bookUrls = dict()  # 用字典来存放小说信息，键为书名，值为链接地址
    rsp = requests.get(url, headers=headers)
    rsp.encoding = 'utf-8'
    bs = BeautifulSoup(rsp.text, 'html.parser')
    for j in range(1):
        time.sleep(10)
        bookList = bs.select('div.c3')[j+5]
        sort = bookList.select('h2.title')[0]
        sort = sort.find('span').text
        for i in range(5):
            bookList = bs.select('div.c3')[j+5]
            book = bookList.select('a')[i]
            if book.attrs['href'] is not None:
                href = 'https://quanben5.com/' + book.attrs['href'] + 'xiaoshuo.html'
                href = href.replace('book', 'list')  # 需要把url中的book替换为list，直接进入章节页面
                bookName = book.text + '(' +sort + ')'
                if bookName not in bookUrls:
                    bookUrls[bookName] = href
                    print("{}:{}".format(bookName,href))
        for bookName in bookUrls.keys():
            getSections(bookUrls[bookName], bookName)
            print('{}已经完成'.format(bookName))


getBooks('https://quanben5.com/')