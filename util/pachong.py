import os
import requests
from bs4 import BeautifulSoup


# 下载图像
def download_image(url, save_path):
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def download_image_(save_file_name, search_name, save_dir, nums=10):
    # 替换为你感兴趣的名人姓名
    celebrity_name = search_name
    # 搜索引擎 URL 模板
    search_url_template = "https://www.bing.com/images/search?q={query}&first={first}&count={count}"
    # 图像保存目录
    image_dir = save_dir
    os.makedirs(image_dir, exist_ok=True)
    # 爬取的图像数量
    num_images = nums

    # 搜索并下载图像
    for i in range(0, num_images, 1):
        search_url = search_url_template.format(query=celebrity_name, first=i, count=10)
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        image_tags = soup.find_all("img", class_="mimg")

        for img_tag in image_tags:
            if "src" in img_tag.attrs:
                img_url = img_tag["src"]
                img_filename = os.path.join(image_dir, f"{save_file_name}_{i}.jpg")
                download_image(img_url, img_filename)
                i += 1
                if i >= num_images:
                    break


if __name__ == '__main__':
    # download_image_('cxk', "蔡徐坤", "../data/face/database/cxk", 25)
    download_image_('obama', "obama", "../data/face/database/BarackObama", 25)
