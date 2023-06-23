import hashlib
import os
import requests
from bs4 import BeautifulSoup

# 搜索引擎 URL 模板
SEARCH_URL_TEMPLATE = "https://www.bing.com/images/search?q={query}&first={first}&count={count}"


# 计算图像哈希值
def calculate_image_hash(image_data):
    md5 = hashlib.md5()
    md5.update(image_data)
    return md5.hexdigest()


# 下载图像并返回图像数据
def download_image(url):
    try:
        response = requests.get(url, stream=True)
        image_data = response.content
        return image_data
    except Exception as e:
        print(f"Error downloading image: {e}")
        return None


def download_celebrity_images(file_prefix, celebrity_name, save_dir, num_images ):
    os.makedirs(save_dir, exist_ok=True)

    image_counter = 0
    image_hashes = set()
    for i in range(0, num_images, 50):
        search_url = SEARCH_URL_TEMPLATE.format(query=f"{celebrity_name}+photos", first=i, count=100)
        response = requests.get(search_url)
        soup = BeautifulSoup(response.text, "html.parser")
        image_tags = soup.find_all("img", class_="mimg")

        for img_tag in image_tags:
            img_url = None

            if "data-src" in img_tag.attrs:
                img_url = img_tag["data-src"]
            elif "src" in img_tag.attrs:
                img_url = img_tag["src"]

            if img_url:
                image_data = download_image(img_url)
                if image_data:
                    image_hash = calculate_image_hash(image_data)
                    if image_hash not in image_hashes:
                        image_hashes.add(image_hash)
                        img_filename = os.path.join(save_dir, f"{file_prefix}_{image_counter}.jpg")
                        with open(img_filename, "wb") as f:
                            f.write(image_data)
                        image_counter += 1
                        if image_counter >= num_images:
                            break

        if image_counter >= num_images:
            break


if __name__ == '__main__':
    download_celebrity_images('cxk', "蔡徐坤", "./cxk", 80)
    # download_celebrity_images('dingzhen', "丁真", "./dingzhen", 80)
    # download_celebrity_images('trump', "特朗普", "./trump", 80)
