from DrissionPage import ChromiumPage, errors
import time
from datetime import datetime

class IG_Parser:
    def __init__(self, account):
        self.account = account
        self.saved_contents = []
        self.post_idx = 0
        # 新增: 存储上一个帖子的内容用于比较
        self.prev_post_data = None  # 新增: 存储上一个帖子的数据

    def start_parse(self, download_num):
        url = 'https://www.instagram.com/' + self.account + '/'
        page = ChromiumPage()
        page.get(url)

        # 点击第一张照片
        first_post = page.ele('._aagw')
        first_post.click()

        pop_window_ele = page.ele('.x1cy8zhl x9f619 x78zum5 xl56j7k x2lwn1j xeuugli x47corl')
        for i in range(download_num):
            print('=== 第', i + 1, '个 Post ===')
            retry_count = 0
            post_text = ''
            post_datetime = ''
            while retry_count < 2:
                try:
                    post_text_ele = pop_window_ele.ele('._ap3a _aaco _aacu _aacx _aad7 _aade')
                    post_text = post_text_ele.text
                    print('post_text:', post_text)
                    time_element = pop_window_ele.ele('._a9ze _a9zf')
                    datetime_str = time_element.text
                    print('datetime:', datetime_str)
                    if time_element:
                        datetime_str = time_element.attr('datetime')
                        if datetime_str:
                            # 转换时间格式
                            dt = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                            post_datetime = dt.strftime("%Y-%m-%d %H:%M:%S")
                    
                    print(f'post_datetime: {post_datetime}')
                    break
                except errors.ElementNotFoundError:
                    print('该 post 为视频')
                    self.post_idx += 1
                    break
                except errors.ElementLostError:
                    retry_count += 1
                    print('retry:', retry_count)
                    time.sleep(3)

            # 新增: 检查是否与上一个帖子相同
            current_post_data = (post_text, post_datetime)  # 新增: 创建当前帖子数据元组
            if self.prev_post_data and self.prev_post_data == current_post_data:
                print('当前帖子与上一个帖子内容相同，停止解析')
                return self.get_saved_contents()  # 新增: 提前返回收集的数据
            
            if post_text:  # 确保 post_text 不为空
                self.download_text(post_text, post_datetime)
                # 新增: 只有在成功保存文本后才更新prev_post_data
                self.prev_post_data = current_post_data  # 新增: 更新上一个帖子数据

            self.post_idx += 1
            
            # 新增: 检查是否达到最大数量
            if i < download_num - 1:
                try:
                    next_btn = pop_window_ele.ele('. _aaqg _aaqh')
                    next_btn.click()
                    time.sleep(1)  # 新增: 等待加载
                except errors.ElementNotFoundError:
                    print('无法找到下一个按钮，停止解析')
                    break
            else:
                print('已达到指定的帖子数量')
            
        # 添加返回收集的数据
        return self.get_saved_contents()

    def download_text(self, text, datetime_str):
        self.current_post = {
            'index': self.post_idx,
            'text': text,
            'datetime': datetime_str
        }
        self.saved_contents.append(self.current_post.copy())
        
    def get_saved_contents(self):
        """返回收集的所有帖子内容"""
        return self.saved_contents


if __name__ == '__main__':
    account = 'axtixt'
    download_num = 10

    parser = IG_Parser(account)
    saved_contents = parser.start_parse(download_num)  # 直接接收返回的数据

    print(saved_contents)