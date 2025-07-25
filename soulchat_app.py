
''' 运行方式
```bash
pip install streamlit # 第一次运行需要安装streamlit
pip install streamlit_chat # 第一次运行需要安装streamlit_chat
streamlit run soulchat_app.py --server.port 9026
```
## 测试访问
http://<your_ip>:9026

'''

import os
import re
import json
import torch
import streamlit as st
from streamlit_chat import message
import pyttsx3  # 使用跨平台的系统级TTS

st.set_page_config(
    page_title="心靈小幫手",
    page_icon="👩‍🏫",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """     
-   版本：心靈小幫手
-   作者：吳世杰
        """
    }
)

from transformers import AutoModel, AutoTokenizer
from igbug import IG_Parser
from xbug import TwitterExtractor
from datetime import datetime
from config import TWITTER_AUTH_TOKEN
from hanlp_restful import HanLPClient
from opencc import OpenCC
import threading
import tempfile
import pygame
from gtts import gTTS
import requests
import pandas as pd
import plotly.express as px

# st-chat uses https://www.dicebear.com/styles for the avatar
# https://emoji6.com/emojiall/
model_name_or_path = 'scutcyr/SoulChat'
# 指定显卡进行推理
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 默认使用0号显卡，避免Windows用户忘记修改该处
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

HanLP = HanLPClient('https://www.hanlp.com/api', auth='ODQ4OEBiYnMuaGFubHAuY29tOmt6WnlQaDRTZ0N6RUt5OGI=', language='zh')

sensitive_keywords = ['殺', '死', '自殺', '不想活了', '想死', '活不下去', '結束生命', '離開人世']
friendship_keywords = ['想識朋友', '認識朋友', '交朋友', '結識朋友', '交友', '想識新朋友', '認識新交友' , '想識新朋友'] 

@st.cache_resource
def load_model():
    model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True).half()
    model.to(device)
    print('Model Load done!')
    return model

@st.cache_resource
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    print('Tokenizer Load done!')
    return tokenizer

model = load_model()
tokenizer = load_tokenizer()

class InstagramDataFetcher:
    def __init__(self, account: str, download_num: int):
        self.account = account
        self.download_num = download_num
        self.saved_contents = []
        self.post_idx = 0

    def fetch_data(self):
        try:
            parser = IG_Parser(self.account)
            # 直接获取解析后的数据
            self.saved_contents = parser.start_parse(self.download_num)
            
            # 添加统一数据结构
            unified_data = []
            for post in self.saved_contents:
                # 处理日期格式 - 现在日期字段名为 'datetime'
                post_date = post.get('datetime', '')
                
                # 确保内容不为空 - 现在内容字段名为 'text'
                content = post.get('text', '')
                
                unified_data.append({
                    'platform': 'Instagram',
                    'id': post.get('id', ''),
                    'content': content,
                    'url': post.get('url', ''),
                    'date': post_date,
                    'likes': post.get('likes', 0),
                    'comments': post.get('comments', 0)
                })
            return unified_data
        except Exception as e:
            raise Exception(f"Instagram數據獲取失敗: {str(e)}")

def generate_report(social_data, sentiment_result):
    """根據情感標籤和日期產生報告"""
    try:
        # 添加情感标签到每条数据
        for i, item in enumerate(social_data):
            if i < len(sentiment_result):
                # 处理情感分析结果可能是数值的情况
                sentiment_value = sentiment_result[i]
                if isinstance(sentiment_value, float):
                    if sentiment_value > 0.6:
                        item['sentiment'] = '正面'
                    elif sentiment_value < 0.4:
                        item['sentiment'] = '負面'
                    else:
                        item['sentiment'] = '中性'
                else:
                    item['sentiment'] = sentiment_value
            else:
                item['sentiment'] = '未知'
        
        # 创建情感分布饼图
        st.subheader("情感分布分析")
        # 提取情感标签
        sentiment_labels = [item['sentiment'] for item in social_data]
        sentiment_counts = pd.Series(sentiment_labels).value_counts()
            
        fig1 = px.pie(
            sentiment_counts, 
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            title="情感分布比例",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig1.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)
        
        # 创建数据框
        df = pd.DataFrame(social_data)
        
        # 确保有日期字段
        if 'date' in df.columns and len(df) > 0:
            # 转换日期格式
            try:
                # 统一日期格式处理
                df['formatted_date'] = pd.to_datetime(df['date'], errors='coerce')
                
                # 移除无效日期
                df = df.dropna(subset=['formatted_date'])
                
                if not df.empty:
                    # 按日期分组统计
                    daily_sentiment = df.groupby([df['formatted_date'].dt.date, 'sentiment']).size().unstack(fill_value=0)
                    
                    # 确保所有情感类别都存在
                    all_sentiments = ['正面', '中性', '負面']
                    for sentiment in all_sentiments:
                        if sentiment not in daily_sentiment.columns:
                            daily_sentiment[sentiment] = 0
                    
                    # 创建每日情感趋势图
                    st.subheader("每日情感趨勢")
                    fig2 = px.line(
                        daily_sentiment, 
                        x=daily_sentiment.index,
                        y=daily_sentiment.columns,
                        title="每日情感變化趨勢",
                        labels={'value': '貼文數量', 'date': '日期'},
                        markers=True
                    )
                    fig2.update_layout(
                        legend_title_text='情感類型',
                        xaxis_title='日期',
                        yaxis_title='貼文數量'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("沒有有效的日期數據")
            except Exception as e:
                st.error(f"日期處理錯誤: {str(e)}")
        
        # 显示情感最强烈的帖子
        st.subheader("代表性貼文")
        tab1, tab2, tab3 = st.tabs(["正面貼文", "中性貼文", "負面貼文"])
        
        # 分别显示各类别代表性帖子
        with tab1:
            st.markdown("### 😊 最具代表性的正面貼文")
            positive_posts = [post for post in social_data if post.get('sentiment') == '正面']
            if positive_posts:
                # 按日期排序（最新的在前）
                positive_posts.sort(key=lambda x: x.get('date', ''), reverse=True)
                for i, post in enumerate(positive_posts[:5]):  # 显示前5条
                    st.markdown(f"**貼文 {i+1}**")
                    st.write(f"📅 日期: {post.get('date', '未知')}")
                    st.write(f"📝 内容: {post.get('content', '無内容')[:200]}...")
                    if 'url' in post and post['url']:  # 确保URL存在且不为空
                        st.markdown(f"[查看原文]({post['url']})")
                    st.divider()
            else:
                st.info("没有找到正面貼文")
        
        with tab2:
            st.markdown("### 😐 最具代表性的中性貼文")
            neutral_posts = [post for post in social_data if post.get('sentiment') == '中性']
            if neutral_posts:
                # 按日期排序（最新的在前）
                neutral_posts.sort(key=lambda x: x.get('date', ''), reverse=True)
                for i, post in enumerate(neutral_posts[:5]):  # 显示前5条
                    st.markdown(f"**貼文 {i+1}**")
                    st.write(f"📅 日期: {post.get('date', '未知')}")
                    st.write(f"📝 内容: {post.get('content', '無内容')[:200]}...")
                    if 'url' in post and post['url']:
                        st.markdown(f"[查看原文]({post['url']})")
                    st.divider()
            else:
                st.info("没有找到中性貼文")
        
        with tab3:
            st.markdown("### 😔 最具代表性的負面貼文")
            negative_posts = [post for post in social_data if post.get('sentiment') == '負面']
            if negative_posts:
                # 按日期排序（最新的在前）
                negative_posts.sort(key=lambda x: x.get('date', ''), reverse=True)
                for i, post in enumerate(negative_posts[:5]):  # 显示前5条
                    st.markdown(f"**貼文 {i+1}**")
                    st.write(f"📅 日期: {post.get('date', '未知')}")
                    st.write(f"📝 内容: {post.get('content', '無内容')[:200]}...")
                    if 'url' in post and post['url']:
                        st.markdown(f"[查看原文]({post['url']})")
                    
                    # 负面内容提供帮助资源
                    content = str(post.get('content', ''))
                    if any(keyword in content for keyword in sensitive_keywords):
                        st.warning("""
                        🚨 重要提醒：
                        香港撒瑪利亞防止自殺會：+852 2389 2222
                        生命熱線：+852 2382 0000
                        我們非常關心您的安全！
                        """)
                    
                    st.divider()
            else:
                st.info("没有找到負面貼文")
                
    except Exception as e:
        st.error(f"產生報告失敗: {str(e)}")
        

# 新增心灵建议生成函数
def generate_advice(report_summary):
    """根据报告摘要生成建设性建议"""
    # 创建更明确的提示词
    prompt = ("你是一位专业的心理健康顾问，请根据以下情感分析报告，为用户提供一心理健康建议：{report_summary}。请考虑以下因素：1. 正面/中性/负面情绪的比例分布 2. 时间趋势中的关键发现（如高峰日、近期变化趋势） 3. 特殊需求内容的数量和类型。使用温暖、支持性的语气，用繁体中文回复。建议应包含：- 情绪管理技巧 - 日常自我照顾方法 - 社交支持建议 - 专业求助指引（如需要）"
    )
    
    try:
        # 调用模型生成建议
        response, _ = model.chat(
            tokenizer, 
            query=prompt, 
            history=None, 
            max_length=2048,
            temperature=0.7,
            top_p=0.9
        )
        
        # 确保返回完整的建议内容
        advice = convert_to_traditional_chinese(response)
        return advice
    except Exception as e:
        error_msg = f"建議生成失敗: {str(e)}"
        return f"## ❌ 錯誤\n\n{error_msg}"

# 生成并显示心灵建议
    st.subheader("💖 專業心理健康建議")
    with st.spinner("正在生成個性化建議..."):
        mental_health_advice = generate_advice(report_summary)
        st.markdown(mental_health_advice)
    
        # 添加紧急求助信息
        st.warning("""
        **🚨 緊急支援：**
        如果您或您認識的人正在經歷困難時期，請立即尋求幫助：
        - 香港撒瑪利亞防止自殺會：+852 2389 2222
        - 生命熱線：+852 2382 0000
        - 醫院管理局精神健康專線：+852 2466 7350
        """)
        
        # 添加本地资源链接
        st.info("""
        **🏥 本地心理健康資源：**
        - 香港心理衞生會：https://www.mhahk.org.hk/
        - 香港心聆：https://www.jciconcern.hk/
        - 明愛家庭服務：https://family.caritas.org.hk/
        - 香港精神科醫學院：https://www.hkcp.org/
        """)

def generate_local_resources(negative_percentage):
    """根据负面情绪比例推荐本地资源"""
    # 基础资源
    resources = {
        "緊急熱線": [
            "香港撒瑪利亞防止自殺會：+852 2389 2222",
            "生命熱線：+852 2382 0000"
        ],
        "心理諮詢服務": [
            "香港心理衛生會：https://www.mhahk.org.hk/",
            "明愛心理健康服務：https://mentalhealth.caritas.org.hk/"
        ]
    }
    
    # 根据负面情绪比例添加额外资源
    if negative_percentage > 20:
        resources["社區支援中心"] = [
            "東華三院心理健康服務：https://www.tungwah.org.hk/",
            "香港青年協會輔導中心：https://www.hkfyg.org.hk/"
        ]
    
    if negative_percentage > 30:
        resources["專業心理治療"] = [
            "香港心理學會認可心理學家名單：https://www.dcp.hkps.org.hk/",
            "醫院管理局精神健康服務：https://www.ha.org.hk/"
        ]
    
    # 添加自我提升资源
    resources["自我提升資源"] = [
        "香港公共圖書館心理健康書籍專區",
        "Mindfulness HK 正念課程：https://mindfulnesshongkong.com/",
        "OpenUp心理支援平台：https://openup.hk/"
    ]
    
    return resources

def generate_comprehensive_report():
    """產生整合社群媒體數據和使用者對話的綜合情感分析報告"""
    try:
        all_data = []
        
        # 添加社交媒体数据
        if 'social_data' in st.session_state:
            all_data.extend(st.session_state.social_data)
        
        # 添加用户对话数据
        if 'user_inputs' in st.session_state and st.session_state.user_inputs:
            for input_data in st.session_state.user_inputs:
                # 新增: 添加标签字段
                tags = []
                content = str(input_data.get('content', ''))
                
                # 检查敏感词
                if any(keyword in content for keyword in sensitive_keywords):
                    tags.append('敏感内容')
                
                # 检查交友需求
                if any(phrase in content for phrase in friendship_keywords):
                    tags.append('交友需求')
                
                all_data.append({
                    'platform': '使用者對話',
                    'content': content,
                    'date': input_data['date'],
                    'tags': tags  # 新增标签字段
                })
        
        if not all_data:
            st.warning("沒有可分析的數據，請先分析社群媒體或進行對話")
            return
        
        # 打印调试信息
        st.write(f"總數據量: {len(all_data)}條")
        st.json(all_data[:2])  # 显示前2条数据用于调试
        
        # 提取文本内容进行情感分析
        contents = [item.get('content', '') for item in all_data]
        
        # 添加加载状态指示器
        with st.spinner("正在分析情感，請稍候..."):
            try:
                # 添加API调用限制处理
                sentiment_result = HanLP.sentiment_analysis(contents)
                st.session_state.sentiment_result = sentiment_result
            except Exception as e:
                if "rate limit" in str(e).lower():
                    st.error("情緒分析API呼叫過於頻繁，請稍後再試")
                else:
                    st.error(f"情緒分析失敗: {str(e)}")
                return
        
        # 添加情感标签到每条数据 - 使用与Instagram/Twitter相同的阈值
        for i, item in enumerate(all_data):
            if i < len(sentiment_result):
                # 处理情感分析结果可能是数值的情况
                sentiment_value = sentiment_result[i]
                if isinstance(sentiment_value, float):
                    # 使用与Instagram/Twitter相同的阈值标准
                    if sentiment_value > 0.6:
                        item['sentiment'] = '正面'
                    elif sentiment_value < 0.4:
                        item['sentiment'] = '負面'
                    else:
                        item['sentiment'] = '中性'
                else:
                    # 如果返回的是字符串标签，直接使用
                    item['sentiment'] = sentiment_value
            else:
                item['sentiment'] = '未知'
                
        sentiment_labels = [item.get('sentiment', '未知') for item in all_data]
        sentiment_counts = pd.Series(sentiment_labels).value_counts()
        
        # 计算百分比
        total = len(all_data)
        positive_percentage = round(sentiment_counts.get('正面', 0) / total * 100, 1) if total > 0 else 0
        neutral_percentage = round(sentiment_counts.get('中性', 0) / total * 100, 1) if total > 0 else 0
        negative_percentage = round(sentiment_counts.get('負面', 0) / total * 100, 1) if total > 0 else 0
        
        # 分析情感趋势
        trend_analysis = "穩定"
        if negative_percentage > 30:
            trend_analysis = "需要注意負面情緒較多"
        elif negative_percentage > 50:
            trend_analysis = "負面情緒佔主導，建議尋求專業協助"
        
        # 创建情感分布饼图
        st.subheader("情感分布分析")
        # 提取情感标签
        sentiment_labels = [item['sentiment'] for item in all_data]
        sentiment_counts = pd.Series(sentiment_labels).value_counts()
        
        if not sentiment_counts.empty:
            fig1 = px.pie(
                sentiment_counts, 
                values=sentiment_counts.values,
                names=sentiment_counts.index,
                title="情感分布比例",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig1.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("沒有足夠的情緒數據進行分析")
        
        # 按平台分析情感分布 - 修复空数据问题
        st.subheader("各平台情感分布")
        if all_data:
            df = pd.DataFrame(all_data)
            
            # 确保平台列存在
            if 'platform' in df.columns and 'sentiment' in df.columns:
                # 创建平台-情感的交叉表
                platform_sentiment = pd.crosstab(
                    df['platform'], 
                    df['sentiment']
                )
                
                # 确保所有平台都有相同的列数
                if not platform_sentiment.empty:
                    # 获取所有可能的情感值
                    all_sentiments = ['正面', '中性', '負面']
                    
                    # 添加缺失的情感列
                    for sentiment in all_sentiments:
                        if sentiment not in platform_sentiment.columns:
                            platform_sentiment[sentiment] = 0
                    
                    # 重新排序列
                    platform_sentiment = platform_sentiment[all_sentiments]
                    
                    # 重置索引
                    platform_sentiment = platform_sentiment.reset_index()
                    
                    # 转换数据为长格式
                    melted_data = platform_sentiment.melt(
                        id_vars=['platform'], 
                        value_vars=all_sentiments,
                        var_name='情感',
                        value_name='數量'
                    )
                    
                    # 创建条形图
                    fig2 = px.bar(
                        melted_data,
                        x='platform',
                        y='數量',
                        color='情感',
                        title="各平台情感分布",
                        labels={'platform': '平台', '數量': '貼文数量'},
                        barmode='group'
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("沒有足夠的平台數據進行分析")
            else:
                st.warning("數據中缺乏必要的平台或情感訊息")
        else:
            st.warning("沒有可用於平台分析的數據")
        
        # 按时间分析情感趋势 - 修复日期处理错误
        st.subheader("情緒趨勢分析")
        if all_data:
            df = pd.DataFrame(all_data)
            
            # 确保有日期字段
            if 'date' in df.columns and not df['date'].empty:
                try:
                    # 统一日期格式处理
                    def format_date(date_str):
                        try:
                            if isinstance(date_str, datetime):
                                return date_str
                            # 尝试多种日期格式
                            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
                                try:
                                    return datetime.strptime(str(date_str), fmt)
                                except ValueError:
                                    continue
                            return None
                        except:
                            return None
                    
                    # 应用日期格式化
                    df['formatted_date'] = df['date'].apply(format_date)
                    
                    # 移除无效日期
                    df = df.dropna(subset=['formatted_date'])
                    
                    if not df.empty:
                        # 转换日期格式
                        df['formatted_date'] = pd.to_datetime(df['formatted_date'])
                        df['date_only'] = df['formatted_date'].dt.date
                        
                        # 按日期分组统计
                        daily_sentiment = df.groupby(['date_only', 'sentiment']).size().unstack(fill_value=0)
                        
                        # 确保所有情感类别都存在
                        all_sentiments = ['正面', '中性', '負面']
                        for sentiment in all_sentiments:
                            if sentiment not in daily_sentiment.columns:
                                daily_sentiment[sentiment] = 0
                        
                        # 创建每日情感趋势图
                        fig3 = px.line(
                            daily_sentiment.reset_index(),  # 重置索引确保数据结构一致
                            x='date_only',
                            y=all_sentiments,
                            title="情緒變化趨勢",
                            labels={'value': '數量', 'date_only': '日期'},
                            markers=True
                        )
                        fig3.update_layout(
                            legend_title_text='情感類型',
                            xaxis_title='日期',
                            yaxis_title='數量'
                        )
                        st.plotly_chart(fig3, use_container_width=True)
                    else:
                        st.warning("沒有有效的日期數據")
                except Exception as e:
                    st.error(f"日期處理錯誤: {str(e)}")
            else:
                st.warning("數據中缺少日期信息")
        else:
            st.warning("沒有可用於趨勢分析的數據")
        
        # 情感关键词分析 - 修复空数据问题
        st.subheader("情緒關鍵字分析")
        col1, col2, col3 = st.columns(3)
        
        # 辅助函数：提取关键词
        def extract_keywords(items, sentiment_name):
            if not items:
                return None
                
            try:
                # 合并所有内容
                combined_text = " ".join([str(item.get('content', '')) for item in items])
                if not combined_text.strip():
                    return None
                
                # 提取关键词
                keywords = HanLP.keyphrase_extraction(combined_text)
                return keywords
            except Exception as e:
                st.error(f"{sentiment_name}關鍵字擷取失敗: {str(e)}")
                return None
        
        # 显示代表性内容 - 修复空数据问题
        st.subheader("代表性内容分析")
        tab1, tab2, tab3, tab4 = st.tabs(["正面内容", "中性内容", "負面内容", "特殊需求"])
        
        # 辅助函数：显示代表性内容
        def display_representative_items(items, sentiment_name):
            """顯示代表性內容的輔助函數"""
            if not items:
                st.info(f"没有找到{sentiment_name}内容")
                return
                    
            # 按内容长度排序（假设长内容更有代表性）
            items.sort(key=lambda x: len(str(x.get('content', ''))), reverse=True)
            for i, item in enumerate(items[:5]):  # 显示前5条
                st.markdown(f"**来源: {item.get('platform', '未知')}**")
                if item.get('date'):
                    st.write(f"📅 日期: {item.get('date')}")
                
                content = str(item.get('content', '無内容'))
                st.write(f"📝 内容: {content[:300]}{'...' if len(content) > 300 else ''}")
                
                # 显示标签
                tags = item.get('tags', [])
                if tags:
                    st.write(f"🏷️ 標籤: {', '.join(tags)}")
                
                if 'url' in item and item['url']:  # 确保URL存在且不为空
                    st.markdown(f"[查看原文]({item['url']})")
                
                # 敏感内容提供帮助资源
                if any(keyword in content for keyword in sensitive_keywords):
                    st.warning("""
                    🚨 重要提醒：
                    香港撒瑪利亞防止自殺會：+852 2389 2222
                    生命熱線：+852 2382 0000
                    我們非常關心您的安全！
                    """)
                
                st.divider()
        
        # 使用辅助函数显示各类内容
        with tab1:
            st.markdown("### 😊 最具代表性的正面内容")
            positive_items = [item for item in all_data if item.get('sentiment') == '正面']
            display_representative_items(positive_items, "正面")
        
        with tab2:
            st.markdown("### 😐 最具代表性的中性内容")
            neutral_items = [item for item in all_data if item.get('sentiment') == '中性']
            display_representative_items(neutral_items, "中性")
        
        with tab3:
            st.markdown("### 😔 最具代表性的負面内容")
            negative_items = [item for item in all_data if item.get('sentiment') == '負面']
            display_representative_items(negative_items, "負面")
            
        # 新增: 特殊需求标签页
        with tab4:
            st.markdown("### 🤝 特殊需求内容")
            special_items = []
            
            # 筛选有特殊标签的内容
            for item in all_data:
                tags = item.get('tags', [])
                if '交友需求' in tags or '敏感内容' in tags:
                    special_items.append(item)
            
            if special_items:
                for i, item in enumerate(special_items):
                    st.markdown(f"**内容 {i+1}**")
                    st.write(f"📅 日期: {item.get('date', '未知')}")
                    
                    content = str(item.get('content', '無内容'))
                    st.write(f"📝 内容: {content[:300]}{'...' if len(content) > 300 else ''}")
                    
                    # 显示标签
                    tags = item.get('tags', [])
                    if tags:
                        st.write(f"🏷️ 標籤: {', '.join(tags)}")
                    
                    # 交友需求显示社区资源
                    if '交友需求' in tags:
                        st.info("""
                        🤝 社區中心資源：
                        - 香港青年協會：https://www.hkfyg.org.hk/
                        - 明愛社區中心：https://www.caritas.org.hk/
                        - 香港遊樂場協會：https://www.hkpa.hk/
                        - 東華三院社區中心：https://www.tungwah.org.hk/
                        """)
                    
                    # 敏感内容提供帮助资源
                    if any(keyword in content for keyword in sensitive_keywords):
                        st.warning("""
                        🚨 重要提醒：
                        香港撒瑪利亞防止自殺會：+852 2389 2222
                        生命熱線：+852 2382 0000
                        我們非常關心您的安全！
                        """)
                    
                    st.divider()
            else:
                st.info("没有找到特殊需求内容")
                
        st.subheader("✨ 心理健康建議")
        
        # 创建报告摘要
        report_summary = f"""
        ## 情感分析摘要
        - 正面情緒: {positive_percentage}%
        - 中性情緒: {neutral_percentage}%
        - 負面情緒: {negative_percentage}%
        
        ## 主要趨勢
        {trend_analysis}
        
        ## 主要觀察
        {f"檢測到{len(special_items)}條特殊需求內容" if 'special_items' in locals() else "未檢測到特殊需求內容"}
        """
        
        # 生成AI建议
        with st.spinner("正在生成專業建議..."):
            advice = generate_advice(report_summary)
            formatted_advice = f"## 📝 專業建議\n\n{advice}"
            st.markdown(formatted_advice, unsafe_allow_html=True)
            st.text(advice)
        
        # 添加本地资源推荐
        st.subheader("🏥 本地支援資源")
        resources = generate_local_resources(negative_percentage)
        
        # 使用tabs替代嵌套的expander
        resource_tabs = st.tabs(list(resources.keys()))
        
        for tab, (category, items) in zip(resource_tabs, resources.items()):
            with tab:
                for item in items:
                    st.write(f"- {item}")
        threading.Thread(target=text_to_speech, args=("報告完成了，快點去看看吧", 'zh-TW', 'gentle_female', True)).start()
        
        # 保存完整分析报告
        if st.button("💾 保存完整報告", key="save_report_button"):
            # 创建保存目录
            if not os.path.exists("./reports"):
                os.makedirs("./reports")
            
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./reports/full_report_{timestamp}.json"
            
            try:
                # 保存数据
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(all_data, f, indent=4, ensure_ascii=False)
                
                st.success(f"完整報告已保存到: {filename}")
            except Exception as e:
                st.error(f"保存報告失敗: {str(e)}")
    
    except Exception as e:
        st.error(f"產生綜合報告時發生錯誤: {str(e)}")
        import traceback
        st.text(traceback.format_exc())  # 打印完整堆栈跟踪

def sister_style_transform(text):
    try:
        # 使用大模型自身的能力进行风格转换
        prompt = (
            "嗨，我是你的知心姐姐。让我用温和的语气来陪伴你：\n"
            f"{text}"
        )
        
        # 调用模型进行风格转换
        styled_response, _ = model.chat(
            tokenizer, 
            query=prompt, 
            history=None, 
            max_length=2048
        )
        
        return styled_response
    except Exception as e:
        print(f"風格轉換失敗: {str(e)}")
        return text  # 失败时返回原始文本

def convert_to_traditional_chinese(text):
    cc = OpenCC('s2t')  # 将简体中文转换为繁体中文
    traditional_text = cc.convert(text)
    return traditional_text

def play_audio(file_path):
    """播放音訊檔案"""
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
    except Exception as e:
        print(f"音訊播放失敗: {str(e)}")
    finally:
        try:
            os.remove(file_path)
        except:
            pass

def online_tts(text, lang='zh-TW', slow=False, pitch=1.0):
    """
    使用 Google TTS 產生溫柔女聲（需要網路連線）
    
     參數:
     text: 要轉換的文本
     lang: 語言代碼 (zh-TW: 繁體中文, zh-CN: 簡體中文)
     slow: 是否放慢語速
     pitch: 音調調整 (1.0 正常, >1.0 更高, <1.0 更低)
     """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as fp:
            temp_file = fp.name
        
        # 使用 gTTS 生成语音
        tts = gTTS(text=text, lang=lang, slow=slow)
        tts.save(temp_file)
        return temp_file
    except Exception as e:
        print(f"線上TTS失敗: {str(e)}")
        return None
        
        return temp_file
    except Exception as e:
        print(f"線上TTS失敗: {str(e)}")
        return None
    
def offline_tts(text, lang='zh-tw', pitch=110, rate=150, volume=0.9):
    """
     使用 pyttsx3 離線產生語音
    
     參數:
     text: 要轉換的文本
     lang: 語言程式碼
     pitch: 音 (50-200, 預設110)
     rate: 語速 (100-300, 預設150)
     volume: 音量 (0.0-1.0, 預設0.9)
     """
    try:
        # 创建临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            temp_file = fp.name
        
        # 初始化引擎
        engine = pyttsx3.init()
        
        # 尝试设置女性声音
        voices = engine.getProperty('voices')
        female_voices = []
        
        for voice in voices:
            if 'female' in voice.name.lower() or 'woman' in voice.name.lower():
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    engine.setProperty('voice', voice.id)
                    break
                else:
                    female_voices.append(voice)
        
        # 如果没有找到中文女性声音，使用第一个女性声音
        if not engine.getProperty('voice') and female_voices:
            engine.setProperty('voice', female_voices[0].id)
        
        # 设置语音参数
        engine.setProperty('rate', rate)
        engine.setProperty('volume', volume)
        engine.setProperty('pitch', pitch)  # 调整音调
        
        # 保存到临时文件
        engine.save_to_file(text, temp_file)
        engine.runAndWait()
        
        return temp_file
    except Exception as e:
        print(f"離線TTS失敗: {str(e)}")
        return None
    
def text_to_speech(text, lang='zh-TW', voice_type='gentle_female', online_first=True):
    """
     文字轉語音主函數
    
     參數:
     text: 要轉換的文本
     lang: 語言程式碼
     voice_type: 聲音類型 (gentle_female: 溫柔女聲)
     online_first: 是否優先使用線上服務
     """
    if not text:
        return
    
    # 温柔女声的参数设置
    if voice_type == 'gentle_female':
        online_params = {'slow': 0, 'pitch': 1.1}  # 稍慢的语速和稍高的音调
        offline_params = {'pitch': 110, 'rate': 150, 'volume': 0.9}
    
    try:
        # 优先尝试在线服务
        if online_first and check_internet_connection():
            audio_file = online_tts(text, lang=lang, **online_params)
            if audio_file:
                play_audio(audio_file)
                return
    except:
        pass
    
    # 在线服务失败或不可用时使用离线服务
    audio_file = offline_tts(text, lang=lang, **offline_params)
    if audio_file:
        play_audio(audio_file)

def check_internet_connection(url="http://www.google.com", timeout=3):
    """檢查網路連線是否可用"""
    try:
        requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False

def answer(user_history, bot_history, sample=True, top_p=0.75, temperature=0.95):
    '''sample：是否抽樣。生成任務，可以設定為True;
     top_p：0-1之間，產生的內容越多樣化
    max_new_tokens=512 lost...'''

    if len(bot_history)>0:
        dialog_turn = 5 # 设置历史对话轮数
        if len(bot_history)>dialog_turn:
            bot_history = bot_history[-dialog_turn:]
            user_history = user_history[-(dialog_turn+1):]
        
        context = "\n".join([f"用户：{user_history[i]}\n心理諮詢師：{bot_history[i]}" for i in range(len(bot_history))])
        input_text = context + "\n用户：" + user_history[-1] + "\n心理諮商師："
    else:
        input_text = "用户：" + user_history[-1] + "\n心理諮商師："
        return "你好！我是你的個人專屬數位輔導老師甜心老師，歡迎找我傾訴、談心，期待幫助你！"
    
    print(input_text)
    if not sample:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=False, top_p=top_p, temperature=temperature, logits_processor=None)
    else:
        response, history = model.chat(tokenizer, query=input_text, history=None, max_length=2048, num_beams=1, do_sample=True, top_p=top_p, temperature=temperature, logits_processor=None)

    Traditionalresponse = convert_to_traditional_chinese(response)
    print("模型原始输出：\n", Traditionalresponse)
    
    Traditionalresponse_ex = sister_style_transform(Traditionalresponse)
    Finalresponse = re.sub("\n+", "\n", Traditionalresponse_ex)
    print('心理諮商師: '+Finalresponse)
    FinalTraditionalresponse = convert_to_traditional_chinese(Finalresponse)
    return Traditionalresponse + FinalTraditionalresponse
    
if 'show_comprehensive_report' in st.session_state and st.session_state.show_comprehensive_report:
    st.session_state.report_showing = True 
    st.session_state.report_displayed = True  # 新增：设置报告显示状态
    
    # 使用容器确保报告显示完整
    with st.container():
        with st.expander("📊 綜合情感分析報告", expanded=True):
            generate_comprehensive_report()
            st.session_state.show_comprehensive_report = False
else:
    # 当报告不再显示时，重置报告显示状态
    st.session_state.report_showing = False
    st.session_state.report_displayed = False  # 新增：重置报告显示状态
    
    # 重置状态，避免重复显示
    st.session_state.show_comprehensive_report = False

# 初始化 session_state 中的 'id'
if 'id' not in st.session_state:
    # 创建保存用户聊天记录的目录
    if not os.path.exists("./history"):
        os.makedirs("./history")
    json_files = os.listdir("./history")
    id = len(json_files)
    st.session_state['id'] = id
    
    
with st.sidebar:
    InstagramFind = "正在獲取Instagram內容，請你等一下唷"
    TwitterFind = "正在獲取Twitter內容，請你等一下唷"
    st.header("🌐 社群媒體數據分析")
    
    # 将平台选择移出表单
    platform = st.selectbox(
        "選擇社交平台", 
        ["Instagram", "Twitter"],
        key="platform_select",
        on_change=lambda: st.session_state.update(platform_changed=True)
    )
    
    # 检查平台是否已更改
    if 'platform_changed' in st.session_state and st.session_state.platform_changed:
        st.session_state.platform_changed = False
        st.session_state.form_cleared = True
        st.session_state.skip_audio = True
    
    # 使用表单容器
    with st.form("social_media_form"):
        # 显示当前选择的平台
        st.write(f"當前選擇平台: **{platform}**")
        
        account_id = st.text_input("輸入帳號ID", help="例如：Instagram帳號或Twitter用户名")
        
        # 平台特定设置 - 动态显示
        if platform == "Instagram":
            download_num = st.number_input("獲取最近幾條貼文", min_value=1, max_value=50, value=10)
        else:
            start_date = st.date_input("開始日期", value=datetime(2024, 1, 1))
            end_date = st.date_input("結束日期", value=datetime.today())
        
        # 分析按钮
        analyze_submitted = st.form_submit_button("🔍 分析帳號內容", 
                                                 help="點擊開始抓取和分析社群媒體內容")
        
        # 生成报告按钮
        report_submitted = st.form_submit_button("📊 生成综合報告", 
                                                help="點擊產生整合社群媒體和使用者對話的綜合報告")
        
    if 'social_data' not in st.session_state:
        st.session_state.social_data = []
    # 处理分析按钮点击
    if analyze_submitted:
        if not account_id:
            st.sidebar.warning("請輸入帳號ID")
        else:
            if platform == "Instagram":
                with st.spinner(InstagramFind):
                    threading.Thread(target=text_to_speech, args=(InstagramFind, 'zh-TW', 'gentle_female', True)).start()
                    try:
                        fetcher = InstagramDataFetcher(account_id, download_num)
                        social_data = fetcher.fetch_data()
                        
                        # 提取文本内容进行情感分析
                        captions = [item['content'] for item in social_data]
                        if captions:
                            sentiment_result = HanLP.sentiment_analysis(captions)
                            st.session_state.sentiment_result = sentiment_result
                        
                        # 统一存储到session_state
                        st.session_state.social_data.extend(social_data)
                        
                        # 添加情感标签到每条数据
                        for i, item in enumerate(social_data):
                            if i < len(sentiment_result):
                                item['sentiment'] = sentiment_result[i]
                            else:
                                item['sentiment'] = '未知'
                        instagramdone = (f"成功拿到{len(social_data)}條Instagram內容！已加入到現有資料啦。")
                        st.success(instagramdone)
                        threading.Thread(target=text_to_speech, args=(instagramdone, 'zh-TW', 'gentle_female', True)).start()
                    except Exception as e:
                        st.error(f"Instagram數據獲取失敗: {str(e)}")
            
            # Twitter部分
            elif platform == "Twitter":
                with st.spinner(TwitterFind):
                    threading.Thread(target=text_to_speech, args=(TwitterFind, 'zh-TW', 'gentle_female', True)).start()
                    try:
                        scraper = TwitterExtractor(TWITTER_AUTH_TOKEN)
                        social_data = scraper.fetch_tweets(
                            f"https://twitter.com/{account_id}/likes",
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d")
                        )
                        
                        for item in social_data:
                            item['platform'] = 'Twitter'
                            
                            # 使用翻译文本作为主要内容
                            # 如果翻译文本存在且不是空字符串，则使用翻译文本
                            if 'translated_text' in item and item['translated_text']:
                                item['content'] = item['translated_text']
                            else:
                                # 否则使用原始文本
                                item['content'] = item.get('text', '')
                        
                        # 提取文本内容进行情感分析 - 使用 content 字段
                        contents = [item.get('content', '') for item in social_data]
                        if contents:
                            sentiment_result = HanLP.sentiment_analysis(contents)
                            st.session_state.sentiment_result = sentiment_result
                        
                        st.session_state.social_data.extend(social_data)
                        
                        # 添加情感标签到每条数据
                        for i, item in enumerate(social_data):
                            if i < len(sentiment_result):
                                item['sentiment'] = sentiment_result[i]
                            else:
                                item['sentiment'] = '未知'
                        
                        Twitterdone = (f"成功拿到{len(social_data)}條Twitter內容！已加入到現有資料啦。")
                        st.success(Twitterdone)
                        threading.Thread(target=text_to_speech, args=(Twitterdone, 'zh-TW', 'gentle_female', True)).start()
                    except Exception as e:
                        st.error(f"Twitter數據獲取失敗: {str(e)}")
            
    # 处理报告按钮点击 - 独立的事件处理
    generatereporttext = "報告生成中，等我一下吧，等等就可以看報告了啦"
    if report_submitted:
        # 确保有数据可分析
        if 'social_data' in st.session_state or 'user_inputs' in st.session_state:
            # 设置报告显示状态
            st.session_state.show_comprehensive_report = True
            
            # 显示提示信息
            st.sidebar.success(generatereporttext)
            
            # 播放语音提示
            threading.Thread(target=text_to_speech, args=(generatereporttext, 'zh-TW', 'gentle_female', True)).start()
            
            # 强制重新运行以显示报告
            st.experimental_rerun()
        else:
            st.sidebar.warning("沒有可分析的數據，請先分析社群媒體或進行對話")


if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True
    # 首次访问播放欢迎语音
    threading.Thread(target=text_to_speech, args=("歡迎來到心靈小幫手，很開心你來找我聊天唷", 'zh-TW', 'gentle_female', True)).start()

# 主标题
st.header("心靈小幫手")
with st.expander("ℹ️ - 關於我們", expanded=False):
    st.write(
        """     
-   版本：心靈小幫手
-   作者：吳世杰
	    """
    )

# https://docs.streamlit.io/library/api-reference/performance/st.cache_resource


if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

# 新增: 初始化用户输入存储
if 'user_inputs' not in st.session_state:
    st.session_state['user_inputs'] = []

user_col, ensure_col = st.columns([5, 1])

def get_text():
    """取得使用者輸入文字的函數"""
    input_text = user_col.text_area("請在下列文字方塊輸入您的諮詢內容：","", key="input", placeholder="請輸入您的求助內容，並且點擊傳送按鈕")
    
    # 检查敏感词但不中断流程
    if any(keyword in input_text for keyword in sensitive_keywords):
        # 显示紧急帮助信息
        st.warning("""
        🚨 偵測到您可能需要緊急協助：
        香港撒瑪利亞防止自殺會：+852 2389 2222
        生命熱線：+852 2382 0000
        我們非常關心您的安全！
        """)
        
    if ensure_col.button("傳送", key="send_button", use_container_width=True):
        if input_text:
            return input_text  
    return None  # 确保没有点击时返回None

# 用户输入处理部分
user_input = None  # 初始化变量
user_input = get_text()  # 获取用户输入

if user_input is not None:  # 确保 user_input 被定义后再使用
    # 检测交友需求
    if any(phrase in user_input for phrase in friendship_keywords):
        # 显示社区中心资源
        st.info("""
        🤝 我們注意到您想認識新朋友，以下是一些社區中心的資訊：
        - 香港青年協會：https://www.hkfyg.org.hk/
        - 明愛社區中心：https://www.caritas.org.hk/
        - 香港遊樂場協會：https://www.hkpa.hk/
        - 東華三院社區中心：https://www.tungwah.org.hk/
        希望您能找到志同道合的朋友！
        """)
    
    # 新增: 保存用户输入时添加标签
    tags = []
    
    # 检查敏感词
    if any(keyword in user_input for keyword in sensitive_keywords):
        tags.append('敏感内容')
    
    # 检查交友需求
    if any(phrase in user_input for phrase in friendship_keywords):
        tags.append('交友需求')
    
    # 保存用户输入用于情感分析
    st.session_state.user_inputs.append({
        'content': user_input,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'tags': tags  # 新增标签字段
    })
    
    st.session_state.past.append(user_input)
    output = answer(st.session_state['past'], st.session_state["generated"])
    st.session_state.generated.append(output)
    
    # 在回复中再次添加紧急帮助信息（如果包含敏感词）
    if any(keyword in user_input for keyword in sensitive_keywords):
        # 在原始回复后添加紧急帮助信息
        emergency_note = (
            "\n\n❤️ 我們非常關心您的安全！"
            "\n如果您需要緊急協助，請立即聯繫："
            "\n香港撒瑪利亞防止自殺會：+852 2389 2222"
            "\n生命熱線：+852 2382 0000"
            "\n您並不孤單，我們都在這裡支持您！"
        )
        st.session_state.generated[-1] += emergency_note
    
    # 将对话历史保存成json文件
    dialog_history = {
        'user': st.session_state['past'],
        'bot': st.session_state["generated"]
    }
    # 确保 'id' 已初始化
    if 'id' in st.session_state:
        with open(os.path.join("./history", str(st.session_state['id'])+'.json'), "w", encoding="utf-8") as f:
            json.dump(dialog_history, f, indent=4, ensure_ascii=False)
    else:
        st.error("會話ID未初始化，無法儲存對話歷史")
    with open(os.path.join("./history", str(st.session_state['id'])+'.json'), "w", encoding="utf-8") as f:
        json.dump(dialog_history, f, indent=4, ensure_ascii=False)

if 'welcome_played' not in st.session_state:
    st.session_state.welcome_played = False

if (st.session_state['generated'] 
    and not st.session_state.get('show_comprehensive_report', False)
    and not st.session_state.get('form_submitted', False)
    and not st.session_state.get('report_showing', False)
    and not st.session_state.get('report_displayed', False)):
    
    # 显示所有历史消息
    for i in range(len(st.session_state['generated'])):
        if i == 0 and not st.session_state.welcome_played:
            # 首次对话时显示欢迎消息
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            message("你好！我是你的個人專屬數位輔導員甜心老師，歡迎找我傾訴、談心❤️，期待幫助你！", key=str(i), avatar_style="avataaars", seed=5)
            
            # 播放欢迎消息并标记为已播放
            if not st.session_state.welcome_played:
                threading.Thread(target=text_to_speech, args=("你好！我是你的個人專屬數位輔導員甜心老師，歡迎找我傾訴、談心❤️，期待幫助你！", 'zh-TW', 'gentle_female', True)).start()
                st.session_state.welcome_played = True
        else:
            # 显示其他对话消息
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed=26)
            message(st.session_state["generated"][i], key=str(i), avatar_style="avataaars", seed=5)
    
    # 只播放最新的回复
    if (len(st.session_state['generated']) > 0 
        and not st.session_state.get('show_comprehensive_report', False)
        and not st.session_state.get('form_submitted', False)
        and not st.session_state.get('report_showing', False)
        and not st.session_state.get('report_displayed', False)):
        
        if not st.session_state.get('skip_audio', False):
            latest_response = st.session_state["generated"][-1]
            threading.Thread(target=text_to_speech, args=(latest_response, 'zh-TW', 'gentle_female', True)).start()
        else:
            # 重置跳过标志
            st.session_state.skip_audio = False

if st.button("清理對話快取"):
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['user_inputs'] = []
    st.success("對話快取已清理")