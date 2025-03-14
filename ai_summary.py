import os
import json
import requests
from typing import Dict, Any, Optional, List

class AIModelManager:
    """AI模型管理器"""
    
    def __init__(self, model_type="ollama", model_name="llama3", api_url=None, api_key=None):
        """
        初始化AI模型管理器
        
        Args:
            model_type: 模型类型，支持 'ollama' 和 'custom'
            model_name: 模型名称
            api_url: API URL
            api_key: API密钥
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_url = api_url
        self.api_key = api_key
    
    def test_connection(self):
        """测试与AI模型的连接"""
        try:
            if self.model_type == "ollama":
                return self._test_ollama_connection()
            elif self.model_type == "custom":
                return self._test_custom_connection()
            else:
                print(f"不支持的模型类型: {self.model_type}")
                return False
        except Exception as e:
            print(f"测试连接时出错: {str(e)}")
            return False
    
    def _test_ollama_connection(self):
        """测试与Ollama的连接"""
        import requests
        
        if not self.api_url:
            self.api_url = "http://localhost:11434"
        
        try:
            # 构建API请求
            url = f"{self.api_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": "Hello, are you working?",
                "stream": False
            }
            
            # 发送请求
            response = requests.post(url, json=payload)
            
            # 检查响应
            if response.status_code == 200:
                return True
            else:
                print(f"Ollama API返回错误: {response.status_code}")
                print(response.text)
                return False
        
        except Exception as e:
            print(f"连接Ollama时出错: {str(e)}")
            return False
    
    def _test_custom_connection(self):
        """测试与自定义API的连接"""
        import requests
        
        if not self.api_url:
            print("未指定API URL")
            return False
        
        try:
            # 构建API请求
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": "Hello, are you working?"}
                ]
            }
            
            # 发送请求
            response = requests.post(self.api_url, json=payload, headers=headers)
            
            # 检查响应
            if response.status_code == 200:
                return True
            else:
                print(f"自定义API返回错误: {response.status_code}")
                print(response.text)
                return False
        
        except Exception as e:
            print(f"连接自定义API时出错: {str(e)}")
            return False
    
    def generate_summary(self, text, max_length=200, language="chinese"):
        """
        生成文本摘要
        
        Args:
            text: 要摘要的文本
            max_length: 摘要的最大长度
            language: 摘要的语言，'chinese'或'english'
        
        Returns:
            生成的摘要文本
        """
        if self.model_type == "ollama":
            return self._generate_summary_ollama(text, max_length, language)
        elif self.model_type == "custom":
            return self._generate_summary_custom(text, max_length, language)
        else:
            print(f"不支持的模型类型: {self.model_type}")
            return None
    
    def _generate_summary_ollama(self, text, max_length=200, language="chinese"):
        """使用Ollama生成摘要"""
        import requests
        
        if not self.api_url:
            self.api_url = "http://localhost:11434"
        
        # 构建提示
        lang_text = "中文" if language == "chinese" else "English"
        prompt = f"""请为以下文本生成一个简洁的{lang_text}摘要，不超过{max_length}个字符:

{text}

摘要:"""
        
        try:
            # 构建API请求
            url = f"{self.api_url}/api/generate"
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # 发送请求
            response = requests.post(url, json=payload)
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                print(f"Ollama API返回错误: {response.status_code}")
                print(response.text)
                return None
        
        except Exception as e:
            print(f"生成摘要时出错: {str(e)}")
            return None
    
    def _generate_summary_custom(self, text, max_length=200, language="chinese"):
        """使用自定义API生成摘要"""
        import requests
        
        if not self.api_url:
            print("未指定API URL")
            return None
        
        # 构建提示
        lang_text = "中文" if language == "chinese" else "English"
        content = f"""请为以下文本生成一个简洁的{lang_text}摘要，不超过{max_length}个字符:

{text}

摘要:"""
        
        try:
            # 构建API请求
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": content}
                ]
            }
            
            # 发送请求
            response = requests.post(self.api_url, json=payload, headers=headers)
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                
                # 尝试从不同的响应格式中提取内容
                if "choices" in result and len(result["choices"]) > 0:
                    # OpenAI格式
                    message = result["choices"][0].get("message", {})
                    return message.get("content", "").strip()
                elif "response" in result:
                    # Ollama格式
                    return result.get("response", "").strip()
                else:
                    # 其他格式
                    return str(result).strip()
            else:
                print(f"自定义API返回错误: {response.status_code}")
                print(response.text)
                return None
        
        except Exception as e:
            print(f"生成摘要时出错: {str(e)}")
            return None 