# 贡献指南

感谢您考虑为语音转文本工具项目做出贡献！以下是一些指导方针，以帮助您为项目做出贡献。

## 行为准则

请尊重所有参与者，保持专业和友好的交流环境。

## 如何贡献

1. **报告Bug**：
   - 使用GitHub Issues提交bug报告
   - 清晰描述问题，包括复现步骤
   - 如可能，提供截图或错误日志

2. **提交功能请求**：
   - 使用GitHub Issues提交功能请求
   - 清晰描述您希望添加的功能及其价值

3. **提交代码**：
   - Fork仓库
   - 创建新分支 (`git checkout -b feature/your-feature`)
   - 提交更改 (`git commit -m 'Add some feature'`)
   - 推送到分支 (`git push origin feature/your-feature`)
   - 创建Pull Request

## 开发流程

1. **设置开发环境**：
   ```bash
   git clone https://github.com/用户名/语音转文本工具.git
   cd 语音转文本工具
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **代码风格**：
   - 遵循PEP 8编码规范
   - 使用有意义的变量名和函数名
   - 添加适当的注释和文档字符串

3. **测试**：
   - 为新功能编写测试
   - 确保所有测试通过

4. **提交信息规范**：
   ```
   <类型>: <描述>

   [可选的正文]

   [可选的脚注]
   ```
   
   类型包括：
   - feat: 新功能
   - fix: 修复bug
   - docs: 文档更新
   - style: 代码风格更改（不影响代码运行）
   - refactor: 代码重构
   - test: 添加测试
   - chore: 构建过程或辅助工具的变动

## 审核流程

所有提交的Pull Request将由维护者审核。可能会要求进行更改以确保代码质量和一致性。

感谢您的贡献！ 