"""
邮箱验证码服务
使用SMTP发送验证码邮件
"""
from flask import Flask, request, jsonify
import smtplib
import random
import string
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

app = Flask(__name__)

# 添加CORS配置
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# 邮件配置 - 清华大学邮箱配置
EMAIL_CONFIG = {
    'smtp_server': 'smtp.tsinghua.edu.cn',  # 清华大学SMTP服务器地址
    'smtp_port': 465,  # SMTP端口 (使用SSL)
    'sender_email': 'zongqr24@mails.tsinghua.edu.cn',  # 发件人邮箱
    'sender_password': 'cRhKiJzCjiRZ6meU',  # 客户端专用密码
    'use_tls': False,  # 不使用TLS
    'use_ssl': True,  # 使用SSL
    'use_mock': False  # 真实模式
}

# 存储验证码和发送时间
verification_codes = {}

def generate_code():
    """生成6位验证码"""
    return ''.join(random.choices(string.digits, k=6))

def send_verification_email(recipient_email, code):
    """发送验证码邮件"""
    try:
        # 调试信息
        print(f"当前 use_mock 设置: {EMAIL_CONFIG['use_mock']}")
        # 模拟模式 - 不发送真实邮件
        if EMAIL_CONFIG['use_mock']:
            print(f"[模拟模式] 验证码 {code} 已发送到邮箱: {recipient_email}")
            return True, "验证码已发送到您的邮箱"

        msg = MIMEMultipart()
        msg['From'] = Header(f"R2P-Guard <{EMAIL_CONFIG['sender_email']}>")
        msg['To'] = recipient_email
        msg['Subject'] = Header('R2P-Guard 注册验证码', 'utf-8')

        # 邮件正文
        body = f'''
        <html>
        <body>
            <h2>您好！</h2>
            <p>感谢您注册 R2P-Guard 运动员状态管理系统。</p>
            <p>您的验证码是：<strong style="font-size: 24px; color: #0071e3;">{code}</strong></p>
            <p>验证码有效期为5分钟，请勿将验证码告诉他人。</p>
            <br>
            <p>如果您没有发起注册请求，请忽略此邮件。</p>
            <p>此邮件由系统自动发送，请勿直接回复。</p>
        </body>
        </html>
        '''
        msg.attach(MIMEText(body, 'html', 'utf-8'))

        # 连接SMTP服务器
        if EMAIL_CONFIG['use_ssl']:
            server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
        else:
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            if EMAIL_CONFIG['use_tls']:
                server.starttls()

        # 登录SMTP服务器
        print(f"尝试登录SMTP服务器: {EMAIL_CONFIG['smtp_server']}")
        print(f"发件人邮箱: {EMAIL_CONFIG['sender_email']}")
        print(f"授权码长度: {len(EMAIL_CONFIG['sender_password'])}")
        server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        print("登录成功")

        # 发送邮件
        text = msg.as_string()
        server.sendmail(EMAIL_CONFIG['sender_email'], recipient_email, text)
        print(f"邮件发送成功到: {recipient_email}")

        # 关闭连接
        server.quit()
        print("SMTP连接已关闭")

        return True, "发送成功"
    except Exception as e:
        print(f"发送邮件失败: {str(e)}")
        return False, str(e)

@app.route('/send_code', methods=['POST'])
def send_code():
    """发送验证码到指定邮箱"""
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({'success': False, 'message': '请提供邮箱地址'}), 400

    email = data['email'].strip().lower()

    # 验证邮箱格式
    if not email or '@' not in email:
        return jsonify({'success': False, 'message': '请输入有效的邮箱地址'}), 400

    # 检查发送频率（60秒内只能发送一次）
    if email in verification_codes:
        last_send_time = verification_codes[email].get('send_time', 0)
        if time.time() - last_send_time < 60:
            remaining_time = int(60 - (time.time() - last_send_time))
            return jsonify({
                'success': False,
                'message': f'请{remaining_time}秒后再试'
            }), 429

    # 生成验证码
    code = generate_code()

    # 发送邮件
    success, message = send_verification_email(email, code)

    if success:
        # 存储验证码和发送时间
        verification_codes[email] = {
            'code': code,
            'send_time': time.time(),
            'verified': False
        }
        return jsonify({'success': True, 'message': '验证码已发送'})
    else:
        return jsonify({'success': False, 'message': f'发送失败: {message}'}), 500

@app.route('/verify_code', methods=['POST'])
def verify_code():
    """验证验证码"""
    data = request.get_json()

    if not data or 'email' not in data or 'code' not in data:
        return jsonify({'success': False, 'message': '请提供邮箱和验证码'}), 400

    email = data['email'].strip().lower()
    code = data['code'].strip()

    if email not in verification_codes:
        return jsonify({'success': False, 'message': '请先发送验证码'}), 400

    stored_data = verification_codes[email]

    # 检查验证码是否过期（5分钟）
    if time.time() - stored_data['send_time'] > 300:
        return jsonify({'success': False, 'message': '验证码已过期，请重新获取'}), 400

    # 检查验证码是否正确
    if stored_data['code'] != code:
        return jsonify({'success': False, 'message': '验证码错误'}), 400

    # 标记为已验证
    verification_codes[email]['verified'] = True

    return jsonify({'success': True, 'message': '验证成功'})

@app.route('/check_email', methods=['POST'])
def check_email():
    """检查邮箱是否已注册"""
    data = request.get_json()

    if not data or 'email' not in data:
        return jsonify({'exists': False}), 400

    email = data['email'].strip().lower()

    # 这个接口需要配合前端存储的用户数据使用
    # 这里只是示例，实际应该查询数据库
    return jsonify({'exists': False, 'email': email})

if __name__ == '__main__':
    print("=" * 50)
    print("邮箱验证码服务")
    print("=" * 50)
    print(f"发件人邮箱: {EMAIL_CONFIG['sender_email']}")
    print(f"SMTP服务器: {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
    print("")
    print("重要提示：")
    print("请确保已获取邮箱授权码，而不是登录密码！")
    print("")
    print("163邮箱获取授权码方法：")
    print("1. 登录163邮箱网页版")
    print("2. 点击 设置 -> POP3/SMTP/IMAP")
    print("3. 开启SMTP服务")
    print("4. 获取授权码")
    print("")
    print("获取授权码后，编辑文件中的 YOUR_AUTH_CODE")
    print("")
    print("启动服务: python email_server.py")
    print("服务地址: http://127.0.0.1:8001")
    print("=" * 50)
    app.run(host='127.0.0.1', port=8001, debug=True)