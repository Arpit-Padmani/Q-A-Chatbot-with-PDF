css = '''<style>
.chat-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    max-width: 600px;
    margin: 0 auto;
}

.chat-message {
    display: flex;
    align-items: flex-start;
    padding: 1rem;
    margin-bottom: 1.2rem;
    border-radius: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.10);
    transition: all 0.3s ease;
}

.chat-message.bot {
    background: #f4f6f8;
    border-left: 4px solid #3b82f6;
}

.chat-message.user {
    background: #fff;
    border-left: 4px solid #10b981;
}

.chat-message .avatar {
    margin-right: 1rem;
}

.chat-message .avatar img {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #e2e8f0;
}

.chat-message .message {
    color: #1f2937;
    font-size: medium;
    line-height: 1.6;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/8943/8943377.png" style="max-height: 78px; max-width: 60px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://icons.veryicon.com/png/o/miscellaneous/user-avatar/user-avatar-male-5.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''